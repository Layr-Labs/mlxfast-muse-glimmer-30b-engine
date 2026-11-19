// The worker must not end the process while an engine queue is still working.
//
// Three properties, each proved here rather than counted:
//
// 1. ORDER. A drain is accepted while the loop's own work runs on (a chained
//    decode). The worker's second shutdown queues behind that work, so the
//    phase close cannot answer until the work is done. The mock models exactly
//    that: a successor step on the engine's own serial queue, and a second
//    shutdown that queues behind it.
// 2. NO PATH ESCAPES. An engine the worker built is stopped and waited for on
//    the error paths too.
// 3. A DRAIN THAT TIMED OUT IS NOT PROOF. A shutdown that ran into the loop's
//    watchdog leaves its queue possibly inside the runtime, so that worker may
//    not run the runtime's teardown at all.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXRunners

@Suite("the engine queue is quiesced before the worker ends")
struct BenchWorkerQueueQuiescenceTests {
    /// What the worker asked of the engines this run built.
    private final class EngineProbe: @unchecked Sendable {
        private let lock = NSLock()
        private var seen: [MockEngine] = []

        func record(_ engine: MockEngine) {
            lock.lock()
            seen.append(engine)
            lock.unlock()
        }

        var shutdowns: Int {
            lock.lock()
            defer { lock.unlock() }
            return seen.reduce(0) { $0 + $1.shutdowns }
        }
    }

    /// A flag two threads can share.
    private final class Flag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false

        func set() {
            lock.lock()
            value = true
            lock.unlock()
        }

        var isSet: Bool {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    /// The end-process codes a test lets the worker record instead of ending.
    private final class CodeBox: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: [Int32] = []

        func append(_ code: Int32) {
            lock.lock()
            stored.append(code)
            lock.unlock()
        }

        var codes: [Int32] {
            lock.lock()
            defer { lock.unlock() }
            return stored
        }
    }

    private struct SubmitRefused: Error {}

    /// A semaphore wait for a synchronous check. `wait` is not available from an
    /// async context, so the wait lives in a synchronous helper.
    private func signalled(_ semaphore: DispatchSemaphore, within seconds: Int = 30) -> Bool {
        semaphore.wait(timeout: .now() + .seconds(seconds)) == .success
    }

    private func makeServer(
        runner: MockRunner,
        transport: ScriptedTransport,
        probe: EngineProbe,
        shutdownTimeout: TimeInterval = CBv2EngineLoopConfig().shutdownTimeout
    ) -> BenchWorkerServer {
        runner.onEngine = { probe.record($0) }
        return BenchWorkerServer(
            runner: runner,
            transport: transport,
            trusted: false,
            speculative: true,
            build: "fixture",
            device: "mock",
            kvBytesCapacity: 1 << 20,
            maxDecodeTokens: 64,
            memory: FixtureMemoryReporter(),
            nonce: "fixturenonce",
            engineShutdownTimeout: shutdownTimeout)
    }

    // MARK: Ordering

    @Test("a phase close does not answer while the loop's successor step is running")
    func phaseCloseWaitsForSuccessor() async throws {
        let started = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let finished = Flag()

        let runner = MockRunner()
        runner.successor = {
            started.signal()
            _ = release.wait(timeout: .now() + 30)
            finished.set()
        }
        let transport = ScriptedTransport(lines: [
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53]}",
            "{\"id\":2,\"kind\":\"phase_diagnostics\"}",
        ])
        let probe = EngineProbe()
        let server = makeServer(runner: runner, transport: transport, probe: probe)

        let run = Task { await server.run() }
        #expect(signalled(started), "the successor is running")
        #expect(
            transport.written.count == 2,
            "the hello and the begin only: the phase close is behind the successor")
        #expect(!finished.isSet)

        release.signal()
        await run.value
        #expect(finished.isSet, "the successor finished before the worker answered")
        #expect(transport.written.count == 3)
        let reply = try JSONDecoder().decode(
            WorkerResponse.self, from: Data(transport.written[2].utf8))
        #expect(reply.ok)
        #expect(probe.shutdowns == 2, "a drain, then the wait behind the queue")
    }

    @Test("end of input stops a live engine before the worker returns")
    func endOfInputStopsEngine() async {
        let probe = EngineProbe()
        let transport = ScriptedTransport(lines: [
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53]}"
        ])
        let server = makeServer(runner: MockRunner(), transport: transport, probe: probe)
        await server.run()
        #expect(probe.shutdowns == 2)
        #expect(server.processEnd == .normal)
    }

    // MARK: Error paths

    @Test("a free-run begin whose submit throws still stops the engine it built")
    func beginSubmitErrorStopsEngine() async throws {
        let probe = EngineProbe()
        let runner = MockRunner()
        runner.submitFailure = SubmitRefused()
        let transport = ScriptedTransport(lines: [
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53]}"
        ])
        let server = makeServer(runner: runner, transport: transport, probe: probe)
        await server.run()
        let reply = try JSONDecoder().decode(
            WorkerResponse.self, from: Data(transport.written[1].utf8))
        #expect(reply.ok == false, "the refusal reaches the wire")
        #expect(probe.shutdowns == 2, "the engine behind the refusal was stopped and waited for")
    }

    @Test("a correctness run whose submit throws still stops the engine it built")
    func correctnessSubmitErrorStopsEngine() async {
        let probe = EngineProbe()
        let runner = MockRunner()
        runner.submitFailure = SubmitRefused()
        let transport = ScriptedTransport(lines: [
            "{\"id\":1,\"kind\":\"correctness\",\"prompt_tokens\":[51,52,53],\"steps\":4}"
        ])
        let server = makeServer(runner: runner, transport: transport, probe: probe)
        await server.run()
        #expect(probe.shutdowns == 2, "the free-run greedy path's engine is stopped on a throw")
    }

    // MARK: The timeout decision

    @Test("a drain that reached its watchdog is not proven quiet")
    func timeoutIsNotProvenQuiet() async {
        let probe = EngineProbe()
        let runner = MockRunner()
        runner.shutdownDelay = .milliseconds(200)
        let transport = ScriptedTransport(lines: [
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53]}",
            "{\"id\":2,\"kind\":\"phase_diagnostics\"}",
        ])
        let server = makeServer(
            runner: runner, transport: transport, probe: probe, shutdownTimeout: 0.02)
        let exits = CodeBox()
        // What the worker had already done when it decided to end.
        let atExit = CodeBox()
        server.endProcessWithoutTeardown = { code in
            exits.append(code)
            atExit.append(Int32(transport.written.count * 10 + transport.flushCount))
        }

        await server.run()
        #expect(server.processEnd == .skipRuntimeTeardown)
        #expect(
            exits.codes == [0],
            "the replies were delivered, so the code is 0; the teardown is refused")
        #expect(
            atExit.codes == [31],
            "three lines written and one flush before the process ends: 3 * 10 + 1")
    }

    @Test("a quiet drain keeps the normal exit")
    func quietDrainKeepsNormalExit() async {
        let probe = EngineProbe()
        let transport = ScriptedTransport(lines: [
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53]}",
            "{\"id\":2,\"kind\":\"phase_diagnostics\"}",
        ])
        let server = makeServer(
            runner: MockRunner(), transport: transport, probe: probe, shutdownTimeout: 5)
        let exits = CodeBox()
        server.endProcessWithoutTeardown = { exits.append($0) }

        await server.run()
        #expect(server.processEnd == .normal)
        #expect(exits.codes.isEmpty, "a proven-quiet engine keeps the process's own teardown")
    }
}
