// Qwen3.8-27B Engine Protocol v1 entry point.
//
// The shared server sends hello on accept. It returns one `ok:false` line for
// an unknown or malformed request and keeps the connection open. Its nonce is
// scoped to the connection and is present on every response, as in the shared
// conformance fixture.

import Foundation
import MLXFastEngine
import MLXLMCommon
import MLXRunners

#if canImport(Darwin)
    import Darwin
#endif

private let buildIdentity = "qwen38-27b@\(Qwen38CheckpointView.modelRevision)"

private func probeDevice() -> String {
    #if canImport(Darwin)
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0,
            size > 0
        else { return "unknown" }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0
        else { return "unknown" }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        let brand = String(decoding: bytes, as: UTF8.self).lowercased()
        for token in brand.split(separator: " ")
        where token.count >= 2 && token.hasPrefix("m") {
            if token.dropFirst().allSatisfy(\.isNumber) { return String(token) }
        }
        return brand.isEmpty ? "unknown" : brand
    #else
        return "unknown"
    #endif
}

private func fail(_ error: some Error) -> Never {
    let message = String(describing: error)
    FileHandle.standardError.write(Data("bench-worker: \(message)\n".utf8))
    exit(2)
}

private func failWithUsage(_ error: some Error, topic: BenchWorkerUsage.Topic) -> Never {
    let message = String(describing: error)
    let text = "bench-worker: \(message)\n\n\(BenchWorkerUsage.text(for: topic))\n"
    FileHandle.standardError.write(Data(text.utf8))
    exit(2)
}

private var residentSignalSources: [DispatchSourceSignal] = []
let arguments = Array(CommandLine.arguments.dropFirst())
let command: BenchWorkerCommand
do {
    command = try BenchWorkerCommand.parse(arguments)
} catch let error as WorkerLaunchError {
    if case .unknownSubcommand = error {
        failWithUsage(error, topic: .general)
    }
    failWithUsage(error, topic: BenchWorkerUsage.Topic(subcommand: arguments.first))
} catch {
    fail(error)
}

switch command {
case .help(let topic):
    print(BenchWorkerUsage.text(for: topic))
    exit(0)
case .manifest(let options):
    do {
        if options.weights != nil {
            let manifest = Qwen35Runner.manifest
            let output = options.digestOnly
                ? manifest.sha256Digest()
                : String(decoding: manifest.canonicalJSON(), as: UTF8.self)
            print(output)
        } else {
            FileHandle.standardOutput.write(Data(try options.render().utf8))
        }
    } catch {
        fail(error)
    }
    exit(0)
case .serve:
    break
}

guard case .serve(let launch) = command else {
    fatalError("unreachable command")
}

let environment = ProcessInfo.processInfo.environment
if launch.subcommand == .runtimeWorker,
    let socket = environment[BenchWorkerResidentEnvironment.socket],
    !socket.trimmingCharacters(in: .whitespaces).isEmpty
{
    let attach = BenchWorkerAttach(
        socketPath: socket,
        weightsPath: launch.weights.path,
        speculative: launch.speculative,
        emitResidentIdentity: BenchWorkerAttach.emitsResidentIdentity(
            speculative: launch.speculative, environment: environment))
    do {
        try attach.relay(input: StdioTransport(), output: StdioTransport())
    } catch {
        fail(error)
    }
    exit(0)
}

guard launch.drafter == nil else {
    fail(WorkerLaunchError.badValue("--drafter (Qwen3.8 port is serial-only)"))
}
if let runnerID = launch.runnerID,
    runnerID != Qwen35Runner.manifest.runnerID
{
    fail(WorkerLaunchError.unknownRunner(runnerID))
}

let loaded: (runner: any Runner, view: Qwen38CheckpointView)
do {
    loaded = try Qwen38RunnerLoader.load(
        weights: launch.weights,
        options: RunnerLoadOptions(
            drafterDirectory: nil,
            kvBytesCapacity: launch.kvBytesCapacity,
            resources: launch.resources))
} catch {
    fail(error)
}
let runner = loaded.runner
_ = loaded.view

switch launch.subcommand {
case .diagParity:
    do {
        try BenchWorkerLegacyParity.run(
            runner: runner, golden: launch.golden!, steps: launch.steps,
            output: FileHandle.standardOutput)
    } catch {
        fail(error)
    }
    BenchWorkerTeardown.synchronizeMLX()
    exit(0)

case .resident:
    guard let socket = launch.socket else { fail(WorkerLaunchError.missingSocket) }
    let listener: BenchWorkerSocketListener
    do {
        listener = try BenchWorkerSocketListener(path: socket)
    } catch {
        fail(error)
    }
    let resident = BenchWorkerResident(
        runner: runner,
        listener: listener,
        weightsPath: launch.weights.path,
        trusted: launch.trusted,
        speculative: launch.speculative,
        build: buildIdentity,
        device: probeDevice(),
        kvBytesCapacity: launch.kvBytesCapacity,
        pidfilePath: launch.pidfile)
    residentSignalSources = BenchWorkerResidentTermination.install(for: resident)
    if BenchWorkerResidentWarm.isEnabled() {
        do {
            _ = try BenchWorkerResidentWarm.run(
                runner: runner, memory: MLXMemoryReporter())
        } catch {
            fail(error)
        }
    }
    resident.serve()
    resident.shutDown()
    BenchWorkerTeardown.synchronizeMLX()
    exit(0)

case .runtimeWorker:
    let server = BenchWorkerServer(
        runner: runner,
        transport: StdioTransport(),
        trusted: launch.trusted,
        speculative: launch.speculative,
        build: buildIdentity,
        device: probeDevice(),
        kvBytesCapacity: launch.kvBytesCapacity)
    await server.run()
    BenchWorkerTeardown.synchronizeMLX()
}
