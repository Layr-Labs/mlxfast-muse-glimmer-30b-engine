// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Engine Protocol v1 server over `Runner`
// (Darkbloom runner contract §6.1, §8, §9).
//
// ONE server for every runner: benchd never links a runner, and this file
// carries no family name. Everything it advertises derives from the loaded
// runner's manifest plus what actually loaded; everything it reports is a
// counter the engine or the stepper already keeps.
//
// It NEVER emits a timing. benchd times the round trip; a worker that
// reported its own numbers would be scoring itself.
//
// WHY THIS IS A LIBRARY FILE while `Executables/bench-worker/main.swift` is
// a thin shim: a test target that depends on an executable target makes
// SwiftPM run that binary as the swift-testing host, which aborts the whole
// package's `@Test` pass — the reason `BenchCBv2Core`/`BenchCBv2` are split
// the same way (see Package.swift). The conformance test drives
// `BenchWorkerServer` in process over a mock runner.

import Foundation
import MLX
import MLXLMCommon

// MARK: - Transport

/// One NDJSON line in, one NDJSON line out. Abstracted so the conformance
/// test drives the same loop the binary runs.
public protocol BenchWorkerTransport: AnyObject {
    /// Next request line, or nil at EOF.
    func readLine() -> String?
    func write(line: String)
}

/// stdin/stdout.
public final class StdioTransport: BenchWorkerTransport {
    public init() {}
    public func readLine() -> String? { Swift.readLine(strippingNewline: true) }
    public func write(line: String) {
        FileHandle.standardOutput.write(Data((line + "\n").utf8))
    }
}

// MARK: - Memory reporting

/// Allocator reporting for `phase_diagnostics`.
///
/// Every member is OPTIONAL because absence is a real answer: a worker with
/// no MLX allocator has nothing to report, and the schema says an absent
/// `cache_memory` is "not asserted", which is a different claim from zero.
public protocol WorkerMemoryReporter: Sendable {
    /// Allocator watermarks read PRE-drain.
    func preDrainSnapshot() -> (active: Int, cache: Int, peak: Int)?
    /// Drain the allocator's free-buffer cache.
    func drain()
    /// Free-buffer cache size AFTER the drain. The parent fails the run
    /// closed unless this is exactly 0.
    func cacheMemoryAfterDrain() -> Int?
    /// Engine-reported peak RSS in GB. Distrusted for scoring.
    func peakRAMGB() -> Double?
}

/// The real MLX allocator.
public struct MLXMemoryReporter: WorkerMemoryReporter {
    public init() {}
    public func preDrainSnapshot() -> (active: Int, cache: Int, peak: Int)? {
        (Memory.activeMemory, Memory.cacheMemory, Memory.peakMemory)
    }
    /// Settle, then drain. A drain only returns buffers whose work has
    /// finished; arrays a phase left in flight (its last asyncEval, its
    /// teardown) come back to the cache AFTER a drain that ran too early,
    /// and the next phase's barrier then reads a non-zero cache it did not
    /// cause (a box saw 320 KiB on one resident connection in five). Waiting
    /// for the stream makes the drain read the settled state.
    public func drain() {
        MLX.Stream().synchronize()
        Memory.clearCache()
    }
    public func cacheMemoryAfterDrain() -> Int? { Memory.cacheMemory }
    public func peakRAMGB() -> Double? { Double(Memory.peakMemory) / 1_000_000_000 }
}

// MARK: - Engine audit seams

/// Cumulative speculative counters an engine may expose. `EngineV2` does.
public protocol CBv2MTPCountersReporting: AnyObject {
    func mtpMetricsSnapshot() -> CBv2MTPMetrics?
}

extension EngineV2: CBv2MTPCountersReporting {}

/// Per-round free-run audit, ASSEMBLED from the engine build's own round
/// journal (`RunnerRoundJournaling`). Every field here is a reading of records
/// the engine wrote at its finalize boundary; nothing is computed from the
/// tokens the worker happened to drain.
/// One verify round as the journal states it, across the cohort's streams.
public struct FreeRunRound: Sendable, Equatable {
    /// Per stream, the tokens this round committed. `0` means the stream was
    /// not generating in this round. A cohort's rows can commit DIFFERENT
    /// widths in one round (a drafter that finalizes each row on its own
    /// accepted prefix), so the width is stated per row rather than as one
    /// number the rows are forced to share.
    public var widths: [Int]
    /// Per stream, the PRE-min natural accept-walk length (0 when inactive).
    public var naturalAccepted: [Int]
    /// Draft tokens proposed across the streams.
    public var drafted: Int
    /// Drafts that passed verification AND were committed, across the streams.
    public var accepted: Int

    /// Tokens this round committed across every stream: the round's entry in
    /// `acceptance_lengths`.
    public var width: Int { widths.reduce(0, +) }
    /// Per stream, whether it was still generating in this round.
    public var active: [Bool] { widths.map { $0 > 0 } }
    /// Streams still generating in this round.
    public var activeStreams: Int { widths.reduce(0) { $0 + ($1 > 0 ? 1 : 0) } }

    public init(widths: [Int], naturalAccepted: [Int], drafted: Int, accepted: Int) {
        self.widths = widths
        self.naturalAccepted = naturalAccepted
        self.drafted = drafted
        self.accepted = accepted
    }

    /// A round whose active rows share one committed width.
    public init(width: Int, active: [Bool], naturalAccepted: [Int], drafted: Int, accepted: Int) {
        self.init(
            widths: active.map { $0 ? width : 0 }, naturalAccepted: naturalAccepted,
            drafted: drafted, accepted: accepted)
    }
}

public struct FreeRunRoundAudit: Sendable, Equatable {
    /// Per round, the committed width. Length is the round count R.
    public var acceptanceLengths: [Int]
    /// Per row, per round, the PRE-min natural accept-walk length. This is
    /// the record's raw `accepted`, not the `min(accepted, confirmed)` the
    /// cumulative counter accumulates — the difference is exactly the
    /// straggler throttling the cohort's one common width imposes, which is
    /// the thing this field exists to make visible.
    public var naturalAcceptedByStream: [[Int]]
    /// Rows still generating at each round. Non-increasing under the fixed-N
    /// cohort policy.
    public var activeStreamsByRound: [Int]
    /// Draft tokens proposed: the sum of every record's `k`.
    public var draftedTotal: Int
    /// Drafts that passed verification AND were committed: the sum of
    /// `min(accepted, confirmed)`, which is the `observedAccepted` the
    /// engine's own cumulative counter accumulates.
    public var acceptedTotal: Int
    /// Tokens the journal accounts for: the sum of every record's
    /// `confirmed`.
    public var committedTotal: Int

    public var rounds: Int { acceptanceLengths.count }

    /// Refusals from journal assembly. Each is a refusal to publish an
    /// audit, never a partial one.
    public enum AuditError: Error, CustomStringConvertible, Equatable {
        /// The journal evicted records this window still needed, so its head
        /// no longer covers the window. An audit assembled from what survives
        /// would look complete while missing rounds. Same rule the 125B worker
        /// applies. A journal that is merely FULL has dropped nothing and is
        /// not refused: only a position that fell off the head is.
        case journalTruncated(dropped: UInt64, cursor: UInt64, cap: Int)
        /// A record names a row this window never submitted.
        case unknownRow(requestID: UInt64)
        /// The journal's own account of the window disagrees with the tokens
        /// the worker drained. One of the two is wrong, so neither is
        /// published.
        case committedTotalDisagreement(audit: Int, drained: Int)

        public var description: String {
            switch self {
            case .journalTruncated(let dropped, let cursor, let cap):
                return "mtp round journal evicted \(dropped) records past the "
                    + "window's cursor \(cursor) (cap \(cap)): coverage cannot be proven"
            case .unknownRow(let requestID):
                return "mtp round journal names row \(requestID), which this "
                    + "window did not submit"
            case .committedTotalDisagreement(let audit, let drained):
                return "mtp round journal accounts for \(audit) tokens, "
                    + "but this window returned \(drained)"
            }
        }
    }

    /// Assemble the window's audit from the journal.
    ///
    /// Records arrive in finalize order, one per (row, round). Grouping by
    /// row and reading position-wise recovers the rounds: a row's r-th
    /// record IS its r-th round, and a row that finished early simply has no
    /// record past its last one — which is what makes
    /// `activeStreamsByRound` non-increasing without computing it from
    /// anything else.
    public static func assemble(
        records: [RunnerRoundRecord],
        slotForRequestID: [UInt64: Int],
        streamCount: Int
    ) throws -> FreeRunRoundAudit? {
        let rounds = try assembleRounds(
            records: records, slotForRequestID: slotForRequestID, streamCount: streamCount)
        guard !rounds.isEmpty else { return nil }
        return FreeRunRoundAudit(rounds: rounds, streamCount: streamCount)
    }

    /// The window's rounds, in order, from the journal. See `assemble`.
    ///
    /// Each row states ITS own committed width for the round. Two rows of a
    /// cohort may commit different widths — that is what a per-row finalize
    /// produces — so a disagreement is data, not a refusal; refusing it would
    /// drop the truth rather than state it.
    public static func assembleRounds(
        records: [RunnerRoundRecord],
        slotForRequestID: [UInt64: Int],
        streamCount: Int
    ) throws -> [FreeRunRound] {
        guard !records.isEmpty else { return [] }

        var byRow = [[RunnerRoundRecord]](repeating: [], count: streamCount)
        for record in records {
            guard let slot = slotForRequestID[record.requestID] else {
                throw AuditError.unknownRow(requestID: record.requestID)
            }
            byRow[slot].append(record)
        }

        let roundCount = byRow.map(\.count).max() ?? 0
        var rounds: [FreeRunRound] = []
        for round in 0 ..< roundCount {
            var widths = [Int](repeating: 0, count: streamCount)
            var naturalAccepted = [Int](repeating: 0, count: streamCount)
            var drafted = 0
            var accepted = 0
            var any = false
            for slot in 0 ..< streamCount where round < byRow[slot].count {
                let record = byRow[slot][round]
                widths[slot] = record.committed
                naturalAccepted[slot] = record.accepted
                drafted += record.drafted
                accepted += min(record.accepted, record.committed)
                any = true
            }
            guard any else { continue }
            rounds.append(
                FreeRunRound(
                    widths: widths, naturalAccepted: naturalAccepted,
                    drafted: drafted, accepted: accepted))
        }
        return rounds
    }

    /// Build the wire audit from rounds. Totals are sums over rounds; the
    /// per-stream natural walk is transposed to the wire's `[stream][round]`.
    ///
    /// `acceptanceLengths` states, per round, the tokens that round committed
    /// across every stream — so its sum is `committedTotal` on a cohort as
    /// well as on a single stream, which is the arithmetic benchd checks
    /// against the tokens the window returned.
    public init(rounds: [FreeRunRound], streamCount: Int) {
        self.acceptanceLengths = rounds.map(\.width)
        self.naturalAcceptedByStream = (0 ..< streamCount).map { stream in
            rounds.map { $0.naturalAccepted[stream] }
        }
        self.activeStreamsByRound = rounds.map(\.activeStreams)
        self.draftedTotal = rounds.reduce(0) { $0 + $1.drafted }
        self.acceptedTotal = rounds.reduce(0) { $0 + $1.accepted }
        self.committedTotal = rounds.reduce(0) { $0 + $1.width }
    }

    /// Split the rounds at the drain boundary.
    ///
    /// WHY. The engine runs ahead of the caller: a verify round commits up to
    /// depth + 1 tokens at once, and the journal may already hold rounds whose
    /// tokens are still in the stream. A window must state EXACTLY the
    /// tokens it returned (benchd: `sum(acceptance_lengths) == N`,
    /// `committed_total == N`), so the rounds are cut at the stream's token
    /// positions: each round covers `widths[stream]` tokens of one stream,
    /// and this window states the tokens whose positions fall in
    /// `[statedPerStream, statedPerStream + drainedPerStream)`.
    ///
    /// - A round wholly inside that span is stated intact, drafts and accepts
    ///   included.
    /// - A round that straddles the END is CLIPPED: this window states the
    ///   width it returned, the drafts that round proposed, and the accepts
    ///   that fit (`min(accepted, width - 1)`; the bonus token is the one past
    ///   the accepts). Its remainder carries into the next window with no
    ///   drafts and no accepts, because those were already stated.
    /// - A round wholly past the end carries intact.
    /// - A round whose head falls BEFORE `statedPerStream` has already had
    ///   that head stated (the `free_decode_begin` seed token), so only its
    ///   in-span part is stated here and its drafts stay with the window that
    ///   saw them.
    ///
    /// Nothing is dropped and nothing is counted twice; a box saw the
    /// alternative, a refusal whenever a round did not land on the count.
    /// The positions are per STREAM, so rows that commit different widths are
    /// cut at each row's own boundary rather than at one shared number.
    public static func split(
        rounds: [FreeRunRound], statedPerStream stated: Int, drainedPerStream drained: Int
    ) -> (window: [FreeRunRound], carry: [FreeRunRound]) {
        var window: [FreeRunRound] = []
        var carry: [FreeRunRound] = []
        let streams = rounds.map { $0.widths.count }.max() ?? 0
        let low = stated
        let high = stated + drained
        // The list starts where this window starts: a carried round is the
        // continuation of the last stated position, and the rounds after it
        // follow in order. Positions are ABSOLUTE within the session, so a round
        // carried out of one window lands at the same place in the next.
        var cursor = [Int](repeating: low, count: streams)
        for round in rounds {
            var headWidths = [Int](repeating: 0, count: streams)
            var tailWidths = [Int](repeating: 0, count: streams)
            var headNatural = [Int](repeating: 0, count: streams)
            var headLostTokens = false
            for stream in 0 ..< streams {
                let width = stream < round.widths.count ? round.widths[stream] : 0
                let start = cursor[stream]
                let end = start + width
                cursor[stream] = end
                let headStart = max(start, low)
                let headEnd = min(end, high)
                if headEnd > headStart { headWidths[stream] = headEnd - headStart }
                let tailStart = max(start, high)
                if end > tailStart { tailWidths[stream] = end - tailStart }
                if start < min(end, low) { headLostTokens = true }
                let natural = stream < round.naturalAccepted.count ? round.naturalAccepted[stream] : 0
                headNatural[stream] = max(0, min(natural, headWidths[stream] - 1))
            }
            let headStated = headWidths.contains { $0 > 0 }
            let tailStated = tailWidths.contains { $0 > 0 }
            // Clipped only when this window returned LESS of the round than it
            // committed on some stream. An unclipped round keeps its own
            // drafts and accepts untouched.
            let headClipped = (0 ..< streams).contains { stream in
                let width = stream < round.widths.count ? round.widths[stream] : 0
                return headWidths[stream] < width
            }
            if headStated {
                var head = round
                head.widths = headWidths
                if headLostTokens {
                    // The front of this round was stated by an earlier window.
                    head.drafted = 0
                    head.accepted = 0
                    head.naturalAccepted = headNatural
                } else if headClipped {
                    head.accepted = min(
                        round.accepted,
                        max(0, headWidths.reduce(0, +) - headWidths.filter { $0 > 0 }.count))
                    head.naturalAccepted = headNatural
                }
                window.append(head)
            }
            if tailStated {
                var tail = round
                tail.widths = tailWidths
                tail.naturalAccepted = [Int](repeating: 0, count: streams)
                // Intact only when the whole round is past the boundary;
                // otherwise this is a straddling remainder and its drafts
                // were stated with the head.
                if headStated || headLostTokens {
                    tail.drafted = 0
                    tail.accepted = 0
                }
                carry.append(tail)
            }
        }
        return (window, carry)
    }

    /// The rounds that state plain SEED steps: each is one engine step that
    /// committed one token in `tokens` different rows, so the round count and
    /// the token count are stated separately. A batched reseed of `b` rows is
    /// ONE round of `b` tokens, not `b` rounds of `b` — which is what counting
    /// rows as rounds overstated. See `FreeRunSession.seedCursor`.
    public static func seedRounds(
        tokens: Int, rounds: Int, streamCount: Int
    ) -> [FreeRunRound] {
        guard tokens > 0, streamCount > 0 else { return [] }
        let roundCount = max(1, min(rounds, tokens))
        let base = tokens / roundCount
        let extra = tokens % roundCount
        return (0 ..< roundCount).map { round in
            let count = min(streamCount, base + (round < extra ? 1 : 0))
            var widths = [Int](repeating: 0, count: streamCount)
            for stream in 0 ..< count { widths[stream] = 1 }
            return FreeRunRound(
                widths: widths,
                naturalAccepted: [Int](repeating: 0, count: streamCount),
                drafted: 0, accepted: 0)
        }
    }

    /// The audit of a serial window: one round of width one per committed
    /// token, nothing drafted, nothing accepted. `rounds` is the longest
    /// stream; a stream that stopped early leaves the later rounds — and only
    /// that row's own width drops, so the per-round entry is the tokens the
    /// round committed across the rows still generating.
    public static func serial(committedByStream: [[Int]]) -> FreeRunRoundAudit {
        let rounds = committedByStream.map(\.count).max() ?? 0
        let roundList = (0 ..< rounds).map { round in
            FreeRunRound(
                widths: committedByStream.map { round < $0.count ? 1 : 0 },
                naturalAccepted: [Int](repeating: 0, count: committedByStream.count),
                drafted: 0, accepted: 0)
        }
        return FreeRunRoundAudit(rounds: roundList, streamCount: committedByStream.count)
    }

    public init(
        acceptanceLengths: [Int],
        naturalAcceptedByStream: [[Int]],
        activeStreamsByRound: [Int],
        draftedTotal: Int,
        acceptedTotal: Int,
        committedTotal: Int
    ) {
        self.acceptanceLengths = acceptanceLengths
        self.naturalAcceptedByStream = naturalAcceptedByStream
        self.activeStreamsByRound = activeStreamsByRound
        self.draftedTotal = draftedTotal
        self.acceptedTotal = acceptedTotal
        self.committedTotal = committedTotal
    }
}

// MARK: - Speculative protocol gate

/// The `--speculative-protocol` argument.
///
/// benchd's official spawn always appends `--speculative-protocol v1.1`
/// (`free_run_spawn_args`), and the flag is what turns the SPECULATIVE
/// surface on. Absent, the worker speaks plain v1: `capabilities`,
/// `spec_modes` and `max_batch_size` are all v1.1-or-later additions to a
/// frozen v1 hello, so a v1-only engine omits them, and a `spec` on a
/// request is refused rather than honoured. That is not a degraded mode with
/// speculation quietly disabled — it is a different protocol version, and
/// answering a `spec` under it would be a worker resolving something the
/// parent never established it could ask for.
public enum SpeculativeProtocol: Sendable, Equatable {
    /// The only accepted value.
    public static let version = "v1.1"

    /// Refusals from the argument. Before the load, always.
    public enum ArgumentError: Error, CustomStringConvertible, Equatable {
        case unsupportedVersion(String)

        public var description: String {
            switch self {
            case .unsupportedVersion(let raw):
                return "--speculative-protocol \(raw) is not \(SpeculativeProtocol.version)"
            }
        }
    }

    /// Whether the speculative surface is enabled. `nil` (flag absent) is
    /// plain v1; any value other than `v1.1` is a refusal, never a fallback
    /// to plain v1 — a parent that asked for a version this worker does not
    /// speak must hear so, not be silently served an older one.
    public static func isEnabled(_ raw: String?) throws -> Bool {
        guard let raw else { return false }
        guard raw == version else { throw ArgumentError.unsupportedVersion(raw) }
        return true
    }
}

// MARK: - Capabilities

/// Capability strings the hello advertises.
public enum WorkerCapability {
    public static let freeRunDecode = "free_run_decode"
    public static let batchedFreeRunDecode = "batched_free_run_decode"
    public static let perStreamTiming = "per_stream_timing"
    public static let cohortReferenceReplay = "cohort_reference_replay"
}

// MARK: - Server

/// Engine Protocol v1 server over one loaded runner.
///
/// The model is loaded ONCE by the caller, before the hello. This object
/// never loads, never downloads, and never tokenizes: token ids cross the
/// boundary in both directions.
public final class BenchWorkerServer: @unchecked Sendable {

    /// The protocol version this worker implements.
    public static let protocolVersion = 1

    private let runner: any Runner
    private let manifest: RunnerManifest
    private let transport: any BenchWorkerTransport
    private let trusted: Bool
    /// `--speculative-protocol v1.1` was given. See ``SpeculativeProtocol``.
    private let speculative: Bool
    private let build: String
    private let device: String
    private let kvBytesCapacity: Int
    private let maxDecodeTokens: Int
    private let memory: any WorkerMemoryReporter
    /// Non-nil only when this session is served by a RESIDENT. See
    /// `BenchWorkerResident`.
    private let residentIdentity: ResidentIdentity?
    private let residentWeights: String?
    private let nonce: String
    private let decoder: JSONDecoder

    /// Timed step-requests completed since the last `phase_diagnostics`.
    ///
    /// A WORKER counter, not `stepper.forwards`: a phase issues verbs across
    /// several stepper sessions (`decode_begin` starts a fresh one), and the
    /// parent's phase-close barrier compares this against the number of timed
    /// steps IT issued. `prefill` and `correctness` are not timed steps and
    /// do not move it.
    private var completedWork = 0

    private var stepper: (any TeacherForcedStepper)?
    private var freeRun: FreeRunSession?

    public init(
        runner: any Runner,
        transport: any BenchWorkerTransport,
        trusted: Bool,
        speculative: Bool = true,
        build: String,
        device: String,
        kvBytesCapacity: Int,
        maxDecodeTokens: Int = 4096,
        memory: any WorkerMemoryReporter = MLXMemoryReporter(),
        residentIdentity: ResidentIdentity? = nil,
        residentWeights: String? = nil,
        nonce: String = UUID().uuidString
    ) {
        self.runner = runner
        self.manifest = type(of: runner).manifest
        self.transport = transport
        self.trusted = trusted
        self.speculative = speculative
        self.build = build
        self.device = device
        self.kvBytesCapacity = kvBytesCapacity
        self.maxDecodeTokens = maxDecodeTokens
        self.memory = memory
        self.residentIdentity = residentIdentity
        self.residentWeights = residentWeights
        self.nonce = nonce
        self.decoder = JSONDecoder()
    }

    // MARK: Hello (§6.1)

    /// Every field derived from the manifest and the loaded state. No runner
    /// writes a hello field by hand, and none is withheld by a constant here.
    public func hello() -> WorkerResponse {
        var response = WorkerResponse(id: 0, ok: true, nonce: nonce)
        response.expertStats = ExpertStreamingStats()
        response.protocolVersion = Self.protocolVersion
        response.backend = manifest.backend
        response.device = device

        // `capabilities`, `spec_modes` and `max_batch_size` are all
        // v1.1-or-later additions to a frozen v1 hello, so a worker speaking
        // plain v1 omits the three of them together rather than advertising
        // an empty capability list — an empty array is a claim ("I support
        // none of these"), absence is the v1 wire.
        if speculative {
            var capabilities: [String] = []
            if manifest.regimes.contains(where: { $0.timing == .freeRun }) {
                capabilities.append(WorkerCapability.freeRunDecode)
            }
            if manifest.regimes.contains(where: { $0.batch.maxWidth > 1 }) {
                capabilities.append(WorkerCapability.batchedFreeRunDecode)
            }
            if manifest.regimes.contains(where: { $0.perStreamTiming }) {
                capabilities.append(WorkerCapability.perStreamTiming)
            }
            if trusted {
                capabilities.append(WorkerCapability.cohortReferenceReplay)
            }
            response.capabilities = capabilities
            response.specModes = runner.loadedDecoders.map(\.rawValue)
            response.maxBatchSize =
                manifest.regimes.map(\.batch.maxWidth).filter { $0 > 1 }.max()
        }
        response.headProvenance = runner.headProvenance.map(WireHeadProvenance.init)
        // Resident identity rides only on the SOCKET, where the attaching
        // worker reads it, checks the weights path and decides what reaches
        // benchd. An in-process worker has no resident and emits neither.
        response.resident = residentIdentity
        response.residentWeights = residentWeights
        response.runner = RunnerIdentity(
            id: manifest.runnerID,
            modelType: runner.loadedModelType,
            manifestSHA256: manifest.sha256Digest(),
            build: build)
        return response
    }

    // MARK: Loop

    /// Emit the unsolicited hello, then serve until EOF.
    public func run() async {
        send(hello())
        while let line = transport.readLine() {
            if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            let request: WorkerRequest
            do {
                request = try decoder.decode(WorkerRequest.self, from: Data(line.utf8))
            } catch {
                var response = WorkerResponse(id: -1, ok: false, nonce: nonce)
                response.error = "unparseable request: \(error)"
                send(response)
                continue
            }
            send(await handle(request))
        }
    }

    private func send(_ response: WorkerResponse) {
        transport.write(line: response.jsonLine())
    }

    /// One verb. Any throw becomes `ok:false` + `error`; the parent discards
    /// the phase.
    public func handle(_ request: WorkerRequest) async -> WorkerResponse {
        do {
            return try await dispatch(request)
        } catch {
            var response = WorkerResponse(id: request.id, ok: false, nonce: nonce)
            response.error = "\(error)"
            return response
        }
    }

    private func dispatch(_ request: WorkerRequest) async throws -> WorkerResponse {
        var response = WorkerResponse(id: request.id, ok: true, nonce: nonce)
        switch request.kind {

        case .prefill:
            let prompt = try require(request.promptTokens, "prompt_tokens")
            response.token = try beginStepper().forward(prompt).argmax

        case .decodeBegin:
            let seed = try require(request.seedTokens, "seed_tokens")
            response.seedToken = try beginStepper().forward(seed).argmax
            if let spec = request.spec {
                response.effectiveSpec = try effectiveSpec(spec)
            }
            completedWork += 1

        case .decodeStep:
            let token = try require(request.token, "token")
            response.token = try currentStepper().forward([token]).argmax
            completedWork += 1

        case .correctness:
            let prompt = try require(request.promptTokens, "prompt_tokens")
            let steps = try require(request.steps, "steps")
            response.tokens = try await freeRunGreedy(prompt: prompt, steps: steps)
            response.peakRAMGB = memory.peakRAMGB()

        case .correctnessBegin:
            let prompt = try require(request.promptTokens, "prompt_tokens")
            let stepper = try beginStepper()
            if let chunk = Self.diagnosticStepperChunk, chunk < prompt.count {
                // DIAGNOSTIC ONLY: prefill the teacher-forced row in chunks
                // of `chunk` tokens, the shape the engine loop gives a free
                // run, so the two prefill shapes can be swapped on a box.
                var output: StepOutput?
                var start = 0
                while start < prompt.count {
                    let end = min(start + chunk, prompt.count)
                    output = try stepper.forward(Array(prompt[start ..< end]))
                    start = end
                }
                fill(&response, with: output!)
            } else {
                fill(&response, with: try stepper.forward(prompt))
            }
            completedWork += 1

        case .correctnessStep:
            let token = try require(request.token, "token")
            fill(&response, with: try currentStepper().forward([token]))
            completedWork += 1

        case .freeDecodeBegin:
            response = try await freeDecodeBegin(request)

        case .freeDecodeRun:
            response = try await freeDecodeRun(request)

        case .cohortReferenceReplay:
            response = try cohortReferenceReplay(request)

        case .phaseDiagnostics:
            // PRE-drain watermarks first: the drain is what makes
            // `cache_memory` assertable, and reading after it would describe
            // the drain rather than the phase.
            let snapshot = memory.preDrainSnapshot()
            let cacheAfterDrain = await releaseSessionsAndSettle()
            // CBV2_STEP_PROFILE=1: the engine loop's per-phase timing table
            // (launch, readback wait, sampler, per-step phases) on stderr,
            // one table per phase, then reset so the next phase reads clean.
            // Diagnostic only; stdout stays the wire.
            if CBv2StepProfiler.enabled {
                FileHandle.standardError.write(
                    Data(
                        ("bench-worker: step profile\n" + CBv2StepProfiler.summaryTable() + "\n")
                            .utf8))
                CBv2StepProfiler.reset()
            }
            response.expertStats = ExpertStreamingStats()
            response.peakRAMGB = memory.peakRAMGB()
            response.completedWork = completedWork
            // The read the settle loop ended on, not a fresh one: a fresh
            // read races the same stragglers the loop just waited out.
            response.cacheMemory = cacheAfterDrain
            response.mlxActiveMemoryBytes = snapshot?.active
            response.mlxCacheMemoryBytes = snapshot?.cache
            response.mlxPeakMemoryBytes = snapshot?.peak
            completedWork = 0
        }
        return response
    }

    // MARK: Stepper verbs (§8)

    private func beginStepper() throws -> any TeacherForcedStepper {
        let stepper = try runner.makeStepper()
        try stepper.begin()
        self.stepper = stepper
        return stepper
    }

    private func currentStepper() throws -> any TeacherForcedStepper {
        guard let stepper else { throw StepperError.notBegun }
        return stepper
    }

    /// `correctness_begin` / `correctness_step` payload. `decode_step` and
    /// `correctness_step` reach the SAME `forward`; only the reporting
    /// differs, which is the invariant benchd's architecture doc calls
    /// non-negotiable.
    private func fill(_ response: inout WorkerResponse, with output: StepOutput) {
        response.token = output.argmax
        response.topLogits = output.topLogits.map {
            CorrectnessTraceLogit(token: $0.token, logit: $0.logit)
        }
        response.expertStats = ExpertStreamingStats()
        response.peakRAMGB = memory.peakRAMGB()
    }

    // MARK: Engine verbs (§8, §9)

    /// The FIXED build (§9): contiguous, `maxConcurrentRequests == batch`, no
    /// prefix cache, no waiting-queue use, no stop tokens on cohort requests.
    /// Nothing here is policy the worker invents — it is the one build the
    /// contract pins for benchd.
    private func engineBuild(batch: Int, spec: SpecConfig, seedLength: Int) throws -> EngineBuild {
        let decoderID = DecoderID(spec.mode)
        var mtpConfig = CBv2MTPConfig(enabled: false)
        if decoderID != .serial {
            let depth = max(1, spec.depth)
            mtpConfig = CBv2MTPConfig(
                enabled: true,
                maxDraftTokens: depth,
                maxSpeculativeBatch: batch,
                fixedDraftTokens: depth,
                verificationMode: .automatic,
                // The envelope is the INTEGRATOR's claim that rectangular
                // target evaluation is argmax-exact at every shape inside it.
                // For a benchmark leg the shape set is exactly this one — B
                // rows by 1+depth verify columns — so the claim is the leg.
                // The engine clamps the requested depth to its tested window
                // (`CBv2MTPConfig.testedMaxDraftTokens`). A drafter that can
                // take a narrower depth says so on itself, through its own
                // `maximumDraftTokens`; there is no per-decoder ceiling here.
                maxAutomaticRectangularTokens: batch * (1 + depth))
        }
        return EngineBuild(
            // The RUNNER's declaration picks the backend. A family that
            // serves only paged storage refuses a contiguous build by name,
            // and a worker that overrode that would be inventing policy for
            // the family it was handed.
            kvBackend: Self.kvBackend(for: runner.manifest),
            kvBytesCapacity: kvBytesCapacity,
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: batch,
                // ONE SHAPE FOR THE ORACLE AND THE RUN. The correctness
                // oracle (teacher-forced stepper) and the goldens it is
                // checked against prefill the seed in ONE forward. The engine
                // loop's default chunks a prompt at 512, and a chunked prefill
                // is not bit-identical to a single forward: the attention and
                // recurrent kernels tile differently, which the fixture model
                // shows as a ~5e-5 logit delta in float32 and the 125B shows
                // as an argmax flip on the first close margin (a box saw the
                // live golden diverge at step 1 with two 512-token chunks
                // and reproduce through nine tokens with one). So a free run
                // prefills its whole seed in one chunk. The diagnostic knob
                // still overrides, by name, for shape experiments.
                //
                // A model that cuts prompts on a grid also needs a chunk that
                // can hold one cell, or the paged pool refuses the model at
                // engine start.
                prefillChunkSize: Self.diagnosticPrefillChunk
                    ?? max(
                        CBv2SchedulerConfig().prefillChunkSize, seedLength,
                        runner.manifest.engine.prefillChunkAlignment ?? 0),
                maxWaiting: batch,
                enablePrefixCache: false),
            loopConfig: CBv2EngineLoopConfig(),
            prefixCache: nil,
            decoder: decoderID,
            mtpConfig: mtpConfig,
            // Ignored by a contiguous build; the pool a paged build plans is
            // sized for the longest row the worker will be handed.
            pagedPool: PagedPoolPlan(
                nominalMaxSequenceLength: max(
                    RunnerEngineAssembly.nominalMaxSequenceLength, seedLength)),
            environment: [:])
    }

    /// The backend the fixed build asks for: the runner's own declaration,
    /// contiguous only when the runner serves it.
    static func kvBackend(for manifest: RunnerManifest) -> KVBackendKind {
        manifest.kvBackends.contains(.contiguous) ? .contiguous : (manifest.kvBackends.first ?? .contiguous)
    }

    /// `correctness`: free-run greedy through the engine, tokens only.
    private func freeRunGreedy(prompt: [Int], steps: Int) async throws -> [Int] {
        let engine = try runner.makeEngine(
            try engineBuild(
                batch: 1, spec: SpecConfig(mode: DecoderID.serial.rawValue),
                seedLength: prompt.count))
        var request = CBv2Request(
            id: CBv2RequestID(1), promptTokens: prompt, maxTokens: steps)
        request.sampling = Self.greedySampling
        request.stopTokens = []
        var tokens: [Int] = []
        for await event in try engine.submit(request) {
            if case .delta(_, let emitted, _) = event {
                tokens.append(contentsOf: emitted)
                if tokens.count >= steps { break }
            }
        }
        await engine.shutdown()
        return Array(tokens.prefix(steps))
    }

    /// One free-run window: the engine, its streams, and the counter baseline
    /// taken at `free_decode_begin` so the run response reports THIS window's
    /// audit, not the process's.
    private final class FreeRunSession {
        let engine: any CBv2Engine
        let batch: Int
        let spec: SpecConfig
        let cohort: Bool
        /// Request id -> cohort slot, in SLOT ORDER. The journal names rows
        /// by request id; the wire names them by slot.
        let slotForRequestID: [UInt64: Int]
        /// Position in the journal's own numbering (see `RunnerRoundJournal`):
        /// the sequence number of the next record this session has not yet
        /// stated. Taken BEFORE the seed forward is submitted, so a round that
        /// finalizes while the seed drains cannot fall between the seed and the
        /// cursor and vanish from every window.
        var journalCursor: UInt64 = 0
        /// `CBv2MTPMetrics.seedSteps` at the same boundary. A SEED step is the
        /// plain forward the planner runs when a row has no carry (the first
        /// step of every window, and after a stale carry): it commits ONE token
        /// in ONE row and writes no round record, so the drain states it as a
        /// round itself. `seedCursor` counts ROWS; `seedRoundCursor` counts the
        /// engine STEPS those rows were seeded in.
        var seedCursor = 0
        var seedRoundCursor = 0
        /// Tokens already returned to the harness, per stream: the seed token
        /// `free_decode_begin` returned, plus every token a run has drained.
        /// The audit's cut is taken at this position, per stream.
        var statedPerStream = 0
        /// Rounds the journal already stated whose tokens this session has
        /// not yet returned: the clipped remainder and any run-ahead rounds.
        /// See `FreeRunRoundAudit.split`.
        var carriedRounds: [FreeRunRound] = []
        /// Tokens read off a stream but not yet returned: a delta can carry
        /// more than the count a run asked for, and dropping the rest would
        /// lose them.
        var pending: [[Int]]
        /// Controller clamp counts as they stood at that same boundary.
        var clampBaseline: [String: Int] = [:]
        var streams: [AsyncStream<CBv2Event>.Iterator]

        init(
            engine: any CBv2Engine, batch: Int, spec: SpecConfig, cohort: Bool,
            slotForRequestID: [UInt64: Int],
            streams: [AsyncStream<CBv2Event>.Iterator]
        ) {
            self.engine = engine
            self.batch = batch
            self.spec = spec
            self.cohort = cohort
            self.slotForRequestID = slotForRequestID
            self.pending = Array(repeating: [], count: batch)
            self.streams = streams
        }
    }

    /// Controller clamp/skip reasons as one histogram. Both maps are
    /// cumulative engine counters; the wire wants the window's delta.
    private static func clampReasons(_ metrics: CBv2MTPMetrics) -> [String: Int] {
        var reasons = metrics.skippedRows
        for (reason, count) in metrics.controllerFallbacks {
            reasons[reason, default: 0] += count
        }
        return reasons
    }

    private func freeDecodeBegin(_ request: WorkerRequest) async throws -> WorkerResponse {
        // The `spec` refusal comes FIRST because it is the more specific
        // answer: under plain v1 the parent asked for a speculative surface
        // by name, and "free_run_decode is not served" would send it looking
        // at the manifest instead of at its own spawn.
        let spec = try effectiveSpec(request.spec)
        guard speculative, manifest.regimes.contains(where: { $0.timing == .freeRun })
        else {
            throw WorkerError.capabilityNotAdvertised(WorkerCapability.freeRunDecode)
        }

        let seeds: [[Int]]
        let cohort: Bool
        if let byStream = request.seedTokensByStream {
            seeds = byStream
            cohort = true
        } else {
            seeds = [try require(request.seedTokens, "seed_tokens")]
            cohort = request.batchSize != nil
        }
        let batch = request.batchSize ?? seeds.count
        guard batch == seeds.count else {
            throw WorkerError.batchMismatch(requested: batch, streams: seeds.count)
        }
        if batch > 1 {
            guard manifest.regimes.contains(where: { $0.batch.maxWidth >= batch }) else {
                throw WorkerError.capabilityNotAdvertised(
                    WorkerCapability.batchedFreeRunDecode)
            }
        }

        let engine = try runner.makeEngine(
            try engineBuild(batch: batch, spec: spec, seedLength: seeds.map(\.count).max() ?? 0))
        // Submit ALL streams before consuming ANY: a cohort drained stream by
        // stream is a sequence of solo runs wearing a cohort's name.
        var iterators: [AsyncStream<CBv2Event>.Iterator] = []
        for (slot, seed) in seeds.enumerated() {
            var cbv2 = CBv2Request(
                id: CBv2RequestID(UInt64(slot + 1)), promptTokens: seed,
                maxTokens: maxDecodeTokens)
            cbv2.sampling = Self.greedySampling
            cbv2.stopTokens = []
            iterators.append(try engine.submit(cbv2).makeAsyncIterator())
        }

        var slotForRequestID: [UInt64: Int] = [:]
        for slot in 0 ..< batch { slotForRequestID[UInt64(slot + 1)] = slot }
        let session = FreeRunSession(
            engine: engine, batch: batch, spec: spec, cohort: cohort,
            slotForRequestID: slotForRequestID, streams: iterators)
        // The audit's starting position is taken BEFORE the seed is submitted:
        // the journal is empty at engine build, and every record it writes from
        // here on belongs to a token this session will return. Taking it after
        // the seed drain used to drop the rounds that finalized in between —
        // their tokens arrived in the first run while their records were
        // skipped by the cursor.
        if let journal = engine as? any RunnerRoundJournaling {
            session.journalCursor = journal.roundJournal().dropped
        }
        if let metrics = (engine as? any CBv2MTPCountersReporting)?.mtpMetricsSnapshot() {
            session.seedCursor = metrics.seedSteps
            session.seedRoundCursor = metrics.seedRounds
            session.clampBaseline = Self.clampReasons(metrics)
        }
        var seedTokens: [Int] = []
        for slot in 0 ..< batch {
            seedTokens.append(
                contentsOf: try await drain(session: session, slot: slot, count: 1))
        }
        // The seed token `free_decode_begin` returns is not part of any run
        // window, so the first window's cut starts after it.
        session.statedPerStream = 1
        self.freeRun = session

        var response = WorkerResponse(id: request.id, ok: true, nonce: nonce)
        if cohort {
            response.seedTokenByStream = seedTokens
            response.effectiveBatchSize = batch
        } else {
            response.seedToken = seedTokens.first
        }
        response.effectiveSpec = spec
        completedWork += 1
        return response
    }

    private func freeDecodeRun(_ request: WorkerRequest) async throws -> WorkerResponse {
        guard let session = freeRun else { throw WorkerError.noFreeRunSession }
        let count = try require(request.count, "count")
        if let batch = request.batchSize, batch != session.batch {
            throw WorkerError.batchMismatch(requested: batch, streams: session.batch)
        }
        var byStream: [[Int]] = []
        for slot in 0 ..< session.batch {
            byStream.append(try await drain(session: session, slot: slot, count: count))
        }

        var response = WorkerResponse(id: request.id, ok: true, nonce: nonce)
        if session.cohort {
            response.tokensByStream = byStream
            response.effectiveBatchSize = session.batch
        } else {
            response.tokens = byStream.first
        }

        let drained = byStream.reduce(0) { $0 + $1.count }
        response.committedTotal = drained

        // AUDIT, assembled from the engine build's OWN round journal. Nothing
        // here is recomputed from the tokens the worker drained: the journal
        // states what each round drafted, accepted and committed, and the
        // worker only groups the records it was given. A build that keeps no
        // journal states no audit — an inferred width distribution would look
        // like evidence while being a guess.
        var audit: FreeRunRoundAudit?
        let journal = session.engine as? any RunnerRoundJournaling
        if let metrics = (session.engine as? any CBv2MTPCountersReporting)?
            .mtpMetricsSnapshot()
        {
            if let journal {
                let view = journal.roundJournal()
                // A cursor still inside the retained range is covered even when
                // the ring is exactly FULL: only a cursor the ring has dropped
                // PAST cannot be proven.
                guard session.journalCursor >= view.dropped else {
                    throw FreeRunRoundAudit.AuditError.journalTruncated(
                        dropped: view.dropped, cursor: session.journalCursor,
                        cap: journal.roundRecordCap)
                }
                let firstNew = Int(session.journalCursor - view.dropped)
                let records = Array(
                    view.records.dropFirst(min(firstNew, view.records.count)))
                let fresh = try FreeRunRoundAudit.assembleRounds(
                    records: records,
                    slotForRequestID: session.slotForRequestID,
                    streamCount: session.batch)
                // Everything retained has now been stated, including the rounds
                // that ran ahead of the drain; those carry below.
                session.journalCursor = view.nextSequence
                // Seed steps taken inside this window (see `seedCursor`): each
                // is one engine step that committed one token in one row with
                // nothing drafted, stated as a round ahead of the rounds it
                // seeded, so the window states every token it returned (benchd:
                // sum of the widths == N). The engine counts ROWS seeded and
                // STEPS taken separately, because a batched reseed seeds
                // several rows in one step.
                let seedDelta = max(0, metrics.seedSteps - session.seedCursor)
                let seedRoundDelta = max(0, metrics.seedRounds - session.seedRoundCursor)
                session.seedCursor = metrics.seedSteps
                session.seedRoundCursor = metrics.seedRounds
                let seeded = FreeRunRoundAudit.seedRounds(
                    tokens: seedDelta, rounds: seedRoundDelta, streamCount: session.batch)
                let cut = FreeRunRoundAudit.split(
                    rounds: session.carriedRounds + seeded + fresh,
                    statedPerStream: session.statedPerStream, drainedPerStream: count)
                session.carriedRounds = cut.carry
                if !cut.window.isEmpty {
                    let assembled = FreeRunRoundAudit(
                        rounds: cut.window, streamCount: session.batch)
                    // The journal's own account is the authority for
                    // `committed_total`. A disagreement with the tokens this
                    // window returned means one of the two is wrong, so
                    // neither is published.
                    guard assembled.committedTotal == drained else {
                        throw FreeRunRoundAudit.AuditError.committedTotalDisagreement(
                            audit: assembled.committedTotal, drained: drained)
                    }
                    audit = assembled
                }
            } else {
                // No journal: the seed cursor still advances, so a build that
                // gains a journal mid-run cannot restate these steps as rounds.
                session.seedCursor = metrics.seedSteps
                session.seedRoundCursor = metrics.seedRounds
            }

            var clamps = Self.clampReasons(metrics)
            for (reason, base) in session.clampBaseline {
                clamps[reason, default: 0] -= base
            }
            clamps = clamps.filter { $0.value > 0 }
            session.clampBaseline = Self.clampReasons(metrics)
            if !clamps.isEmpty { response.depthClampReasons = clamps }
            // How the target verified, so an artifact can prove the window
            // ran as one forward rather than one forward per column. Only on
            // a window that verified: the frozen fixtures carry no rounds and
            // stay byte-identical.
            let verifyRounds =
                metrics.rectangularVerificationRounds + metrics.serialVerificationRounds
            if verifyRounds > 0 {
                response.verificationMode = metrics.verificationMode.rawValue
                response.rectangularVerificationRounds = metrics.rectangularVerificationRounds
                response.serialVerificationRounds = metrics.serialVerificationRounds
            }
        }

        // The tokens are now the harness's: the next window's cut starts here.
        // Advanced for every window, journaled or not, so a build that gains a
        // journal mid-run still cuts at the right position.
        session.statedPerStream += count

        // A SERIAL window journals no verify rounds, but the protocol carries
        // the audit triple on EVERY free_decode_run (PROTOCOL.md: benchd
        // refuses a reply without `acceptance_lengths`, and the box saw the
        // official warmup leg fail on exactly that). The serial statement is
        // a statement, not a guess: each committed token is its own round of
        // width one, nothing was drafted, nothing was accepted. A non-serial
        // window with an empty journal stays ABSENT, so an MTP engine that
        // failed to journal is refused by benchd rather than reported as
        // zero.
        if audit == nil, session.spec.mode == SpecConfig.serialMode {
            audit = FreeRunRoundAudit.serial(committedByStream: byStream)
        }

        if let audit {
            response.acceptanceLengths = audit.acceptanceLengths
            response.draftedTotal = audit.draftedTotal
            response.acceptedTotal = audit.acceptedTotal
            // The journal's own token account, not a second count of the
            // drained stream: the two were proven equal above, and the audit
            // is the one that carries the per-round evidence.
            response.committedTotal = audit.committedTotal
            // `rounds`, `natural_accepted_by_stream` and
            // `active_streams_by_round` are the v1.2 COHORT trio; the
            // single-stream form carries `acceptance_lengths` alone, whose
            // length already IS R.
            if session.cohort {
                response.rounds = audit.rounds
                response.naturalAcceptedByStream = audit.naturalAcceptedByStream
                response.activeStreamsByRound = audit.activeStreamsByRound
            }
        }
        // `spec_decoder` and `verify_replay_disagreements` have no journal
        // field at this pin, so both stay ABSENT — which is not the same
        // claim as naming the request's own mode back, nor as reporting 0.

        // Timed work for a free-run window is engine STEPS, not tokens: one
        // per verify round when the engine journals rounds, otherwise one per
        // committed token (a serial window's steps and tokens coincide).
        completedWork += audit?.rounds ?? count
        return response
    }

    /// Drain exactly `count` committed tokens from one stream.
    ///
    /// A delta carries as many tokens as the engine committed at once (up to a
    /// whole verify round), which can be MORE than the count this window asked
    /// for. The surplus is kept in the session's per-stream buffer and returned
    /// ahead of anything read next, so no token is lost and none is returned
    /// twice.
    private func drain(
        session: FreeRunSession, slot: Int, count: Int
    ) async throws -> [Int] {
        var out = session.pending[slot]
        session.pending[slot] = []
        // The iterator is moved out and back rather than mutated in place:
        // a mutating call across an `await` on `session.streams[slot]` would
        // hold an exclusive access over a suspension point.
        var iterator = session.streams[slot]
        defer { session.streams[slot] = iterator }
        while out.count < count {
            guard let event = await iterator.next() else {
                throw WorkerError.streamEndedEarly(
                    slot: slot, got: out.count, want: count, reason: "stream closed")
            }
            switch event {
            case .delta(_, let tokens, _):
                out.append(contentsOf: tokens)
            case .finished(let reason, _):
                guard out.count >= count else {
                    throw WorkerError.streamEndedEarly(
                        slot: slot, got: out.count, want: count, reason: "\(reason)")
                }
            }
        }
        let delivered = Array(out.prefix(count))
        session.pending[slot] = Array(out.dropFirst(count))
        return delivered
    }

    /// Trusted-build oracle: teacher-forced replay of the candidate's own
    /// committed journal, through the engine's own path.
    private func cohortReferenceReplay(_ request: WorkerRequest) throws -> WorkerResponse {
        guard trusted else {
            throw WorkerError.capabilityNotAdvertised(
                WorkerCapability.cohortReferenceReplay)
        }
        let width = request.replayWidth ?? "cohort"
        guard width == "cohort" || width == "canonical" else {
            throw WorkerError.badReplayWidth(width)
        }
        let seeds = try require(request.replaySeedsByStream, "replay_seeds_by_stream")
        let committed = try require(request.committedByStream, "committed_by_stream")
        guard seeds.count == committed.count else {
            throw WorkerError.batchMismatch(
                requested: seeds.count, streams: committed.count)
        }
        let engine = try runner.makeEngine(
            try engineBuild(
                batch: 1, spec: SpecConfig(mode: DecoderID.serial.rawValue),
                seedLength: seeds.map(\.count).max() ?? 0))

        var streams: [CohortReferenceReplayReport.Stream] = []
        for (slot, seed) in seeds.enumerated() {
            let journal = committed[slot]
            let reference = try engine.teacherForcedTop1(
                promptTokens: seed, continuation: journal)
            streams.append(
                CohortReferenceReplayReport.Stream(
                    slot: slot,
                    positions: zip(journal, reference).map {
                        CohortReferenceReplayReport.Position(
                            committedToken: $0, sequentialArgmax: $1)
                    }))
        }

        var response = WorkerResponse(id: request.id, ok: true, nonce: nonce)
        response.effectiveBatchSize = seeds.count
        response.cohortReferenceReplay = CohortReferenceReplayReport(
            replayWidth: width, streams: streams)
        return response
    }

    // MARK: Helpers

    private func effectiveSpec(_ requested: SpecConfig?) throws -> SpecConfig {
        guard let requested else { return SpecConfig(mode: DecoderID.serial.rawValue) }
        // A `spec` under plain v1 is refused, never honoured and never
        // ignored: the parent is asking for a surface this worker did not
        // advertise, and silently running serial while it believes it asked
        // for mtp is how a leg measures one thing and is scored as another.
        guard speculative else { throw WorkerError.speculativeProtocolRequired }
        guard runner.loadedDecoders.contains(DecoderID(requested.mode)) else {
            throw WorkerError.modeNotAdvertised(requested.mode)
        }
        // The echo is the SERVED depth, in the REQUESTED decoder's own module
        // block. The engine clamps a request above the manifest's range to its
        // ceiling (clamp, not refuse, is the ruled wire posture), so an
        // unclamped echo would let benchd seal a "depth 7" leg that ran at 6.
        var depth = requested.depth
        if requested.mtp != nil || requested.dflash != nil,
            let range = manifest.decoders.first(where: { $0.mode == requested.mode })?.depth
        {
            depth = min(max(depth, range.lowerBound), range.upperBound)
        }
        let decoderID = DecoderID(requested.mode)
        if decoderID == .mtp {
            return SpecConfig(mode: requested.mode, mtp: MtpSpec(depth: depth))
        }
        if decoderID == .dflash {
            return SpecConfig(mode: requested.mode, dflash: DFlashSpec(depth: depth))
        }
        return SpecConfig(mode: requested.mode)
    }

    private func require<T>(_ value: T?, _ field: String) throws -> T {
        guard let value else { throw WorkerError.missingField(field) }
        return value
    }

    private func releaseSessions() {
        stepper = nil
        freeRun = nil
    }

    /// How many drains the phase-close settle tries, and the pause between
    /// two of them. A free that lands after `Memory.clearCache()` (a step's
    /// last `asyncEval` temporaries, the engine loop's teardown on its own
    /// queue, a Metal command buffer's completion handler) arrives
    /// milliseconds later; drains issued back to back with no pause all
    /// run before it and each reads the same straggler. The budget is
    /// bounded, so a worker that really leaks still reports a non-zero
    /// cache and the parent fails the run closed.
    static let drainSettleAttempts = 50
    /// See ``drainSettleAttempts``.
    static let drainSettleInterval: Duration = .milliseconds(20)

    /// The phase-close release: shut the free-run engine DOWN before the
    /// drain, then drain until the allocator is quiescent. Returns the
    /// free-buffer cache size the LAST drain left, which is what goes on
    /// the wire as `cache_memory`.
    ///
    /// WHY. Dropping the last reference to an engine does not free its
    /// arrays at that instant: the loop's task tears them down on its own
    /// thread, and frees that land after the drain sit in the cache the
    /// barrier then reads (a box saw 384 bytes on a resident, down from
    /// 320 KiB before drains settled the stream). Awaiting shutdown puts the
    /// teardown before the drain. Four drains with no pause between them
    /// did not cover the stragglers that remained: on 2026-09-25 six ranked
    /// runs failed the parent's `cache_memory == 0` barrier with 240 KiB to
    /// 3.7 MiB left, every one on the candidate's warm-up leg. The settle
    /// now waits ``drainSettleInterval`` between drains, for at most
    /// ``drainSettleAttempts`` drains, and stops at the first zero read.
    private func releaseSessionsAndSettle() async -> Int? {
        if let session = freeRun {
            await session.engine.shutdown()
        }
        releaseSessions()
        var cache: Int?
        for attempt in 1 ... Self.drainSettleAttempts {
            memory.drain()
            cache = memory.cacheMemoryAfterDrain()
            if (cache ?? 0) == 0 { break }
            if attempt < Self.drainSettleAttempts {
                try? await Task.sleep(for: Self.drainSettleInterval)
            }
        }
        return cache
    }

    private static let greedySampling = CBv2SamplingParams(temperature: 0, topP: 1, topK: 0)

    /// DIAGNOSTIC knobs, environment only, never on a scored path by
    /// default. `BENCH_WORKER_DIAG_PREFILL_CHUNK` sets the engine's prefill
    /// chunk size for free runs (default: the scheduler's 512);
    /// `BENCH_WORKER_DIAG_STEPPER_CHUNK` prefills the teacher-forced row in
    /// chunks of that many tokens (default: one forward). They exist to swap
    /// the two prefill SHAPES on a box where the free run and the
    /// teacher-forced path disagree after a long seed and the fixture model
    /// cannot reproduce it. A set knob is reported on the hello's stderr
    /// summary path by name so no run can carry one silently.
    static let diagnosticPrefillChunk: Int? = positiveEnvironment(
        "BENCH_WORKER_DIAG_PREFILL_CHUNK")
    static let diagnosticStepperChunk: Int? = positiveEnvironment(
        "BENCH_WORKER_DIAG_STEPPER_CHUNK")

    private static func positiveEnvironment(_ name: String) -> Int? {
        guard let raw = ProcessInfo.processInfo.environment[name],
            let value = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)), value > 0
        else { return nil }
        FileHandle.standardError.write(
            Data("bench-worker: DIAGNOSTIC \(name)=\(value) (not a scored configuration)\n".utf8))
        return value
    }
}

/// Worker-level refusals. Every one is an `ok:false` line.
public enum WorkerError: Error, CustomStringConvertible, Equatable {
    case missingField(String)
    case modeNotAdvertised(String)
    case capabilityNotAdvertised(String)
    case batchMismatch(requested: Int, streams: Int)
    case noFreeRunSession
    case streamEndedEarly(slot: Int, got: Int, want: Int, reason: String)
    case badReplayWidth(String)
    case speculativeProtocolRequired

    public var description: String {
        switch self {
        case .missingField(let field):
            return "request is missing \(field)"
        case .modeNotAdvertised(let mode):
            return "spec mode \(mode) was not advertised"
        case .capabilityNotAdvertised(let capability):
            return "\(capability) is not served by this adapter"
        case .batchMismatch(let requested, let streams):
            return "batch_size \(requested) disagrees with \(streams) streams"
        case .noFreeRunSession:
            return "free_decode_run without a free_decode_begin"
        case .streamEndedEarly(let slot, let got, let want, let reason):
            return "stream \(slot) produced \(got) of \(want) tokens (finished: \(reason))"
        case .badReplayWidth(let width):
            return "replay_width \(width) is not cohort or canonical"
        case .speculativeProtocolRequired:
            return "spec requires --speculative-protocol "
                + "\(SpeculativeProtocol.version)"
        }
    }
}
