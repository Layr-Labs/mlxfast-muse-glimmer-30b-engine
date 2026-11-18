// Copyright © 2026 Eigen Labs.
//
// The engine's per-round records reach the wire as the free-run audit, and
// the audit satisfies the four consistency checks the benchmarker applies.
// These tests hold no weights: they read the mapping and the assembler the
// bench-worker runs, on records shaped the way the engine writes them.

import Testing
import MLXLMCommon

@testable import MLXRunners

@Suite("the engine's round records become the free-run audit")
struct RunnerRoundJournalTests {

    /// One single-stream window of three verify rounds: depth 3 committing 4,
    /// then depth 3 committing 2, then depth 2 committing 1. Seven tokens.
    private static let window: [CBv2MTPRoundAuditRecord] = [
        .init(requestID: 1, drafted: 3, accepted: 3, committed: 4),
        .init(requestID: 1, drafted: 3, accepted: 1, committed: 2),
        .init(requestID: 1, drafted: 2, accepted: 1, committed: 1),
    ]

    @Test("Each library record becomes the seam record, field for field")
    func mappingIsOneForOne() {
        let records = EngineV2.runnerRoundRecords(from: Self.window)
        #expect(records.count == Self.window.count)
        for (seam, library) in zip(records, Self.window) {
            #expect(seam.requestID == library.requestID)
            #expect(seam.drafted == library.drafted)
            #expect(seam.accepted == library.accepted)
            #expect(seam.committed == library.committed)
        }
    }

    @Test("The seam's retention cap is the library's")
    func capIsTheLibrarysCap() {
        #expect(CBv2MTPRoundAuditRecord.retainedRecordCap == 8192)
    }

    @Test("Records in, the four numbers the benchmarker checks out")
    func auditSatisfiesTheBenchmarkChecks() throws {
        let records = EngineV2.runnerRoundRecords(from: Self.window)
        let audit = try #require(
            try FreeRunRoundAudit.assemble(
                records: records, slotForRequestID: [1: 0], streamCount: 1))
        // The tokens this window returned: 4 + 2 + 1.
        let tokens = 7
        // sum(acceptance_lengths) == N
        #expect(audit.acceptanceLengths == [4, 2, 1])
        #expect(audit.acceptanceLengths.reduce(0, +) == tokens)
        // drafted_total >= accepted_total
        #expect(audit.draftedTotal == 8)
        #expect(audit.acceptedTotal == 5)
        #expect(audit.draftedTotal >= audit.acceptedTotal)
        // completed_work == R + 1 (the seed step, then one step per round)
        #expect(audit.rounds == 3)
        #expect(audit.rounds + 1 == 4)
        // committed_total == N == tokens.len()
        #expect(audit.committedTotal == tokens)
    }

    @Test("No records states no audit, and the serial fallback is unchanged")
    func emptyJournalStatesNothing() throws {
        let records = EngineV2.runnerRoundRecords(from: [])
        #expect(records.isEmpty)
        // The bench-worker's non-serial path publishes nothing when it was
        // handed nothing; only a serial window states the unit fallback.
        #expect(
            try FreeRunRoundAudit.assemble(
                records: records, slotForRequestID: [1: 0], streamCount: 1) == nil)
        let serial = FreeRunRoundAudit.serial(
            committedByStream: [[51, 52, 53], [54, 55]])
        #expect(serial.acceptanceLengths == [1, 1, 1])
        #expect(serial.draftedTotal == 0)
        #expect(serial.acceptedTotal == 0)
        #expect(serial.committedTotal == 5)
        #expect(serial.activeStreamsByRound == [2, 2, 1])
    }
}
