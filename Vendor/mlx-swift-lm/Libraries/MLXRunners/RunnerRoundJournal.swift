// Copyright © 2026 Eigen Labs.
//
// MLXRunners — the engine's own round journal, read through the runner seam.
//
// `EngineV2` records one entry per finalized verify round in
// `CBv2MTPMetrics.roundAudits`, from values its finalize boundary already
// holds. This file states those entries as the seam's `RunnerRoundRecord`, one
// for one: no field is derived from the tokens a window drained, and an empty
// journal yields no records, so the bench-worker states no audit rather than a
// guessed one.

import MLXLMCommon

extension EngineV2: RunnerRoundJournaling {
    /// The library's own retention cap, so the seam reads the number the
    /// writer enforces.
    public var roundRecordCap: Int { CBv2MTPRoundAuditRecord.retainedRecordCap }

    /// The log as it stands: the records still retained and how many older
    /// ones it evicted. Read from ONE snapshot, so a record finalized between
    /// two reads cannot make the pair disagree.
    public func roundJournal() -> RunnerRoundJournal {
        guard let log = mtpMetricsSnapshot()?.roundAudits else { return RunnerRoundJournal() }
        return RunnerRoundJournal(
            records: Self.runnerRoundRecords(from: log.records), dropped: log.dropped)
    }

    /// The field mapping, one for one. Static so a test can check it without a
    /// resident model.
    static func runnerRoundRecords(
        from records: [CBv2MTPRoundAuditRecord]
    ) -> [RunnerRoundRecord] {
        records.map {
            RunnerRoundRecord(
                requestID: $0.requestID, drafted: $0.drafted,
                accepted: $0.accepted, committed: $0.committed)
        }
    }
}
