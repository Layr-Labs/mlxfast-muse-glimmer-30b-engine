// Copyright © 2026 Eigen Labs.
//
// MLXRunners — one verified round as the runner layer states it.
//
// A free-run window must state, round by round, how many tokens each verify
// round committed, so a consumer can prove the window's tokens came from
// verify rounds rather than from plain decode. The batching contracts publish
// per-window MTP counters and a drafter's own round callbacks, but no
// per-round record: neither a counter delta nor a draft callback can say what
// one round's committed width was. This layer may not add a record to the
// library, so the seam lives here. An engine build that CAN name its rounds
// conforms to `RunnerRoundJournaling`; the bench-worker states an audit ONLY
// from records it was handed. No records means no audit, never a width
// distribution inferred from a token count.

import Foundation

/// One verify round for one row: what it drafted, what it committed.
public struct RunnerRoundRecord: Sendable, Equatable {
    /// The request the round belonged to, by the id the engine was given.
    public var requestID: UInt64
    /// Draft tokens the round proposed for this row.
    public var drafted: Int
    /// Draft tokens that survived verification for this row, BEFORE the
    /// cohort's one common committed width clips them.
    public var accepted: Int
    /// The width this round committed for every active row.
    public var committed: Int

    public init(requestID: UInt64, drafted: Int, accepted: Int, committed: Int) {
        self.requestID = requestID
        self.drafted = drafted
        self.accepted = accepted
        self.committed = committed
    }
}

/// An engine build that keeps a per-round journal of its verify rounds.
///
/// Records are delivered in finalize order, one per (row, round), oldest
/// first. A row that finished early simply has no record past its last one.
public protocol RunnerRoundJournaling: AnyObject {
    /// The log as it stands: every record still retained, oldest first, and
    /// how many older ones it has evicted.
    func roundJournal() -> RunnerRoundJournal
    /// The journal's retention cap.
    var roundRecordCap: Int { get }
}

/// One reading of a journal: what it still holds, and what it dropped.
///
/// `dropped` is the count of records evicted BEFORE `records.first`, so the
/// record at index `i` was appended as number `dropped + i`. A reader that
/// tracks its position in that numbering can tell a FULL journal (nothing
/// lost: the position is still inside the retained range) from a TRUNCATED
/// one (the position fell off the head, and coverage can no longer be
/// proven). Comparing the retained COUNT against the cap cannot tell the
/// two apart, and refuses a window that lost nothing.
public struct RunnerRoundJournal: Sendable, Equatable {
    public var records: [RunnerRoundRecord]
    public var dropped: UInt64

    public init(records: [RunnerRoundRecord] = [], dropped: UInt64 = 0) {
        self.records = records
        self.dropped = dropped
    }

    /// The sequence number of the next record the journal will append.
    public var nextSequence: UInt64 { dropped + UInt64(records.count) }
}
