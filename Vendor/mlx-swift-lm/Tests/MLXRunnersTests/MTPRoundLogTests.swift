// MTPRoundLogTests.swift
//
// The round log is a bounded ring: appending past its cap must overwrite the
// oldest record instead of shifting the array, and it must keep reporting the
// retained records in append order. No weights, no engine.

import MLXLMCommon
import Testing

@testable import MLXRunners

@Suite("the engine's round log is a bounded ring")
struct MTPRoundLogTests {

    private static func record(_ committed: Int, row: UInt64 = 1)
        -> CBv2MTPRoundAuditRecord
    {
        CBv2MTPRoundAuditRecord(
            requestID: row, drafted: 2, accepted: 1, committed: committed)
    }

    /// Until the cap, the ring is just an append-ordered log.
    @Test("Below the cap the log holds every record in append order")
    func appendsInOrder() {
        var log = CBv2MTPRoundAuditLog()
        for value in 1 ... 5 { log.append(Self.record(value)) }
        #expect(log.dropped == 0)
        #expect(log.count == 5)
        #expect(log.records.map(\.committed) == [1, 2, 3, 4, 5])
        // The record at index i was appended as number dropped + i.
        #expect(log.records.first?.committed == 1)
    }

    /// Past the cap the OLDEST record goes, one per append, and the eviction
    /// count says how many did — which is what separates a full log from a
    /// truncated one.
    @Test("Past the cap the oldest record is evicted and counted")
    func evictsOldest() {
        let cap = CBv2MTPRoundAuditRecord.retainedRecordCap
        var log = CBv2MTPRoundAuditLog()
        for value in 0 ..< (cap + 3) { log.append(Self.record(value)) }
        #expect(log.count == cap)
        #expect(log.dropped == 3)
        #expect(log.records.count == cap)
        // Oldest first: the first three were evicted, the newest is last.
        #expect(log.records.first?.committed == 3)
        #expect(log.records.last?.committed == cap + 2)
    }

    /// Exactly at the cap nothing has been dropped, so a reader whose cursor is
    /// still at zero is fully covered. This is the state the worker used to
    /// refuse.
    @Test("A log exactly at its cap has dropped nothing")
    func fullIsNotTruncated() {
        let cap = CBv2MTPRoundAuditRecord.retainedRecordCap
        var log = CBv2MTPRoundAuditLog()
        for value in 0 ..< cap { log.append(Self.record(value)) }
        #expect(log.count == cap)
        #expect(log.dropped == 0)
        #expect(log.records.first?.committed == 0)
        #expect(log.records.last?.committed == cap - 1)
    }
}
