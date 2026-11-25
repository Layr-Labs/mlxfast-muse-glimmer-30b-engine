// CBv2VerifyTree.swift
//
// Draft-tree verify windows: a chain of `chain` drafts plus leaf columns,
// each leaf the drafter's second choice at one chain depth. A leaf column
// sits after the chain in storage but attends the prefix up to its parent
// and itself, at its logical position. When the chain stops at depth d and
// the target's own token there equals the depth-d leaf, the leaf's row is
// the target's next step: the round commits one more token.

import Foundation

public struct CBv2VerifyTreeLayout: Hashable, Sendable {
    /// Chain drafts (verify columns 1 ... chain).
    public let chain: Int
    /// Logical depth (1 ... chain) of each leaf column, in column order.
    public let leafDepths: [Int]

    public init(chain: Int, leafDepths: [Int]) {
        precondition(chain >= 1 && leafDepths.allSatisfy { (1 ... chain).contains($0) })
        self.chain = chain
        self.leafDepths = leafDepths
    }

    /// Draft columns in the window (`k`).
    public var drafts: Int { chain + leafDepths.count }
    /// Verify columns, seed included.
    public var columns: Int { 1 + drafts }

    /// Logical position offset of verify column `t` from the seed column.
    public func depth(column t: Int) -> Int {
        t <= chain ? t : leafDepths[t - chain - 1]
    }

    /// Verify column of the leaf at `depth`, if any.
    public func leafColumn(depth: Int) -> Int? {
        leafDepths.firstIndex(of: depth).map { 1 + chain + $0 }
    }
}

/// The layout of the verify forward being built. Set by the engine around
/// one round's target graph construction; layers read it while building.
public enum CBv2VerifyTree {
    /// `CBV2_VERIFY_TREE=0` turns tree windows off.
    public static let enabled =
        ProcessInfo.processInfo.environment["CBV2_VERIFY_TREE"] != "0"
    /// Largest chain that gets leaves.
    public static let maximumTreeChain: Int = {
        if let raw = ProcessInfo.processInfo.environment["CBV2_VERIFY_TREE_CHAIN"],
            let value = Int(raw)
        {
            return value
        }
        return 3
    }()

    nonisolated(unsafe) public private(set) static var current: CBv2VerifyTreeLayout?
    /// A layer built the window without the tree's positions or masks.
    nonisolated(unsafe) public static var failed = false
    /// Attention calls that applied the leaf masks.
    nonisolated(unsafe) public static var maskedAttentionCalls = 0
    /// RoPE calls that applied the leaf positions.
    nonisolated(unsafe) public static var rowPositionCalls = 0

    public static func begin(_ layout: CBv2VerifyTreeLayout?) {
        current = layout
        failed = false
        maskedAttentionCalls = 0
        rowPositionCalls = 0
    }

    /// Ends construction; true when every layer honored the layout.
    public static func end() -> Bool {
        let ok =
            current != nil && !failed && maskedAttentionCalls > 0 && rowPositionCalls > 0
        current = nil
        return ok
    }

    /// The active layout when a call over `columns` verify columns builds.
    public static func active(columns: Int) -> CBv2VerifyTreeLayout? {
        guard let current, current.columns == columns else { return nil }
        return current
    }

    /// The layouts `CBv2MTPRoundDriver.treeLayout` can produce under a
    /// declared depth (for load-time kernel warm-up).
    public static func reachableLayouts(maxDrafts: Int) -> [CBv2VerifyTreeLayout] {
        guard enabled else { return [] }
        return (1 ... max(1, maximumTreeChain)).compactMap { chain in
            let leaves = min(chain, maxDrafts - chain)
            guard chain < maxDrafts, leaves >= 1 else { return nil }
            return CBv2VerifyTreeLayout(chain: chain, leafDepths: Array(1 ... leaves))
        }
    }

    /// Builds `body` as if `layout` were the window being verified.
    public static func building<T>(_ layout: CBv2VerifyTreeLayout, _ body: () -> T) -> T {
        let saved = (current, failed, maskedAttentionCalls, rowPositionCalls)
        current = layout
        defer { (current, failed, maskedAttentionCalls, rowPositionCalls) = saved }
        return body()
    }

    /// Per-column logical offsets `[columns]` int32.
    public static func rowOffsets(_ layout: CBv2VerifyTreeLayout) -> [Int32] {
        (0 ..< layout.columns).map { Int32(layout.depth(column: $0)) }
    }
}
