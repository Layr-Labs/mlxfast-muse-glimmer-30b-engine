// CBv2VerifyTree.swift
//
// Draft-tree verify windows: a chain of `chain` drafts plus branches. A
// branch starts at one chain depth with the drafter's second choice there
// and continues with the drafter's own proposals for the following depths
// (the block proposes every position from the anchor, so those proposals
// stand for the corrected path too). Branch columns sit after the chain in
// storage; each attends the prefix up to its parent depth, its own branch
// so far and itself, at its logical position. When the chain stops at
// depth d and the target's own token there opens the depth-d branch, the
// branch's matching prefix and the target token after it are committed.

import Foundation

public struct CBv2VerifyTreeLayout: Hashable, Sendable {
    /// Chain drafts (verify columns 1 ... chain).
    public let chain: Int
    /// Depth (1 ... chain) where each branch starts, in column order.
    public let branchDepths: [Int]
    /// Columns of each branch (at least 1).
    public let branchLengths: [Int]

    public init(chain: Int, branchDepths: [Int], branchLengths: [Int]) {
        precondition(
            chain >= 1 && branchDepths.count == branchLengths.count
                && branchDepths.allSatisfy { (1 ... chain).contains($0) }
                && branchLengths.allSatisfy { $0 >= 1 })
        self.chain = chain
        self.branchDepths = branchDepths
        self.branchLengths = branchLengths
    }

    /// Draft columns in the window (`k`).
    public var drafts: Int { chain + branchLengths.reduce(0, +) }
    /// Verify columns, seed included.
    public var columns: Int { 1 + drafts }

    /// First verify column of branch `b`.
    public func branchStart(_ b: Int) -> Int {
        1 + chain + branchLengths[0 ..< b].reduce(0, +)
    }

    /// Branch and offset in it of verify column `t` (nil for the chain).
    public func branchPosition(column t: Int) -> (branch: Int, offset: Int)? {
        guard t > chain else { return nil }
        var start = 1 + chain
        for (b, length) in branchLengths.enumerated() {
            if t < start + length { return (b, t - start) }
            start += length
        }
        return nil
    }

    /// Logical position offset of verify column `t` from the seed column.
    public func depth(column t: Int) -> Int {
        guard let p = branchPosition(column: t) else { return t }
        return branchDepths[p.branch] + p.offset
    }

    /// Skipped columns `[from, to)` relative to the seed column: a branch
    /// column skips the chain from its start depth and every earlier branch.
    public func hole(column t: Int) -> (from: Int, to: Int)? {
        guard let p = branchPosition(column: t) else { return nil }
        return (branchDepths[p.branch], branchStart(p.branch))
    }

    /// The branch that starts at `depth`, if any.
    public func branch(depth: Int) -> Int? { branchDepths.firstIndex(of: depth) }

    /// Block position whose proposal fills verify column `t` of a branch:
    /// offset 0 is the second choice at the start depth, offset j the
    /// block's own proposal j positions deeper.
    public func proposalIndex(column t: Int) -> (index: Int, alternative: Bool)? {
        guard let p = branchPosition(column: t) else { return nil }
        return (branchDepths[p.branch] - 1 + p.offset, p.offset == 0)
    }
}

/// The layout of the verify forward being built. Set by the engine around
/// one round's target graph construction; layers read it while building.
public enum CBv2VerifyTree {
    private static let environment = ProcessInfo.processInfo.environment
    /// `CBV2_VERIFY_TREE=1` turns tree windows on. Off by default: on the
    /// ranked box the extra verify rows cost more than second choices gained
    /// (73369b36: hard-prompt decode 20.2 -> 23.2 and 20.3 -> 21.1 ms/token).
    public static let enabled = environment["CBV2_VERIFY_TREE"] == "1"
    /// Largest chain that gets branches.
    public static let maximumTreeChain: Int =
        environment["CBV2_VERIFY_TREE_CHAIN"].flatMap { Int($0) } ?? 3
    /// Longest branch (1: second-choice leaves only).
    public static let maximumBranchLength: Int =
        max(1, environment["CBV2_VERIFY_TREE_BRANCH"].flatMap { Int($0) } ?? 2)

    /// The tree for a chain of `chain` drafts within `maxDrafts` drafts and
    /// a block of `blockDrafts` proposals: a leaf at every depth 1 ...
    /// chain while the window allows, then branches grown one column at a
    /// time, shallowest first, up to `maximumBranchLength`.
    public static func layout(chain: Int, maxDrafts: Int, blockDrafts: Int)
        -> CBv2VerifyTreeLayout?
    {
        guard enabled, chain >= 1, chain <= maximumTreeChain, chain < maxDrafts else {
            return nil
        }
        var budget = maxDrafts - chain
        let leaves = min(chain, budget)
        guard leaves >= 1 else { return nil }
        let depths = Array(1 ... leaves)
        var lengths = Array(repeating: 1, count: leaves)
        budget -= leaves
        var grew = true
        while budget > 0, grew {
            grew = false
            for b in depths.indices where budget > 0 {
                // The block holds proposals for positions 0 ..< blockDrafts.
                guard lengths[b] + 1 <= maximumBranchLength,
                    depths[b] - 1 + lengths[b] < blockDrafts
                else { continue }
                lengths[b] += 1
                budget -= 1
                grew = true
            }
        }
        return CBv2VerifyTreeLayout(chain: chain, branchDepths: depths, branchLengths: lengths)
    }

    nonisolated(unsafe) public private(set) static var current: CBv2VerifyTreeLayout?
    /// A layer built the window without the tree's positions or masks.
    nonisolated(unsafe) public static var failed = false
    /// Attention calls that applied the branch masks.
    nonisolated(unsafe) public static var maskedAttentionCalls = 0
    /// RoPE calls that applied the branch positions.
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

    /// The layouts `layout` can produce under a declared depth (for
    /// load-time kernel warm-up).
    public static func reachableLayouts(maxDrafts: Int) -> [CBv2VerifyTreeLayout] {
        guard enabled else { return [] }
        return (1 ... max(1, maximumTreeChain)).compactMap {
            layout(chain: $0, maxDrafts: maxDrafts, blockDrafts: maxDrafts)
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
