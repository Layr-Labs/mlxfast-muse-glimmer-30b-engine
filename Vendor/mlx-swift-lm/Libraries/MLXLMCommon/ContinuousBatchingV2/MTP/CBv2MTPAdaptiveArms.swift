// CBv2MTPAdaptiveArms.swift
//
// Online choice between exact verify policies for low-acceptance rounds.
// Every arm emits the target's own row-exact argmax stream, so switching
// arms never changes a token; only the wall time per committed token does.
// The controller is host-only arithmetic over commit-to-commit wall times
// the engine already observes at the acceptance readback (the same kind of
// host clock the depth controller's committed-goodput accounting uses).

import Foundation

/// One exact policy for a low-acceptance round.
public struct CBv2AdaptArm: Equatable, Sendable {
    public let name: String
    /// nil: plain chain (tree off). Otherwise the deepest dynamic-leaf depth.
    public let leafDepth: Int?
    public init(name: String, leafDepth: Int?) {
        self.name = name
        self.leafDepth = leafDepth
    }
    public static let chain = CBv2AdaptArm(name: "A0", leafDepth: nil)
    public static let leaf1 = CBv2AdaptArm(name: "A1", leafDepth: 1)
    public static let leaf3 = CBv2AdaptArm(name: "A2", leafDepth: 3)

    /// `CBV2_ADAPT_ARMS` = comma list of A0,A1,A2 (default all three).
    static func parse(_ text: String?) -> [CBv2AdaptArm] {
        let all = [chain, leaf1, leaf3]
        guard let text else { return [chain, leaf1, leaf3] }
        let picked = text.split(separator: ",").compactMap { token in
            all.first { $0.name == token.trimmingCharacters(in: .whitespaces).uppercased() }
        }
        return picked.isEmpty ? [chain, leaf1, leaf3] : picked
    }
}

public struct CBv2AdaptConfig: Sendable {
    public var arms: [CBv2AdaptArm]
    /// Consecutive rounds per block (the first of each block only anchors).
    public var blockRounds = 8
    /// First-pass block: 6 measured rounds plus the anchor round.
    public var firstBlockRounds = 7
    /// Eliminate an arm whose ms per token is this much worse than the best.
    public var eliminateMargin = 0.12
    /// Rounds between probes of an eliminated arm (doubles per failed probe).
    public var probeEvery = 300
    /// Rounds in a probe block of an eliminated arm.
    public var probeRounds = 4
    /// Blocks per arm before exploiting.
    public var exploreBlocks = 3
    /// One recheck block every this many blocks while exploiting.
    public var recheckEvery = 10
    /// Another arm must beat the current one by this fraction to switch.
    public var switchMargin = 0.03
    /// Per-observation decay of an arm's statistics.
    public var decay = 0.98

    public init(arms: [CBv2AdaptArm]) { self.arms = arms }

    static func fromEnvironment(_ env: [String: String] = ProcessInfo.processInfo.environment)
        -> CBv2AdaptConfig?
    {
        guard env["CBV2_ADAPT"] != "0" else { return nil }
        var c = CBv2AdaptConfig(arms: CBv2AdaptArm.parse(env["CBV2_ADAPT_ARMS"]))
        if let r = env["CBV2_ADAPT_BLOCK"].flatMap({ Int($0) }), r >= 1 { c.blockRounds = r }
        if let e = env["CBV2_ADAPT_EXPLORE"].flatMap({ Int($0) }), e >= 1 { c.exploreBlocks = e }
        return c
    }
}

/// Per-request controller. `nextArm()` is called at plan time once per
/// adaptive round; `observe` at the acceptance readback with that round's
/// arm. Plan and observe may interleave (a round can be planned before the
/// previous one is read back), so the schedule follows plan order and the
/// statistics follow the arm tag each observation carries.
public struct CBv2AdaptController {
    public let config: CBv2AdaptConfig
    private var sumWall: [Double]
    private var sumTokens: [Double]
    private var rounds: [Int]
    private var plannedRounds = 0
    private var block = 0
    private var inBlock = 0
    private var blockArm = 0
    private var best = 0
    private var recheckCursor = 0
    private var exploited = false
    private var blocksPlanned: [Int] = []
    private var alive: [Bool] = []
    private var currentBlockRounds = 7
    private var nextProbeAt = 0
    private var probeInterval = 150
    private var probeCursor = 0
    private var exploitBlocks = 0
    public private(set) var probes = 0
    public private(set) var switches = 0
    // Commit-to-commit anchor.
    private var lastNanos: UInt64?
    private var lastArm = -1
    private var expectedCount: Int?
    private var meanRoundNanos = 0.0

    public init(config: CBv2AdaptConfig) {
        self.config = config
        let n = config.arms.count
        sumWall = Array(repeating: 0, count: n)
        sumTokens = Array(repeating: 0, count: n)
        rounds = Array(repeating: 0, count: n)
        blockArm = 0
        blocksPlanned = Array(repeating: 0, count: n)
        alive = Array(repeating: true, count: n)
        currentBlockRounds = config.firstBlockRounds
        probeInterval = config.probeEvery
        nextProbeAt = config.probeEvery
    }

    public func msPerToken(_ arm: Int) -> Double? {
        sumTokens[arm] > 0 ? sumWall[arm] / sumTokens[arm] / 1e6 : nil
    }

    private mutating func chooseBlock() {
        let n = config.arms.count
        currentBlockRounds = config.blockRounds
        // Pass 0: one short block per arm, in arm order (A0 first).
        if let first = (0 ..< n).first(where: { blocksPlanned[$0] == 0 }) {
            blockArm = first
            currentBlockRounds = config.firstBlockRounds
            blocksPlanned[first] += 1
            return
        }
        let measured = (0 ..< n).compactMap { a in msPerToken(a).map { (a, $0) } }
        let low = measured.min(by: { $0.1 < $1.1 })
        // Successive elimination against the current best; arms without
        // data stay alive.
        if let low {
            for a in 0 ..< n {
                alive[a] = msPerToken(a).map { $0 <= low.1 * (1 + config.eliminateMargin) } ?? true
            }
        }
        // Continue exploring survivors up to exploreBlocks blocks each.
        let pending = (0 ..< n).filter { alive[$0] && blocksPlanned[$0] < config.exploreBlocks }
        if let a = pending.min(by: { (blocksPlanned[$0], $0) < (blocksPlanned[$1], $1) }) {
            blockArm = a
            blocksPlanned[a] += 1
            return
        }
        // Exploit: incumbent keeps the lead unless beaten by the margin.
        var cand = best
        if let low {
            if !exploited || msPerToken(best) == nil {
                cand = low.0
            } else if let cur = msPerToken(best), low.1 < cur * (1 - config.switchMargin) {
                cand = low.0
            }
        }
        exploited = true
        if cand != best { best = cand; switches += 1 }
        exploitBlocks += 1
        blockArm = best
        blocksPlanned[best] += 1
        let eliminated = (0 ..< n).filter { !alive[$0] }
        let survivors = (0 ..< n).filter { alive[$0] && $0 != best }
        if plannedRounds >= nextProbeAt, !eliminated.isEmpty {
            // Probe an eliminated arm at most once per (doubling) interval.
            probeCursor += 1
            blockArm = eliminated[probeCursor % eliminated.count]
            currentBlockRounds = config.probeRounds
            probeInterval *= 2
            nextProbeAt = plannedRounds + probeInterval
            probes += 1
        } else if !survivors.isEmpty, exploitBlocks % config.recheckEvery == 0 {
            probeCursor += 1
            blockArm = survivors[probeCursor % survivors.count]
        }
    }

    /// Arm index for the next round of this request.
    public mutating func nextArm() -> Int {
        if plannedRounds == 0 { chooseBlock() }
        if inBlock >= currentBlockRounds {
            block += 1
            inBlock = 0
            chooseBlock()
        }
        inBlock += 1
        plannedRounds += 1
        return blockArm
    }

    /// Break the interval chain (a seed or plain step intervened).
    public mutating func resetAnchor() { lastNanos = nil; expectedCount = nil; lastArm = -1 }

    /// A round read back at `nowNanos` that committed `tokens` tokens;
    /// `countBefore` is the request's generated-token count before it.
    public mutating func observe(
        arm: Int, nowNanos: UInt64, tokens: Int, countBefore: Int
    ) {
        defer {
            lastNanos = nowNanos; lastArm = arm; expectedCount = countBefore + tokens
        }
        guard let last = lastNanos, nowNanos > last, tokens > 0,
            expectedCount == countBefore, lastArm == arm
        else { return }  // first round of a block (or after a gap) only anchors
        var wall = Double(nowNanos - last)
        if meanRoundNanos > 0 { wall = min(wall, 4 * meanRoundNanos) }
        meanRoundNanos = meanRoundNanos == 0 ? wall : 0.9 * meanRoundNanos + 0.1 * wall
        for a in 0 ..< config.arms.count where a == arm {
            sumWall[a] = sumWall[a] * config.decay + wall
            sumTokens[a] = sumTokens[a] * config.decay + Double(tokens)
            rounds[a] += 1
        }
    }

    public func summary() -> String {
        config.arms.enumerated().map { i, arm in
            let m = msPerToken(i).map { String(format: "%.2f", $0) } ?? "-"
            return "\(arm.name)=\(m)ms/tok(\(rounds[i])r)"
        }.joined(separator: " ") + " best=\(config.arms[best].name) switches=\(switches) probes=\(probes)"
    }
}
