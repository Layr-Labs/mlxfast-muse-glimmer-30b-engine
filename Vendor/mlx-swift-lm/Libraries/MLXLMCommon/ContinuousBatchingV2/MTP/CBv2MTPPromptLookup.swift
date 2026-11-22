// CBv2MTPPromptLookup.swift
//
// Prompt-lookup (n-gram) proposals for the speculative round, and the per-
// request chooser between them and the drafter's own proposals.
//
// Propose-and-decide: this only changes WHICH draft tokens enter the verify
// window. The pinned target scores every column and the accept walk keeps a
// draft only when it equals the target's own token, so output is unchanged.
// The algorithm reads only the request's own committed tokens (prompt plus
// generated so far); it holds no cross-request state and no prompt-keyed
// cache. `MUSE_PROMPT_LOOKUP=0` switches it off.

import Foundation

struct CBv2MTPPromptLookupChain {
    /// Tokens that followed the most recent earlier occurrence of the suffix.
    var tokens: [Int32]
    /// Suffix length that matched (2...4).
    var n: Int
}

/// What one round proposed, kept for the round's finalize.
struct CBv2MTPPromptLookupRound {
    var chain: CBv2MTPPromptLookupChain?
    /// The chain's tokens replace the drafter's at positions `0 ..< used`
    /// (0: the drafter's proposals stand).
    var used: Int
}

struct CBv2MTPPromptLookupState {
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MUSE_PROMPT_LOOKUP"] else {
            return true
        }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()
    static let maxSuffix = 4
    static let minSuffix = 2
    /// Newest tokens searched for an earlier occurrence.
    static let searchLimit = 12288
    static let alpha = 0.3
    /// Accepted-length priors per matched suffix length (index n - 2): a
    /// longer suffix is a stronger match. The drafter's prior sits at the
    /// DFlash mean on the hard prompts.
    static let lookupPrior: [Double] = [1.2, 1.8, 2.4]
    static let drafterPrior = 2.0

    /// EWMA of accepted drafts per round for the lookup chain, by suffix
    /// length, and for the drafter. Counterfactual: updated whenever the
    /// round could score the source against the target's tokens.
    var lookup: [Double] = CBv2MTPPromptLookupState.lookupPrior
    var drafter = CBv2MTPPromptLookupState.drafterPrior

    /// Most recent earlier occurrence of the last n tokens (n = 4, 3, 2) and
    /// the (up to `depth`) tokens that followed it.
    static func chain(history: [Int], depth: Int) -> CBv2MTPPromptLookupChain? {
        let count = history.count
        guard depth > 0, count > minSuffix else { return nil }
        let lowest = max(0, count - searchLimit)
        return history.withUnsafeBufferPointer { h -> CBv2MTPPromptLookupChain? in
            var n = min(maxSuffix, count - 1)
            while n >= minSuffix {
                let tail = count - n
                var j = count - n - 1
                while j >= lowest {
                    if h[j] == h[tail] {
                        var same = true
                        var i = 1
                        while i < n {
                            if h[j + i] != h[tail + i] { same = false; break }
                            i += 1
                        }
                        if same {
                            let end = min(j + n + depth, count)
                            let followed = (j + n ..< end).map { Int32(truncatingIfNeeded: h[$0]) }
                            if !followed.isEmpty {
                                return CBv2MTPPromptLookupChain(tokens: followed, n: n)
                            }
                        }
                    }
                    j -= 1
                }
                n -= 1
            }
            return nil
        }
    }

    /// Expected accepted drafts of the chain, capped by its length.
    func lookupExpectation(_ chain: CBv2MTPPromptLookupChain) -> Double {
        min(lookup[chain.n - Self.minSuffix], Double(chain.tokens.count))
    }

    /// Use the chain when its expected accepted length beats the drafter's.
    func prefersLookup(_ chain: CBv2MTPPromptLookupChain) -> Bool {
        lookupExpectation(chain) > drafter
    }

    /// Leading matches of `proposal` against the target's tokens, looking
    /// no further than the position the round actually decided
    /// (`accepted`, inclusive): beyond it the target's tokens followed a
    /// rejected draft.
    static func countedAccepted(proposal: [Int32], targets: [Int], accepted: Int) -> Int {
        var a = 0
        let limit = min(proposal.count, accepted + 1, targets.count)
        while a < limit, Int(proposal[a]) == targets[a] { a += 1 }
        return a
    }

    mutating func observeLookup(chain: CBv2MTPPromptLookupChain, length: Int) {
        let i = chain.n - Self.minSuffix
        lookup[i] += Self.alpha * (Double(length) - lookup[i])
    }

    mutating func observeDrafter(length: Int) {
        drafter += Self.alpha * (Double(length) - drafter)
    }
}
