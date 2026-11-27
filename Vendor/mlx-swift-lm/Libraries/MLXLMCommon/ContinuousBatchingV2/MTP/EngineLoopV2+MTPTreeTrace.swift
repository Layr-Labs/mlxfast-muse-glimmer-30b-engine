// Copyright © 2026 Eigen Labs.

import Foundation

// Measurement-only round log (`CBV2_TREE_TRACE`, see `CBv2TreeTrace`).
// Nothing here runs unless the flag is set; nothing here changes a plan, a
// verify input or a committed token.

extension EngineLoopV2 {
    /// What a `round` line records before the round commits.
    struct MTPTraceRoundHead {
        let index: Int
        let reason: String
        let prompt: Int
        let generated: Int
        let seed: Int
        let window: CBv2MTPRoundDriver.TraceWindow
        let proposal: CBv2MTPProposalTrace?
        let nanos: UInt64
    }

    /// Reads the round's plan inputs and the drafter's block trace. Call
    /// before the row's first `recordSampled` and before `finalizeRound`
    /// (which clears the drafter's round).
    func mtpTraceRoundHead(
        rec: CBv2ScheduledRequest, assistantState: (any CBv2MTPRequestState)?,
        driver mtp: CBv2MTPRoundDriver
    ) -> MTPTraceRoundHead {
        let (reason, index) = mtp.traceTakeRound(rec.id)
        let generated = rec.generatedTokenCount
        let window = mtp.traceWindow(
            for: rec.id, remainingTokens: rec.request.maxTokens - generated)
        var proposal: CBv2MTPProposalTrace?
        if CBv2TreeTrace.tracesProposals, let state = assistantState,
            let tracer = mtp.drafter as? any CBv2MTPProposalTracing
        {
            proposal = tracer.takeProposalTrace(requestState: state)
        }
        return MTPTraceRoundHead(
            index: index, reason: reason, prompt: rec.request.promptTokens.count,
            generated: generated, seed: rec.tokens.last ?? -1, window: window,
            proposal: proposal, nanos: CBv2TreeTrace.nowNanos())
    }

    /// One `round` line. `extra`: further JSON members (no braces).
    func mtpEmitTraceRound(
        _ head: MTPTraceRoundHead, id: CBv2RequestID, k: Int, drafts: [Int], targets: [Int],
        accepted: Int, kept: [Int], extra: String? = nil
    ) {
        var members: [String] = [
            "\"ev\":\"round\"",
            "\"req\":\(id.raw)",
            "\"n\":\(head.index)",
            "\"t_ns\":\(head.nanos)",
            "\"prompt\":\(head.prompt)",
            "\"gen\":\(head.generated)",
            "\"seed\":\(head.seed)",
            "\"k\":\(k)",
            "\"reason\":" + CBv2TreeTrace.string(head.reason),
            "\"low\":\(head.window.low)",
            "\"frac\":" + CBv2TreeTrace.number(head.window.fraction),
            "\"win\":\(head.window.window)",
            "\"ceil\":\(head.window.ceiling)",
            "\"acc_p\":" + CBv2TreeTrace.list(head.window.probabilities),
            "\"drafts\":" + CBv2TreeTrace.list(drafts),
            "\"targets\":" + CBv2TreeTrace.list(targets),
            "\"acc\":\(accepted)",
            "\"kept\":" + CBv2TreeTrace.list(kept),
        ]
        if let proposal = head.proposal { members.append(proposal.jsonMembers) }
        if let extra, !extra.isEmpty { members.append(extra) }
        CBv2TreeTrace.emit("{" + members.joined(separator: ",") + "}")
    }

    /// One `tree` line (`CBV2_VERIFY_TREE` rounds of an eligible-planned
    /// request): the layout and its leaf tokens, whether the forward took
    /// every tree variant, the probe, the hit, and the sibling hit rates.
    func mtpEmitTraceTree(
        _ tree: CBv2VerifyTree.RoundTree, id: CBv2RequestID, k: Int, drafts: [Int],
        accepted: Int, confirmed: Int, leafColumn: Int?, probe: Bool?,
        rates: CBv2VerifyTree.SiblingRates
    ) {
        var leaves = "[]"
        if let layout = tree.layout {
            leaves = "[" + layout.leaves.enumerated().map { index, leaf in
                let column = layout.column(ofLeaf: index)
                let token = column - 1 < drafts.count ? drafts[column - 1] : -1
                return "[\(leaf.row),\(leaf.rank),\(token)]"
            }.joined(separator: ",") + "]"
        }
        let probeLeaf: String = tree.layout?.probe.map { String($0) } ?? "null"
        let probeText: String = probe.map { "\($0)" } ?? "null"
        let hitText: String = leafColumn.map { String($0) } ?? "null"
        let members: [String] = [
            "\"ev\":\"tree\"",
            "\"req\":\(id.raw)",
            "\"t_ns\":\(CBv2TreeTrace.nowNanos())",
            "\"k\":\(k)",
            "\"chain\":\(tree.chain)",
            "\"leaves\":" + leaves,
            "\"probe_leaf\":" + probeLeaf,
            "\"why\":" + CBv2TreeTrace.string(tree.why),
            "\"valid\":\(tree.valid)",
            "\"fwd\":" + CBv2TreeTrace.string(tree.forward),
            "\"probe\":" + probeText,
            "\"acc\":\(accepted)",
            "\"committed\":\(confirmed)",
            "\"hit\":" + hitText,
            "\"s2\":" + CBv2TreeTrace.list(rates.rank2),
            "\"s3\":" + CBv2TreeTrace.list(rates.rank3),
        ]
        CBv2TreeTrace.emit("{" + members.joined(separator: ",") + "}")
    }

    /// One `done` line: the request's generated tokens (the serial tape).
    func mtpEmitTraceDone(_ rec: CBv2ScheduledRequest, reason: CBv2FinishReason) {
        let prompt = rec.request.promptTokens.count
        let generated = rec.tokens.count > prompt ? Array(rec.tokens[prompt...]) : []
        CBv2TreeTrace.emit(
            "{\"ev\":\"done\",\"req\":\(rec.id.raw),\"t_ns\":\(CBv2TreeTrace.nowNanos())"
                + ",\"prompt\":\(prompt),\"reason\":" + CBv2TreeTrace.string("\(reason)")
                + ",\"generated\":" + CBv2TreeTrace.list(generated) + "}")
    }
}
