// Copyright © 2026 Eigen Labs.

import Foundation

/// Measurement-only round log for the draft-tree study (`CBV2_TREE_TRACE`).
///
/// Unset, empty or `0`: nothing is computed, nothing is read back, every
/// served path is the one without this file (`enabled` is a constant read
/// once). `1` or `stderr`: one JSON object per line on standard error, each
/// prefixed `CBV2TREE `. Any other value: a file path, appended to (falls
/// back to standard error when it cannot be opened).
///
/// Lines (`ev` names the kind):
/// - `round`: one verified MTP round of one request: the plan's window and
///   why, the request's regime, the drafts, the target tokens, what was
///   committed, and the drafter's top entries at every block row
///   (`CBv2MTPProposalTrace`).
/// - `tree`: one tree round (`CBV2_VERIFY_TREE`): the layout, whether a leaf
///   committed, and why leaves were not armed.
/// - `done`: a finished request's generated tokens, so the serial tape can be
///   rebuilt offline and every block row's true next token is known.
///
/// Tracing adds host readbacks and changes timing; the window and every
/// committed token depend only on accept/reject history, never on time.
public enum CBv2TreeTrace {
    public static let enabled: Bool = {
        guard let value = ProcessInfo.processInfo.environment["CBV2_TREE_TRACE"],
            !value.isEmpty, value != "0"
        else { return false }
        return true
    }()

    /// Top entries per block row the drafter records.
    public static let width = 4

    /// The drafter's top entries are computed and read back (`round` lines
    /// carry them); `CBV2_TREE_TRACE_TOPK=0` keeps only the round outcome
    /// and its host timestamp (cheaper, for per-row cost curves).
    public static let tracesProposals: Bool =
        enabled && ProcessInfo.processInfo.environment["CBV2_TREE_TRACE_TOPK"] != "0"

    /// Measurement only (`CBV2_TREE_FORCE_WINDOW`=1...7): every verify round
    /// runs that many drafts, bypassing the acceptance window and ceiling
    /// (still at most the declared depth). The committed tokens are the same
    /// under exact acceptance; only the verify width changes. Never set in a
    /// ranked run: it removes the pair-1 ceiling.
    public static let forcedWindow: Int? = {
        // Only together with the trace: a stray variable alone never changes a plan.
        guard enabled,
            let value = ProcessInfo.processInfo.environment["CBV2_TREE_FORCE_WINDOW"],
            let depth = Int(value), (1 ... 7).contains(depth)
        else { return nil }
        return depth
    }()

    /// Host monotonic nanoseconds, for per-round wall intervals.
    public static func nowNanos() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    private final class Sink: @unchecked Sendable {
        let lock = NSLock()
        let handle: FileHandle
        let prefix: String

        init() {
            let value = ProcessInfo.processInfo.environment["CBV2_TREE_TRACE"] ?? ""
            if value != "1", value != "stderr", value.hasPrefix("/") {
                if !FileManager.default.fileExists(atPath: value) {
                    _ = FileManager.default.createFile(atPath: value, contents: nil)
                }
                if let handle = FileHandle(forWritingAtPath: value) {
                    _ = try? handle.seekToEnd()
                    self.handle = handle
                    self.prefix = ""
                    return
                }
            }
            self.handle = FileHandle.standardError
            self.prefix = "CBV2TREE "
        }

        func write(_ line: String) {
            guard let data = (prefix + line + "\n").data(using: .utf8) else { return }
            lock.lock()
            defer { lock.unlock() }
            try? handle.write(contentsOf: data)
        }
    }

    private static let sink = Sink()

    /// Writes one line. Call only when `enabled`.
    public static func emit(_ line: String) {
        guard enabled else { return }
        sink.write(line)
    }

    /// A finite number, or `null`: JSON has no infinities.
    public static func number(_ value: Float) -> String {
        value.isFinite ? "\(value)" : "null"
    }

    public static func number(_ value: Double) -> String {
        value.isFinite ? "\(value)" : "null"
    }

    public static func list(_ values: [Int]) -> String {
        "[" + values.map(String.init).joined(separator: ",") + "]"
    }

    public static func list(_ values: [Int32]) -> String {
        "[" + values.map { String($0) }.joined(separator: ",") + "]"
    }

    public static func list(_ values: [Float]) -> String {
        "[" + values.map { number($0) }.joined(separator: ",") + "]"
    }

    public static func list(_ values: [Double]) -> String {
        "[" + values.map { number($0) }.joined(separator: ",") + "]"
    }

    /// Rows of `width` entries each as a JSON list of lists.
    public static func rows<T>(_ values: [T], width: Int, _ item: (T) -> String) -> String {
        guard width > 0 else { return "[]" }
        var parts: [String] = []
        var index = 0
        while index < values.count {
            let end = min(values.count, index + width)
            parts.append("[" + values[index ..< end].map(item).joined(separator: ",") + "]")
            index = end
        }
        return "[" + parts.joined(separator: ",") + "]"
    }

    /// A JSON string literal (quotes and backslashes escaped; control
    /// characters dropped). Only engine-chosen names pass through here.
    public static func string(_ value: String) -> String {
        var out = "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            default:
                if scalar.value >= 0x20 { out.unicodeScalars.append(scalar) }
            }
        }
        return out + "\""
    }
}

/// The drafter's view of one round's block, host side: for each of `rows`
/// block rows (row `i` proposes the token `i + 1` places after the round's
/// seed), its top `width` vocabulary entries by the head product the
/// proposals rank, best first, ties to the lower id (the argmax rule).
public struct CBv2MTPProposalTrace: Sendable {
    public var rows: Int
    public var width: Int
    /// `[rows * width]`, row-major.
    public var ids: [Int32]
    /// The head product the proposals were ranked by (unscaled for argmax
    /// rounds), as float.
    public var raw: [Float]
    /// The same entries under the target's readout scale (`weight_scale_2`,
    /// output multiplier, softcap): `exp(score - logSumExp)` is the drafter's
    /// probability of the entry.
    public var scores: [Float]
    /// `[rows]`: log-sum-exp of each row's readout scores over the vocabulary.
    public var logSumExp: [Float]
    /// `[rows]`: the round's proposals as drafted (the memo), when known.
    public var proposals: [Int32]
    /// Where the block was drafted: `start` (index 0 of the round),
    /// `batched` (`startRounds`), `spec` (post-verify speculation).
    public var source: String

    public init(
        rows: Int, width: Int, ids: [Int32], raw: [Float], scores: [Float],
        logSumExp: [Float], proposals: [Int32], source: String
    ) {
        self.rows = rows
        self.width = width
        self.ids = ids
        self.raw = raw
        self.scores = scores
        self.logSumExp = logSumExp
        self.proposals = proposals
        self.source = source
    }

    /// JSON members (no braces) for a `round` line.
    public var jsonMembers: String {
        let parts: [String] = [
            "\"src\":" + CBv2TreeTrace.string(source),
            "\"prop\":" + CBv2TreeTrace.list(proposals),
            "\"top_ids\":" + CBv2TreeTrace.rows(ids, width: width) { String($0) },
            "\"top_raw\":" + CBv2TreeTrace.rows(raw, width: width) { CBv2TreeTrace.number($0) },
            "\"top_sc\":" + CBv2TreeTrace.rows(scores, width: width) { CBv2TreeTrace.number($0) },
            "\"lse\":" + CBv2TreeTrace.list(logSumExp),
        ]
        return parts.joined(separator: ",")
    }
}

/// A request-stateful drafter that can hand over the current round's block
/// trace (`CBV2_TREE_TRACE` only). Reading it may wait for the device.
public protocol CBv2MTPProposalTracing: AnyObject {
    func takeProposalTrace(requestState: any CBv2MTPRequestState) -> CBv2MTPProposalTrace?
}
