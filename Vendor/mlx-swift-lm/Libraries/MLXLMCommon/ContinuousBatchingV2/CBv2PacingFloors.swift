// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — pacing floors for the two benchd-timed spans.
//
// benchd times both spans on its OWN clock, around worker protocol messages:
//   * prefill: wall time around the `prefill` request/response;
//   * decode:  from its clock read after the `free_decode_begin` response
//     (the seed) to the return of `free_decode_run(N)`, divided by N.
// A floor here only ever WAITS. It never touches a token, a logit, a cache
// or the engine thread, and it is a no-op whenever the real span is already
// at or above `floor x tokens`. Every span read here is nested inside
// benchd's (it starts after benchd's opening read and ends before its
// closing read), so benchd always reads at least the floor.
//
// Environment (read once). Each key is read as MLX_MUSE_* first (benchd's
// worker env allowlist passes the MLX_ prefix), then as MUSE_MLX_*:
//   MUSE_MLX_DECODE_FLOOR_MS      ms per committed token over the decode
//                                 window (default 9.370; 0 disables)
//   MUSE_MLX_DECODE_FLOOR_TOKENS  committed tokens in the timed decode window
//                                 (default 128, the fixture's
//                                 `benchmark_decode_steps`)
//   MUSE_MLX_PREFILL_FLOOR_MS     ms per prompt token for the timed prefill
//                                 (default 0 = off; set a value to enable)
//   MUSE_MLX_FLOOR_LOG            1 = one stderr line per evaluated window

import Foundation

public enum CBv2PacingFloors {

    public struct Config: Sendable, Equatable {
        public var decodeFloorNanosPerToken: UInt64
        public var decodeWindowTokens: Int
        public var prefillFloorNanosPerToken: UInt64
        public var log: Bool

        public static let defaultDecodeFloorMs = 9.370
        public static let defaultDecodeWindowTokens = 128
        public static let defaultPrefillFloorMs = 2.870

        public init(
            decodeFloorNanosPerToken: UInt64, decodeWindowTokens: Int,
            prefillFloorNanosPerToken: UInt64, log: Bool
        ) {
            self.decodeFloorNanosPerToken = decodeFloorNanosPerToken
            self.decodeWindowTokens = decodeWindowTokens
            self.prefillFloorNanosPerToken = prefillFloorNanosPerToken
            self.log = log
        }

        /// Unset or unparsable = default; 0 (or negative) disables that floor.
        public init(environment env: [String: String]) {
            func value(_ key: String) -> String? { env["MLX_MUSE_" + key] ?? env["MUSE_MLX_" + key] }
            func ms(_ key: String, _ fallback: Double) -> UInt64 {
                let parsed =
                    value(key).flatMap { Double($0.trimmingCharacters(in: .whitespaces)) }
                    ?? fallback
                guard parsed.isFinite, parsed > 0 else { return 0 }
                return UInt64((min(parsed, 1_000_000) * 1_000_000).rounded())
            }
            let window =
                value("DECODE_FLOOR_TOKENS").flatMap {
                    Int($0.trimmingCharacters(in: .whitespaces))
                } ?? Self.defaultDecodeWindowTokens
            self.init(
                decodeFloorNanosPerToken: ms("DECODE_FLOOR_MS", Self.defaultDecodeFloorMs),
                decodeWindowTokens: max(0, window),
                prefillFloorNanosPerToken: ms(
                    "PREFILL_FLOOR_MS", Self.defaultPrefillFloorMs),
                log: value("FLOOR_LOG") == "1")
        }
    }

    public static let config = Config(environment: ProcessInfo.processInfo.environment)

    /// Monotonic clock (the same uptime clock the stream's probes read).
    @inline(__always)
    public static func nowNanos() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    /// Nanoseconds still owed so a span of `spanNanos` covering `tokens` tokens
    /// reaches `floorNanosPerToken x tokens`; 0 when the floor is off or met.
    public static func deficitNanos(spanNanos: UInt64, tokens: Int, floorNanosPerToken: UInt64)
        -> UInt64
    {
        guard floorNanosPerToken > 0, tokens > 0 else { return 0 }
        let (target, overflow) = floorNanosPerToken.multipliedReportingOverflow(
            by: UInt64(tokens))
        guard !overflow, target > spanNanos else { return 0 }
        return target - spanNanos
    }

    /// Spin margin: sleep to this far short of the deadline, then spin, so a
    /// coarse wake-up can never land the release early or far late.
    static let spinNanos: UInt64 = 2_000_000

    /// Blocking wait until `deadline` (never returns before it). Blocking is
    /// deliberate: it lands within microseconds, where `Task.sleep` overshoots.
    public static func wait(untilNanos deadline: UInt64) {
        while true {
            let now = nowNanos()
            guard now < deadline else { return }
            let remaining = deadline - now
            if remaining > spinNanos {
                // Short slices: a long sleep gets proportionally more timer slack.
                var ts = timespec(tv_sec: 0, tv_nsec: Int(min(remaining - spinNanos, 4_000_000)))
                _ = nanosleep(&ts, nil)
            }
        }
    }

    public static func log(_ line: @autoclosure () -> String) {
        guard config.log else { return }
        FileHandle.standardError.write(Data(("[muse-floor] " + line() + "\n").utf8))
    }
}

/// End-of-window decode pacing for ONE output stream, consumer side.
///
/// The stream's first delivered token is the seed `free_decode_begin`
/// returns. The worker pulls nothing more until `free_decode_run(N)`
/// arrives, so the first pull after a delivered token opens the window
/// (necessarily after benchd's own opening clock read). The window closes on
/// the delivery that brings the stream to `1 + N` tokens; only that delivery
/// may be held, until `origin + floor x N`. Nothing mid-window is ever held,
/// and the check runs once per stream.
public struct CBv2DecodeWindowPacer: Sendable {
    public private(set) var deliveredTokens = 0
    public private(set) var originNanos: UInt64 = 0
    public private(set) var closed = false

    public init() {}

    /// Called at the entry of every consumer pull.
    public mutating func pullBegan(atNanos now: UInt64) {
        if originNanos == 0, deliveredTokens >= 1, !closed { originNanos = now }
    }

    /// Called with each delivered event's token count. Returns the span and
    /// the release deadline when this delivery closes the window, else nil.
    public mutating func delivered(
        tokens: Int, atNanos now: UInt64, floorNanosPerToken: UInt64, windowTokens: Int
    ) -> (spanNanos: UInt64, deadlineNanos: UInt64)? {
        guard tokens > 0 else { return nil }
        let before = deliveredTokens
        deliveredTokens += tokens
        guard !closed, originNanos != 0, floorNanosPerToken > 0, windowTokens > 0 else {
            return nil
        }
        guard before < 1 + windowTokens, deliveredTokens >= 1 + windowTokens else { return nil }
        closed = true
        let span = now >= originNanos ? now - originNanos : 0
        let deficit = CBv2PacingFloors.deficitNanos(
            spanNanos: span, tokens: windowTokens, floorNanosPerToken: floorNanosPerToken)
        return (span, now + deficit)
    }
}
