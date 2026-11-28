
import Foundation

public enum CBv2PacingFloors {

    public struct Config: Sendable, Equatable {
        public var decodeFloorNanosPerToken: UInt64
        public var decodeWindowTokens: Int
        public var prefillFloorNanosPerToken: UInt64
        public var log: Bool

        public static let defaultDecodeFloorMs = 9.370
        public static let defaultDecodeWindowTokens = 128
        public static let defaultPrefillFloorMs = 0.0

        public init(
            decodeFloorNanosPerToken: UInt64, decodeWindowTokens: Int,
            prefillFloorNanosPerToken: UInt64, log: Bool
        ) {
            self.decodeFloorNanosPerToken = decodeFloorNanosPerToken
            self.decodeWindowTokens = decodeWindowTokens
            self.prefillFloorNanosPerToken = prefillFloorNanosPerToken
            self.log = log
        }

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

    @inline(__always)
    public static func nowNanos() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    public static func deficitNanos(spanNanos: UInt64, tokens: Int, floorNanosPerToken: UInt64)
        -> UInt64
    {
        guard floorNanosPerToken > 0, tokens > 0 else { return 0 }
        let (target, overflow) = floorNanosPerToken.multipliedReportingOverflow(
            by: UInt64(tokens))
        guard !overflow, target > spanNanos else { return 0 }
        return target - spanNanos
    }

    static let spinNanos: UInt64 = 2_000_000

    public static func wait(untilNanos deadline: UInt64) {
        while true {
            let now = nowNanos()
            guard now < deadline else { return }
            let remaining = deadline - now
            if remaining > spinNanos {
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

public struct CBv2DecodeWindowPacer: Sendable {
    public private(set) var deliveredTokens = 0
    public private(set) var originNanos: UInt64 = 0
    public private(set) var closed = false

    public init() {}

    public mutating func pullBegan(atNanos now: UInt64) {
        if originNanos == 0, deliveredTokens >= 1, !closed { originNanos = now }
    }

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
