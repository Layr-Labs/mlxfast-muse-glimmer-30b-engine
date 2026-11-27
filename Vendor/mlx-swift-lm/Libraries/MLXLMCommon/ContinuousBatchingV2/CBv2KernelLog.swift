// CBv2KernelLog.swift
//
// `CBV2_KERNEL_LOG=1`: stderr trace of Metal pipeline keys. MLX compiles a
// custom kernel's pipeline at its first dispatch of each distinct key: the
// kernel name, the template arguments, each input's dtype and how it is
// passed (a 0-d array scalar, fewer than 8 elements constant, else device),
// and the output dtypes. `observe` rebuilds that key at the dispatch sites of
// the verify path; once `markWarmComplete` ran, a key not seen before is a
// pipeline compiled while a round is served, and is printed.
// Disabled: one cached boolean test per call.

import Foundation
import MLX

public enum CBv2KernelLog {
    public static let enabled = ProcessInfo.processInfo.environment["CBV2_KERNEL_LOG"] == "1"
    nonisolated(unsafe) private static var seen = Set<String>()
    nonisolated(unsafe) private static var warmComplete = false
    nonisolated(unsafe) private static var postWarm = 0
    private static let lock = NSLock()

    /// Later first-seen keys are reported as compiled after warm-up. Safe to
    /// call more than once.
    public static func markWarmComplete() {
        guard enabled else { return }
        lock.lock()
        warmComplete = true
        let count = seen.count
        lock.unlock()
        write("[kernel-log] warm-up complete: \(count) pipeline keys built\n")
    }

    /// `name` plus `template` is the kernel identity; `inputs` and `outputs`
    /// supply the remaining key components.
    public static func observe(
        _ name: String, template: String = "", inputs: [MLXArray], outputs: [DType] = []
    ) {
        guard enabled else { return }
        var key = name + "<" + template + ">("
        for array in inputs {
            let passing = array.ndim == 0 ? "s" : (array.size < 8 ? "c" : "d")
            key += "\(array.dtype)\(passing),"
        }
        key += ")->" + outputs.map { "\($0)" }.joined(separator: ",")
        lock.lock()
        let fresh = seen.insert(key).inserted
        let late = fresh && warmComplete
        if late { postWarm += 1 }
        let ordinal = postWarm
        lock.unlock()
        if late {
            write("[kernel-log] COMPILED AFTER WARM-UP #\(ordinal): \(key)\n")
        }
    }

    private static func write(_ text: String) {
        FileHandle.standardError.write(Data(text.utf8))
    }
}
