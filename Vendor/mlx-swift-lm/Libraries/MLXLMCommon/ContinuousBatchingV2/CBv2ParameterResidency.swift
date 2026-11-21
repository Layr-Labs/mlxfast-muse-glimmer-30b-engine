// Copyright © 2026 Eigen Labs.
//
// Process-lifetime GPU residency of a model's or drafter's parameter buffers.

import Foundation
import MLX
import Metal

/// A steppable model or drafter whose parameter buffers stay resident for the
/// life of the process. `residentParameterKey` names the storage, not the
/// wrapper, so adapters rebuilt per engine share one registration.
public protocol CBv2ResidentParametersProviding: AnyObject {
    var residentParameterKey: ObjectIdentifier { get }
    /// Evaluated parameter arrays. Unevaluated arrays are skipped.
    func residentParameterArrays() -> [MLXArray]
}

@_silgen_name("cbv2_array_backing")
private func cbv2ArrayBacking(_ opaque: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer?

/// The parameter buffers of each provider are added once to committed Metal
/// residency sets, locked in memory (`mlock`), and requested resident; the
/// requests stay standing between requests and engines. Every later engine
/// build requests the same sets again on a background queue, before its first
/// step, so residency the OS dropped while the process idled is restored off
/// the step path. Memory placement only: no array, kernel or value changes,
/// and MLX's own allocator, its wired limit and its residency sets are left
/// alone. `CBV2_PARAMETER_RESIDENCY=0` turns it off.
public enum CBv2ParameterResidency {
    static let enabled =
        ProcessInfo.processInfo.environment["CBV2_PARAMETER_RESIDENCY"] != "0"

    private final class Registration: @unchecked Sendable {
        let arrays: [MLXArray]
        /// MTLResidencySet values (macOS 15+), held untyped so the type does not
        /// need an availability annotation; used only behind `#available`.
        let sets: [AnyObject]
        init(arrays: [MLXArray], sets: [AnyObject]) {
            self.arrays = arrays
            self.sets = sets
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var registrations: [ObjectIdentifier: Registration] = [:]
    private static let queue = DispatchQueue(
        label: "cbv2.parameter.residency", qos: .userInitiated)

    /// Registers each provider among `owners` once, then (re)requests every
    /// registered set of those owners.
    public static func prepare(_ owners: [AnyObject]) {
        guard enabled else { return }
        guard #available(macOS 15.0, iOS 18.0, *) else { return }
        for owner in owners {
            guard let provider = owner as? any CBv2ResidentParametersProviding else { continue }
            let key = provider.residentParameterKey
            let existing = lock.withLock { registrations[key] }
            if let existing {
                let sets = existing.sets
                queue.async {
                    for set in sets { (set as! any MTLResidencySet).requestResidency() }
                }
                continue
            }
            guard let registration = register(provider.residentParameterArrays()) else {
                continue
            }
            lock.withLock { registrations[key] = registration }
        }
    }

    @available(macOS 15.0, iOS 18.0, *)
    private static func register(_ arrays: [MLXArray]) -> Registration? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        // Bounded sets, as MLX bounds its own (a 5% share of the working set).
        let cap = max(Int(device.recommendedMaxWorkingSetSize / 20), 64 << 20)
        var seen = Set<ObjectIdentifier>()
        var groups: [[any MTLBuffer]] = [[]]
        var groupBytes = 0
        var kept: [MLXArray] = []
        for array in arrays {
            // Small arrays sub-allocated from MLX's heap are left out.
            guard let opaque = array.ctx.ctx, let pointer = cbv2ArrayBacking(opaque),
                let buffer = Unmanaged<AnyObject>.fromOpaque(pointer).takeUnretainedValue()
                    as? any MTLBuffer,
                buffer.length > 0, buffer.heap == nil,
                seen.insert(ObjectIdentifier(buffer)).inserted
            else { continue }
            kept.append(array)
            if groupBytes + buffer.length > cap, !groups[groups.count - 1].isEmpty {
                groups.append([])
                groupBytes = 0
            }
            groups[groups.count - 1].append(buffer)
            groupBytes += buffer.length
        }
        guard !kept.isEmpty else { return nil }
        var sets: [any MTLResidencySet] = []
        for group in groups where !group.isEmpty {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.initialCapacity = group.count
            guard let set = try? device.makeResidencySet(descriptor: descriptor) else { break }
            for buffer in group {
                set.addAllocation(buffer)
                if buffer.storageMode == .shared {
                    _ = mlock(buffer.contents(), buffer.length)
                }
            }
            set.commit()
            set.requestResidency()
            sets.append(set)
        }
        guard !sets.isEmpty else { return nil }
        return Registration(arrays: kept, sets: sets.map { $0 as AnyObject })
    }
}
