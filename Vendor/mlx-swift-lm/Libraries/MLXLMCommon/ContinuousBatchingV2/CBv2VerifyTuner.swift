// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN

/// Load-time choice among the exact verify-projection schedules that already
/// exist in `CBv2SmallProjection`, `CBv2VerifyScales` and `CBv2VerifyNVFP4`.
///
/// Every candidate computes the stock row-exact product bit for bit (same
/// lane chains, same joins, same rounding points; only the thread-to-output
/// mapping, tile width and load schedule differ). The tuner does not rely on
/// that claim: it computes the stock product on the same activations for two
/// layers of each shape and drops any candidate whose output differs in a
/// single bit (NaN payloads included). Winners are kept per (layer, rows)
/// for the verify row counts 2 to 8; everything else keeps the routing the
/// tree had before.
///
/// This runs once, inside the engine-build warm-up (`warmVerifyProjections`),
/// before `free_decode_begin`; it reads and writes no weight, cache or
/// sampler state, and the only thing it leaves behind is the table below.
/// `CBV2_VERIFY_TUNE=0` turns it off; `CBV2_VERIFY_TUNE_BUDGET` (seconds,
/// default 18) bounds it; `CBV2_VERIFY_TUNE_REPS` (default 24) sets the
/// timed repetitions per candidate.
public enum CBv2VerifyTuner {
    public static let enabled = ProcessInfo.processInfo.environment["CBV2_VERIFY_TUNE"] != "0"

    public enum Site: String, Sendable { case qkvg, o, gateUp = "gate_up", down, lmHead = "lm_head" }

    /// One schedule. `kind` selects the kernel family, `a`, `b`, `c` its knobs.
    public struct Variant: Hashable, Sendable, CustomStringConvertible {
        public enum Kind: String, Sendable {
            /// CBv2VerifyScales NVFP4: a = rows per tile (8, 16, 32), b = prefetch.
            case scales
            /// CBv2VerifyNVFP4: a = fragments, b = prefetch.
            case vn
            /// CBv2SmallProjection NVFP4 (one dispatch per group): a = TN, b = prefetch.
            case sp4
            /// CBv2SmallProjection FP8: a = rows/thread split, b = rows/thread flat, c = prefetch.
            case fp8sp
            /// CBv2VerifyScales FP8: a = row groups.
            case fp8vs
        }
        public var kind: Kind
        public var a: Int
        public var b: Int
        public var c: Int
        /// FP8 `fp8sp`: 1 = activation rows outermost (same chains).
        public var d: Int
        public init(_ kind: Kind, _ a: Int, _ b: Int = 0, _ c: Int = 0, _ d: Int = 0) {
            self.kind = kind
            self.a = a
            self.b = b
            self.c = c
            self.d = d
        }
        public var description: String {
            switch kind {
            case .scales: return "scales(nb\(a),pf\(b))"
            case .vn: return "vn(tn\(a),pf\(b))"
            case .sp4: return "sp4(tn\(a),pf\(b))"
            case .fp8sp: return "fp8sp(tm\(a)/\(b),pf\(c)\(d != 0 ? ",ro" : ""))"
            case .fp8vs: return "fp8vs(bm\(a))"
            }
        }
    }

    // MARK: Table

    private static let lock = NSLock()
    nonisolated(unsafe) private static var table = [ObjectIdentifier: [Variant?]]()
    nonisolated(unsafe) private static var populated = false

    /// The tuned schedule of the group led by `layer` at `rows` (2 to 8), or nil.
    @inline(__always)
    public static func variant(_ layer: AnyObject, rows: Int) -> Variant? {
        guard populated, rows >= 2, rows <= 8 else { return nil }
        return table[ObjectIdentifier(layer)]?[rows]
    }

    static func store(_ layers: [AnyObject], rows: Int, _ v: Variant) {
        lock.lock()
        defer { lock.unlock() }
        for layer in layers {
            var row = table[ObjectIdentifier(layer)] ?? Array(repeating: nil, count: 9)
            row[rows] = v
            table[ObjectIdentifier(layer)] = row
        }
    }

    static func publish() {
        lock.lock()
        populated = !table.isEmpty
        lock.unlock()
    }

    // MARK: Applying a variant

    /// The group's products through `v`, or nil when `v` does not apply to
    /// these layers and rows (the caller then keeps its default routing).
    public static func apply(_ v: Variant, _ layers: [any UnaryLayer], _ x: MLXArray)
        -> [MLXArray]?
    {
        switch v.kind {
        case .scales:
            var out = [MLXArray]()
            for layer in layers {
                guard layer is ModelOptWeightOnlyNVFP4Linear,
                    let y = CBv2VerifyScales.product(
                        layer, x, tileRows: v.a, prefetch: v.b != 0)
                else { return nil }
                out.append(y)
            }
            return out
        case .vn:
            let geometry = CBv2VerifyNVFP4.Geometry(fragments: v.a, prefetch: v.b)
            if layers.count == 2, let pair = CBv2VerifyNVFP4.products(layers, x, geometry: geometry) {
                return pair
            }
            var out = [MLXArray]()
            for layer in layers {
                guard let nv = layer as? ModelOptWeightOnlyNVFP4Linear,
                    let y = CBv2VerifyNVFP4.product(nv, x, geometry: geometry)
                else { return nil }
                out.append(y)
            }
            return out
        case .sp4:
            guard let nv = layers as? [ModelOptWeightOnlyNVFP4Linear] else { return nil }
            return CBv2SmallProjection.nvfp4Products(nv, x, tile: (v.a, v.b != 0))
        case .fp8sp:
            guard let fp8 = layers as? [ModelOptCalibratedLinear] else { return nil }
            return CBv2SmallProjection.fp8Products(
                fp8, x, tuning: .init(tmSplit: v.a, tmFlat: v.b, prefetch: v.c, rowsOuter: v.d != 0))
        case .fp8vs:
            var out = [MLXArray]()
            for layer in layers {
                guard layer is ModelOptCalibratedLinear,
                    let y = CBv2VerifyScales.product(
                        layer, x, rowGroupsOverride: v.a, anyShape: true)
                else { return nil }
                out.append(y)
            }
            return out
        }
    }

    // MARK: Candidates

    /// Largest power of two not above `32 / rows` (the FP8 kernel keeps at
    /// most 32 accumulators per lane).
    private static func tmCap(_ rows: Int) -> Int {
        var t = 1
        while t * 2 * rows <= 32 { t *= 2 }
        return t
    }

    static func candidates(fp8: Bool, rows: Int) -> [Variant] {
        var list = [Variant]()
        if fp8 {
            let cap = tmCap(rows)
            let shapes: [(Int, Int, Int)] = [
                (2, 2, 0), (1, 1, 0), (1, 2, 0), (2, 1, 0), (4, 2, 0), (2, 4, 0), (4, 4, 0),
                (8, 8, 0), (2, 2, 2), (2, 2, 4), (2, 2, 8), (4, 4, 2), (8, 8, 2), (1, 1, 4),
            ]
            var seen = Set<Variant>()
            for (s, f, p) in shapes {
                let v = Variant(.fp8sp, min(s, cap), min(f, cap), p)
                if seen.insert(v).inserted { list.append(v) }
            }
            // Wider exact trees (5 to 8 rows): the same schedules with the
            // activation rows outermost in the inner loop (the same chains,
            // fewer values live; every candidate is checked against the stock
            // bytes). Measured -5 to -14% on an M4 at 5 to 8 rows.
            if rows >= 5 {
                let ro: [(Int, Int, Int)] = [
                    (2, 2, 1), (4, 4, 1), (2, 2, 2), (4, 4, 2), (2, 2, 4), (4, 2, 1), (2, 4, 1),
                ]
                for (s, f, p) in ro {
                    let v = Variant(.fp8sp, min(s, cap), min(f, cap), p, 1)
                    if seen.insert(v).inserted { list.append(v) }
                }
            }
            list.append(Variant(.fp8vs, 1))
            list.append(Variant(.fp8vs, 4))
        } else {
            list += [
                Variant(.scales, 8, 1), Variant(.scales, 16, 0), Variant(.scales, 16, 1),
                Variant(.scales, 32, 0),
                Variant(.vn, 1, 0), Variant(.vn, 1, 1), Variant(.vn, 2, 0), Variant(.vn, 2, 1),
                Variant(.sp4, 1, 0), Variant(.sp4, 1, 1), Variant(.sp4, 2, 1),
            ]
        }
        return list
    }

    // MARK: Search

    /// One shape class: every layer group that shares a site and shape, with
    /// the probe width and whether its layers are FP8.
    public struct Group {
        public var site: Site
        public var key: String
        public var fp8: Bool
        public var width: Int
        public var groups: [[any UnaryLayer]]
        public init(site: Site, key: String, fp8: Bool, width: Int, groups: [[any UnaryLayer]]) {
            self.site = site
            self.key = key
            self.fp8 = fp8
            self.width = width
            self.groups = groups
        }
    }

    /// Whether every layer of the class has the operand layout the kernels
    /// read. A class that does not is left on the default routing.
    private static func valid(_ cls: Group) -> Bool {
        guard !cls.groups.isEmpty, cls.width > 64, cls.width % 4 == 0 else { return false }
        for group in cls.groups {
            guard !group.isEmpty else { return false }
            for layer in group {
                if cls.fp8 {
                    guard let l = layer as? ModelOptCalibratedLinear,
                        l.inputDimensions == cls.width, l.outputDimensions >= 8,
                        l.weight.dtype == .uint8,
                        l.weight.shape == [l.outputDimensions, cls.width],
                        l.graphConstants != nil
                    else { return false }
                } else {
                    guard let l = layer as? ModelOptWeightOnlyNVFP4Linear,
                        l.inputDimensions == cls.width, cls.width % 128 == 0,
                        l.outputDimensions >= 8, l.weight.dtype == .uint32,
                        l.weight.shape == [l.outputDimensions, cls.width / 8],
                        l.weightScale.dtype == .uint8,
                        l.weightScale.shape == [l.outputDimensions, cls.width / 16],
                        l.weightScale2.dtype == .float32, l.weightScale2.size == 1
                    else { return false }
                }
            }
        }
        return true
    }

    private static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

    private static func median(_ v: [Double]) -> Double {
        let s = v.sorted()
        return s[s.count / 2]
    }

    private static func bitsEqual(_ a: [MLXArray], _ b: [MLXArray]) -> Bool {
        guard a.count == b.count else { return false }
        for (p, q) in zip(a, b) {
            guard p.shape == q.shape, p.dtype == q.dtype else { return false }
            let bits: DType = p.dtype.size == 4 ? .uint32 : .uint16
            if (p.view(dtype: bits) .!= q.view(dtype: bits)).asType(.int32).sum().item(Int.self) != 0 {
                return false
            }
        }
        return true
    }

    /// Times and checks every candidate for every shape class and row count.
    /// `stock` computes the stock row-exact products; `baseline` the routing
    /// the tree uses without the table. Both take a layer group and `[1, rows,
    /// width]` activations.
    public static func tune(
        _ classes: [Group], dtype: DType,
        stock: ([any UnaryLayer], MLXArray) -> [MLXArray],
        baseline: (Site, [any UnaryLayer], MLXArray) -> [MLXArray]
    ) {
        guard enabled, !classes.isEmpty, dtype == .bfloat16,
            Device.defaultDevice().deviceType == .gpu
        else { return }
        let env = ProcessInfo.processInfo.environment
        let budget = Double(env["CBV2_VERIFY_TUNE_BUDGET"] ?? "") ?? 18
        let reps = max(8, Int(env["CBV2_VERIFY_TUNE_REPS"] ?? "") ?? 24)
        let start = now()
        let deadline = start + budget
        var log = [String]()
        var rejected = 0, tried = 0, chosen = 0
        // The caller lists the largest byte classes first, so a spent budget costs the least.
        let order = classes.filter(valid)
        guard !order.isEmpty else { return }
        for (ci, cls) in order.enumerated() {
            let probes = pickProbes(cls.groups.count)
            // Share what is left of the budget equally among the classes left.
            let classDeadline = now() + max(0, deadline - now()) / Double(order.count - ci)
            for rows in 2 ... 8 {
                if now() > classDeadline { break }
                let key = MLXRandom.key(UInt64(1000 * rows + cls.width % 997))
                let xs = (0 ..< 2).map { i in
                    (MLXRandom.normal([1, rows, cls.width], dtype: .float32, key: key)
                        * (i == 0 ? 0.5 : 1.5 + Float(i))).asType(dtype)
                }
                eval(xs)
                // Exactness references on two layers with two activations.
                var refs = [(Int, Int, [MLXArray])]()
                for (pi, p) in probes.prefix(2).enumerated() {
                    let ref = stock(cls.groups[p], xs[pi])
                    eval(ref)
                    refs.append((p, pi, ref))
                }
                func run(_ group: [any UnaryLayer], _ x: MLXArray, _ v: Variant?) -> [MLXArray]? {
                    v.map { apply($0, group, x) } ?? baseline(cls.site, group, x)
                }
                func time(_ v: Variant?, _ n: Int) -> Double? {
                    // First call compiles and warms the weights' pages.
                    guard let first = run(cls.groups[probes[0]], xs[0], v) else { return nil }
                    eval(first)
                    var t = [Double]()
                    for i in 0 ..< n {
                        let g = cls.groups[probes[i % probes.count]]
                        guard let out = run(g, xs[0], v) else { return nil }
                        let t0 = now()
                        eval(out)
                        t.append(now() - t0)
                    }
                    return median(t)
                }
                guard let base = time(nil, reps) else { continue }
                var best: (Variant, Double)?
                for v in candidates(fp8: cls.fp8, rows: rows) {
                    if now() > classDeadline { break }
                    tried += 1
                    var ok = true
                    for (p, pi, ref) in refs {
                        guard let out = apply(v, cls.groups[p], xs[pi]) else { ok = false; break }
                        eval(out)
                        if !bitsEqual(out, ref) { ok = false; rejected += 1; break }
                    }
                    guard ok, let t = time(v, reps) else { continue }
                    if best == nil || t < best!.1 { best = (v, t) }
                }
                // Confirm against the baseline with fresh, interleaved reps.
                if let (v, _) = best, let b2 = time(nil, reps), let t2 = time(v, reps),
                    t2 < min(base, b2) * 0.985
                {
                    for g in cls.groups { store(g.map { $0 as AnyObject }, rows: rows, v) }
                    chosen += 1
                    log.append(
                        String(
                            format: "VTUNE %@ %@ rows=%d %@ %.1fus (default %.1fus)",
                            cls.site.rawValue, cls.key, rows, v.description, t2 * 1e6,
                            min(base, b2) * 1e6))
                } else {
                    log.append(
                        String(
                            format: "VTUNE %@ %@ rows=%d default %.1fus", cls.site.rawValue,
                            cls.key, rows, base * 1e6))
                }
            }
        }
        publish()
        log.append(
            String(
                format: "VTUNE done: %d classes, %d candidates tried, %d rejected (not bit-identical), "
                    + "%d overrides, %.1f s",
                order.count, tried, rejected, chosen, now() - start))
        FileHandle.standardError.write(Data((log.joined(separator: "\n") + "\n").utf8))
    }

    /// Up to six layer groups spread over the model, so the timed weights
    /// rotate through more bytes than the system-level cache holds.
    private static func pickProbes(_ count: Int) -> [Int] {
        let n = min(count, 6)
        guard n > 1 else { return [0] }
        return (0 ..< n).map { $0 * (count - 1) / (n - 1) }
    }
}
