import Foundation
import MLX
import MLXFast

/// One-pass paged decode attention for short contexts (speculative verify
/// windows). The two-pass path costs two dependent dispatches per layer (the
/// partition pass, then the merge) and its partition pass walks a partition's
/// tokens serially for a whole head group. Here one threadgroup owns one
/// (row, head group) and holds every (partition, strided state) of it as a
/// simdgroup; the partition and global merges run through threadgroup
/// memory. Per head the float operations and their order are the two-pass
/// kernels' (`paged_attention_onepass_impl` in pagedattention.metal), so the
/// output is bit-identical; only the schedule changes.
///
/// Kill switch: `MUSE_PAGED_ONEPASS=0` (two-pass path).
public enum PagedOnePassAttention {
    private static let environment = ProcessInfo.processInfo.environment
    static let enabled = environment["MUSE_PAGED_ONEPASS"] != "0"
    /// Each state issues the K/V loads of 4 tokens before using them (same bits).
    /// `MUSE_PAGED_ONEPASS_BLOCK=0` restores the per-token loop.
    static let block = environment["MUSE_PAGED_ONEPASS_BLOCK"] == "0" ? 1 : 4
    /// The direct-slab decode entry (microbenchmarks, probes) takes the
    /// one-pass kernel only on request.
    static let slabEnabled = environment["MUSE_PAGED_ONEPASS_SLAB"] == "1"
    /// Partitions one threadgroup can hold (contexts up to `4 * PTOK`).
    static let maxPartitions: Int = {
        if let raw = environment["MUSE_PAGED_ONEPASS_MAXPART"], let value = Int(raw),
            (1 ... 8).contains(value)
        {
            return value
        }
        return 4
    }()
    /// Match compiled MAXPART to `plan.maxPartitions` (`1...maxPartitions`) when `CBV2_ONEPASS_DYN_PART != "0"`.
    static let dynamicMaxPartitions: Bool =
        environment["CBV2_ONEPASS_DYN_PART"] != "0"
    private static let requestedHeads: Int = {
        if let raw = environment["MUSE_PAGED_ONEPASS_HPT"], let value = Int(raw), value >= 1 {
            return value
        }
        // Two heads share each K/V row load: at 8 rows the one-head form is
        // throughput-bound (measured 43 vs 31 us per layer), at 1 row equal.
        return 2
    }()
    /// `MUSE_PAGED_ONEPASS_CHECK=1` or `CBV2_ONEPASS_CHECK=1`: every one-pass dispatch also runs the
    /// two-pass kernels and compares raw bits (slow; a self-test, not a mode).
    static let checks =
        environment["MUSE_PAGED_ONEPASS_CHECK"] == "1"
        || environment["CBV2_ONEPASS_CHECK"] == "1"
    private static let logs = environment["MUSE_PAGED_ONEPASS_LOG"] == "1"
    /// Set by the load-time probe in `warm` when a one-pass pipeline cannot
    /// hold the kernel's threadgroup directly. MLX would then re-resolve a
    /// constrained pipeline on every dispatch (host string work per layer);
    /// the two-pass kernels are used instead. Same bytes either way.
    nonisolated(unsafe) private(set) static var threadgroupProbeFailed = false
    /// Set by `warm` when the `DYN_TREE = true` one-pass variant fails the threadgroup probe;
    /// leaves `DYN_TREE = false` unaffected and falls back to two-pass only for dynamic-tree rounds.
    nonisolated(unsafe) private(set) static var dynTreeProbeFailed = false
    /// `MUSE_PAGED_ONEPASS_PROBE_FAIL=1`: the probe asks for a threadgroup no
    /// pipeline holds, to exercise the two-pass fallback (a self-test).
    private static let forceProbeFailure = environment["MUSE_PAGED_ONEPASS_PROBE_FAIL"] == "1"
    /// `CBV2_PAGED_FUSED_GATE=0`: disables the low-acceptance 0-ULP fused
    /// `output * sigmoid(gate)` (+ optional FP8 cast) epilogue inside one-pass
    /// paged attention (default enabled after load-time 0-ULP self-check).
    public static let fusedGateEnvEnabled: Bool = environment["CBV2_PAGED_FUSED_GATE"] != "0"
    /// `CBV2_PAGED_LOWTGEM=0`: disables the low-TGEM (12.5 KB at MAXPART=2, 25.0 KB at MAXPART=4,
    /// HPT=NSG=4) register-resident-Q one-pass fused gate kernel.
    public static let lowTgemEnvEnabled: Bool = environment["CBV2_PAGED_LOWTGEM"] != "0"
    nonisolated(unsafe) private static var fusedGateVerified = false
    nonisolated(unsafe) private static var lowTgemVerified = false
    nonisolated(unsafe) static var fusedGateSuspended = false

    public static var fusedGateActive: Bool {
        enabled && fusedGateEnvEnabled && !fusedGateSuspended
            && !threadgroupProbeFailed
            && lock.withLock { fusedGateVerified }
            && (!CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.xnorDraftHeadEligible)
    }

    static var lowTgemActive: Bool {
        fusedGateActive && lowTgemEnvEnabled && lock.withLock { lowTgemVerified }
    }

    /// Eligible `HPT` (`== simdgroups`, e.g. `4` for Muse Glimmer 30B `GQA=16, D=128`) for
    /// `paged_attention_onepass_lowtgem_impl` whose `red_smem` fits inside Metal's 32 KB cap.
    static func lowTgemHeads(headDim: Int, gqa: Int, simdgroups: Int, maxPart: Int) -> Int? {
        guard headDim == 128, simdgroups >= 2, gqa % simdgroups == 0 else { return nil }
        let floats = maxPart * simdgroups * (simdgroups - 1) * (headDim + 2)
        guard floats * 4 <= PagedAttentionKernel.threadgroupMemoryLimit else { return nil }
        return simdgroups
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var checked = 0
    nonisolated(unsafe) private static var differing = 0

    /// Self-test bookkeeping: evaluates both results and counts elements
    /// whose bits differ.
    static func compare(onePass: MLXArray, twoPass: MLXArray) {
        let a = onePass.asType(.float32).asArray(Float.self)
        let b = twoPass.asType(.float32).asArray(Float.self)
        var bad = a.count == b.count ? 0 : max(a.count, b.count)
        if a.count == b.count {
            for i in 0 ..< a.count where a[i].bitPattern != b[i].bitPattern { bad += 1 }
        }
        lock.withLock {
            checked += 1
            differing += bad
            if bad != 0 || checked % 500 == 0 {
                FileHandle.standardError.write(
                    Data(
                        "[onepass-check] dispatches=\(checked) rows=\(onePass.dim(0)) elements=\(a.count) differing=\(bad) total_differing=\(differing)\n"
                            .utf8))
            }
        }
    }

    /// Compiles the segmented variants a verify window can reach (one per
    /// binding class) so no decode round pays a kernel compilation.
    public static func warm(
        dtype: DType, queryHeads: Int, kvHeads: Int, headDim: Int, pageSize: Int,
        softcap: Bool
    ) {
        guard enabled, kvHeads > 0, queryHeads % kvHeads == 0,
            PagedAttentionKernel.partitionTargetThreadgroups == 0,
            PagedAttentionKernel.supportedHeadDims.contains(headDim),
            PagedAttentionKernel.partitionTokens % pageSize == 0,
            let source = try? PagedAttentionResources.loadSourceForCurrentProcess()
        else { return }
        let gqa = queryHeads / kvHeads
        guard let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa),
            let hpt = eligible(headDim: headDim, gqa: gqa, simdgroups: nsg, partitions: 1, rows: 1)
        else { return }
        let ptok = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokens, pageSize: pageSize)
        let pageElements = kvHeads * pageSize * headDim
        let storage = MLXArray.zeros([2 * pageElements], dtype: dtype)
        let q = MLXArray.zeros([1, queryHeads, headDim], dtype: dtype)
        let seqinfo = MLXArray([Int32(0), 1, 1, 0, 0, 0, 0, 0], [1, 8])
        let params = MLXArray([Float(1), 1, 0, 0, 0, 0, 0, 0])
        let previous = MLXArray.zeros([1], type: Int32.self)
        var record = [Int32](repeating: 0, count: PagedSegmentDispatchPlan.recordStride)
        record[3] = 1
        record[6] = 1
        var outputs = [MLXArray]()
        defer {
            if !outputs.isEmpty { eval(outputs) }
        }
        if CBv2DraftRegime.verifyTreeMode != 0 {
            outputs += PagedSegmentTransfers.warmCopyKernels(
                dtype: dtype, kvHeads: kvHeads, headDim: headDim, pageSize: pageSize)
        }
        outputs += PagedSegmentAttention.warmTwoPass(
            dtype: dtype, queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim,
            pageSize: pageSize, softcap: softcap, source: source)
        for bindings in PagedSegmentDispatchPlan.bindingClasses {
            let offsets = [Int64](repeating: Int64(pageElements), count: max(8, bindings))
            // Probe first: one threadgroup on zero inputs, output discarded.
            let held =
                (try? withError { () -> Bool in
                    let probed = segments(
                        queries: q, backing: [MLXArray](repeating: storage, count: bindings),
                        seqinfo: seqinfo, params: params, previous: previous,
                        valueOffsets: MLXArray(offsets), records: MLXArray(record),
                        dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                        simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                        source: source, probe: true)
                    eval(probed.fence, probed.out)
                    return true
                }) ?? false
            if logs {
                FileHandle.standardError.write(
                    Data("[onepass] threadgroup probe bindings=\(bindings) held=\(held)\n".utf8))
            }
            guard held else {
                threadgroupProbeFailed = true
                return
            }
            let warmDynTree = CBv2DraftRegime.verifyTreeMode != 0 && CBv2DraftRegime.dynamicTreeEnabled
            if warmDynTree && !dynTreeProbeFailed {
                let dynHeld =
                    (try? withError { () -> Bool in
                        let probed = segments(
                            queries: q, backing: [MLXArray](repeating: storage, count: bindings),
                            seqinfo: seqinfo, params: params, previous: previous,
                            valueOffsets: MLXArray(offsets), records: MLXArray(record),
                            dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                            simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                            source: source, probe: true, treeMeta: dummyTreeMeta)
                        eval(probed.fence, probed.out)
                        return true
                    }) ?? false
                if !dynHeld {
                    dynTreeProbeFailed = true
                }
            }
            let partRange = dynamicMaxPartitions ? (1 ... maxPartitions) : (maxPartitions ... maxPartitions)
            for mp in partRange {
                let result = segments(
                    queries: q, backing: [MLXArray](repeating: storage, count: bindings),
                    seqinfo: seqinfo, params: params, previous: previous,
                    valueOffsets: MLXArray(offsets), records: MLXArray(record), dtype: dtype,
                    kvHeads: kvHeads, pageSize: pageSize, hpt: hpt, simdgroups: nsg,
                    partitionTokens: ptok, softcap: softcap, source: source, maxPart: mp,
                    forceMaxPart: true)
                outputs += [result.fence, result.out]
                if warmDynTree && !dynTreeProbeFailed {
                    let dynResult = segments(
                        queries: q, backing: [MLXArray](repeating: storage, count: bindings),
                        seqinfo: seqinfo, params: params, previous: result.fence,
                        valueOffsets: MLXArray(offsets), records: MLXArray(record), dtype: dtype,
                        kvHeads: kvHeads, pageSize: pageSize, hpt: hpt, simdgroups: nsg,
                        partitionTokens: ptok, softcap: softcap, source: source,
                        treeMeta: dummyTreeMeta, maxPart: mp, forceMaxPart: true)
                    outputs += [dynResult.fence, dynResult.out]
                }
            }
        }
        if dtype == .bfloat16 && fusedGateEnvEnabled && !threadgroupProbeFailed {
            verifyAndWarmFusedGate(
                queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim, pageSize: pageSize,
                hpt: hpt, nsg: nsg, ptok: ptok, softcap: softcap, source: source)
        }
    }

    /// Load-time 0-ULP parity check and warmup for the fused `sigmoid(gate)` (+ FP8 cast)
    /// epilogue across `CAST_FP8 = false` and `CAST_FP8 = true`. Fail-closed on any bit
    /// mismatch or Metal threadgroup/compile error.
    private static func verifyAndWarmFusedGate(
        queryHeads: Int, kvHeads: Int, headDim: Int, pageSize: Int,
        hpt: Int, nsg: Int, ptok: Int, softcap: Bool, source: String
    ) {
        let pageElements = kvHeads * pageSize * headDim
        var kvVals = [Float]()
        kvVals.reserveCapacity(2 * pageElements)
        for i in 0 ..< (2 * pageElements) {
            kvVals.append(Float(((i * 11 + 3) % 29) - 14) * 0.0625)
        }
        let storage = MLXArray(kvVals, [2 * pageElements]).asType(.bfloat16)
        let qElems = queryHeads * headDim
        var qVals = [Float](), gVals = [Float]()
        qVals.reserveCapacity(qElems); gVals.reserveCapacity(qElems)
        for i in 0 ..< qElems {
            qVals.append(Float(((i * 7 + 5) % 23) - 11) * 0.0625)
            gVals.append(Float(((i * 13 + 1) % 31) - 15) * 0.125)
        }
        let q = MLXArray(qVals, [1, queryHeads, headDim]).asType(.bfloat16)
        let gate = MLXArray(gVals, [1, queryHeads, headDim]).asType(.bfloat16)
        let seqinfo = MLXArray([Int32(0), Int32(min(pageSize, 4)), 1, 0, 0, 0, 0, 0], [1, 8])
        let params = MLXArray([Float(1), Float(1.0 / Double(headDim).squareRoot()), 0, 0, 0, 0, 0, 0])
        let previous = MLXArray.zeros([1], type: Int32.self)
        var record = [Int32](repeating: 0, count: PagedSegmentDispatchPlan.recordStride)
        record[3] = 1
        record[6] = 1
        let offsets = [Int64](repeating: Int64(pageElements), count: 8)
        let testMu = MLXArray([Float(1.7), Float(1.0 / 1.7)])
        let testBd = MLXArray([Float(448.0)])

        let passed: Bool =
            (try? withError { () -> Bool in
                let ref = segments(
                    queries: q, backing: [storage], seqinfo: seqinfo, params: params,
                    previous: previous, valueOffsets: MLXArray(offsets), records: MLXArray(record),
                    dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                    simdgroups: nsg, partitionTokens: ptok, softcap: softcap, source: source,
                    maxPart: 1, forceMaxPart: true)
                let refGated = ref.out * sigmoid(gate)
                let refCast = ModelOptActivationKernels.fp8(
                    refGated, multipliers: testMu, overflowBound: testBd)
                let gotGated = segmentsFusedGate(
                    queries: q, backing: [storage], seqinfo: seqinfo, params: params,
                    previous: ref.fence, valueOffsets: MLXArray(offsets), records: MLXArray(record),
                    gate: gate, fp8Multipliers: nil, fp8OverflowBound: nil,
                    dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                    simdgroups: nsg, partitionTokens: ptok, softcap: softcap, source: source,
                    maxPart: 1, forceMaxPart: true)
                let gotCast = segmentsFusedGate(
                    queries: q, backing: [storage], seqinfo: seqinfo, params: params,
                    previous: gotGated.fence, valueOffsets: MLXArray(offsets), records: MLXArray(record),
                    gate: gate, fp8Multipliers: testMu, fp8OverflowBound: testBd,
                    dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                    simdgroups: nsg, partitionTokens: ptok, softcap: softcap, source: source,
                    maxPart: 1, forceMaxPart: true)
                let diff0 = (contiguous(gotGated.out).view(dtype: .uint16)
                    .!= contiguous(refGated).view(dtype: .uint16)).asType(.int32).sum()
                let diff1 = (contiguous(gotCast.out).view(dtype: .uint16)
                    .!= contiguous(refCast).view(dtype: .uint16)).asType(.int32).sum()
                let total = diff0 + diff1
                eval(total)
                guard total.item(Int.self) == 0 else { return false }
                let warmDynTree = CBv2DraftRegime.verifyTreeMode != 0
                    && CBv2DraftRegime.dynamicTreeEnabled && !dynTreeProbeFailed
                let partRange = dynamicMaxPartitions ? (1 ... maxPartitions) : (maxPartitions ... maxPartitions)
                var warmOut = [MLXArray]()
                var fence = gotCast.fence
                for bindings in PagedSegmentDispatchPlan.bindingClasses {
                    let bOffsets = MLXArray([Int64](repeating: Int64(pageElements), count: max(8, bindings)))
                    let backing = [MLXArray](repeating: storage, count: bindings)
                    for castFP8 in [false, true] {
                        let mu: MLXArray? = castFP8 ? testMu : nil
                        let bd: MLXArray? = castFP8 ? testBd : nil
                        let probed = segmentsFusedGate(
                            queries: q, backing: backing, seqinfo: seqinfo, params: params,
                            previous: fence, valueOffsets: bOffsets, records: MLXArray(record),
                            gate: gate, fp8Multipliers: mu, fp8OverflowBound: bd,
                            dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                            simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                            source: source, probe: true)
                        eval(probed.fence, probed.out)
                        fence = probed.fence
                        for mp in partRange {
                            let r = segmentsFusedGate(
                                queries: q, backing: backing, seqinfo: seqinfo, params: params,
                                previous: fence, valueOffsets: bOffsets, records: MLXArray(record),
                                gate: gate, fp8Multipliers: mu, fp8OverflowBound: bd,
                                dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                                simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                                source: source, maxPart: mp, forceMaxPart: true)
                            fence = r.fence
                            warmOut += [r.fence, r.out]
                            if warmDynTree {
                                let rd = segmentsFusedGate(
                                    queries: q, backing: backing, seqinfo: seqinfo, params: params,
                                    previous: fence, valueOffsets: bOffsets, records: MLXArray(record),
                                    gate: gate, fp8Multipliers: mu, fp8OverflowBound: bd,
                                    dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                                    simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                                    source: source, treeMeta: dummyTreeMeta, maxPart: mp, forceMaxPart: true)
                                fence = rd.fence
                                warmOut += [rd.fence, rd.out]
                            }
                        }
                    }
                }
                if !warmOut.isEmpty { eval(warmOut) }
                return true
            }) ?? false
        lock.withLock { fusedGateVerified = passed }
        guard passed, lowTgemEnvEnabled,
            let ltHpt = lowTgemHeads(
                headDim: headDim, gqa: queryHeads / kvHeads, simdgroups: nsg, maxPart: maxPartitions)
        else { return }
        let ltPassed: Bool =
            (try? withError { () -> Bool in
                let ref = segments(
                    queries: q, backing: [storage], seqinfo: seqinfo, params: params,
                    previous: previous, valueOffsets: MLXArray(offsets), records: MLXArray(record),
                    dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                    simdgroups: nsg, partitionTokens: ptok, softcap: softcap, source: source,
                    maxPart: 1, forceMaxPart: true)
                let refGated = ref.out * sigmoid(gate)
                let refCast = ModelOptActivationKernels.fp8(
                    refGated, multipliers: testMu, overflowBound: testBd)
                let gotGated = segmentsFusedGate(
                    queries: q, backing: [storage], seqinfo: seqinfo, params: params,
                    previous: ref.fence, valueOffsets: MLXArray(offsets), records: MLXArray(record),
                    gate: gate, fp8Multipliers: nil, fp8OverflowBound: nil,
                    dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: ltHpt,
                    simdgroups: nsg, partitionTokens: ptok, softcap: softcap, source: source,
                    maxPart: 1, forceMaxPart: true, forceLowTgem: true)
                let gotCast = segmentsFusedGate(
                    queries: q, backing: [storage], seqinfo: seqinfo, params: params,
                    previous: gotGated.fence, valueOffsets: MLXArray(offsets), records: MLXArray(record),
                    gate: gate, fp8Multipliers: testMu, fp8OverflowBound: testBd,
                    dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: ltHpt,
                    simdgroups: nsg, partitionTokens: ptok, softcap: softcap, source: source,
                    maxPart: 1, forceMaxPart: true, forceLowTgem: true)
                let diff0 = (contiguous(gotGated.out).view(dtype: .uint16)
                    .!= contiguous(refGated).view(dtype: .uint16)).asType(.int32).sum()
                let diff1 = (contiguous(gotCast.out).view(dtype: .uint16)
                    .!= contiguous(refCast).view(dtype: .uint16)).asType(.int32).sum()
                let total = diff0 + diff1
                eval(total)
                guard total.item(Int.self) == 0 else { return false }
                let warmDynTree = CBv2DraftRegime.verifyTreeMode != 0
                    && CBv2DraftRegime.dynamicTreeEnabled && !dynTreeProbeFailed
                let partRange = dynamicMaxPartitions ? (1 ... maxPartitions) : (maxPartitions ... maxPartitions)
                var warmOut = [MLXArray]()
                var fence = gotCast.fence
                for bindings in PagedSegmentDispatchPlan.bindingClasses {
                    let bOffsets = MLXArray([Int64](repeating: Int64(pageElements), count: max(8, bindings)))
                    let backing = [MLXArray](repeating: storage, count: bindings)
                    for castFP8 in [false, true] {
                        let mu: MLXArray? = castFP8 ? testMu : nil
                        let bd: MLXArray? = castFP8 ? testBd : nil
                        let probed = segmentsFusedGate(
                            queries: q, backing: backing, seqinfo: seqinfo, params: params,
                            previous: fence, valueOffsets: bOffsets, records: MLXArray(record),
                            gate: gate, fp8Multipliers: mu, fp8OverflowBound: bd,
                            dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: ltHpt,
                            simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                            source: source, probe: true, forceLowTgem: true)
                        eval(probed.fence, probed.out)
                        fence = probed.fence
                        for mp in partRange {
                            let r = segmentsFusedGate(
                                queries: q, backing: backing, seqinfo: seqinfo, params: params,
                                previous: fence, valueOffsets: bOffsets, records: MLXArray(record),
                                gate: gate, fp8Multipliers: mu, fp8OverflowBound: bd,
                                dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: ltHpt,
                                simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                                source: source, maxPart: mp, forceMaxPart: true, forceLowTgem: true)
                            fence = r.fence
                            warmOut += [r.fence, r.out]
                            if warmDynTree {
                                let rd = segmentsFusedGate(
                                    queries: q, backing: backing, seqinfo: seqinfo, params: params,
                                    previous: fence, valueOffsets: bOffsets, records: MLXArray(record),
                                    gate: gate, fp8Multipliers: mu, fp8OverflowBound: bd,
                                    dtype: .bfloat16, kvHeads: kvHeads, pageSize: pageSize, hpt: ltHpt,
                                    simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                                    source: source, treeMeta: dummyTreeMeta, maxPart: mp,
                                    forceMaxPart: true, forceLowTgem: true)
                                fence = rd.fence
                                warmOut += [rd.fence, rd.out]
                            }
                        }
                    }
                }
                if !warmOut.isEmpty { eval(warmOut) }
                return true
            }) ?? false
        lock.withLock { lowTgemVerified = ltPassed }
    }

    /// Heads per threadgroup: the largest divisor of `gqa` at or under the
    /// requested count whose threadgroup memory fits Metal's limit.
    static func headsPerThreadgroup(headDim: Int, gqa: Int, simdgroups: Int) -> Int? {
        var hpt = min(gqa, requestedHeads)
        while hpt >= 1 {
            let floats =
                hpt * headDim + maxPartitions * simdgroups * hpt * (headDim + 2)
                + maxPartitions * hpt * (headDim + 2)
            if gqa % hpt == 0, floats * 4 <= PagedAttentionKernel.threadgroupMemoryLimit {
                return hpt
            }
            hpt -= 1
        }
        return nil
    }

    static func eligible(
        headDim: Int, gqa: Int, simdgroups: Int, partitions: Int, rows: Int,
        dynTree: Bool = false
    ) -> Int? {
        guard !threadgroupProbeFailed, !(dynTree && dynTreeProbeFailed),
            partitions >= 1, partitions <= maxPartitions,
            rows <= 64, 32 * simdgroups * maxPartitions <= 1024
        else { return nil }
        return headsPerThreadgroup(headDim: headDim, gqa: gqa, simdgroups: simdgroups)
    }

    private static let scratch = """
        threadgroup float q_smem[HPT * D];
        threadgroup float red_smem[MAXPART * NSG * HPT * (D + 2)];
        threadgroup float part_smem[MAXPART * HPT * (D + 2)];
        """

    private static let slabBody = """
        const int kvh = kcache_shape[1];
        const int maxp = tables_shape[1];
        \(scratch)
        const cbv2::PagedSlabAccessor<T> cache{kcache, vcache, (size_t)kvh * S * D};
        cbv2::paged_attention_onepass_impl<
            T, D, S, GQA, HPT, NSG, PTOK, MAXPART, HAS_SOFTCAP, cbv2::PagedSlabAccessor<T>,
            BLOCK>(
            q, cache, tables, seqinfo, params, kvh, maxp, q_smem, red_smem, part_smem, out,
            threadgroup_position_in_grid, thread_position_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """

    /// Records of a single-bucket plan are row-major with each row's own
    /// partition count, so (row, partition) is a prefix sum away.
    private static func segmentBody(bindings: Int) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        return """
            const int row = threadgroup_position_in_grid.y;
            int first = 0;
            for (int r = 0; r < row; r++) {
                first += (seqinfo[r * 8 + 1] + PTOK - 1) / PTOK;
            }
            const int own = (seqinfo[row * 8 + 1] + PTOK - 1) / PTOK;
            const int wanted = (int)simdgroup_index_in_threadgroup / NSG;
            const device int32_t* record =
                records + (first + min(wanted, own - 1)) * STRIDE;
            const cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)> cache{
                {\(pointers)}, value_offsets, record, (size_t)KVH * S * D};
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
                fence[0] = previous[0] + 1;
            }
            \(scratch)
            cbv2::paged_attention_onepass_impl<
                T, D, S, GQA, HPT, NSG, PTOK, MAXPART, HAS_SOFTCAP,
                cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)>, BLOCK, DYN_TREE>(
                q, cache, seqinfo, seqinfo, params, KVH, 0, q_smem, red_smem, part_smem, out,
                threadgroup_position_in_grid, thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, tree_meta);
            """
    }

    private static func name(
        _ kind: String, dtype: DType, headDim: Int, pageSize: Int, gqa: Int, hpt: Int,
        simdgroups: Int, partitionTokens: Int, softcap: Bool, maxPart: Int = maxPartitions
    ) -> String {
        let d: String
        switch dtype {
        case .float16: d = "f16"
        case .float32: d = "f32"
        case .bfloat16: d = "bf16"
        default: d = "dt\(dtype)"
        }
        return "cbv2_paged_onepass_\(kind)_\(d)_d\(headDim)_s\(pageSize)_g\(gqa)_h\(hpt)"
            + "_n\(simdgroups)_t\(partitionTokens)_p\(maxPart)_cap\(softcap ? 1 : 0)"
            + "_b\(block)"
    }

    /// Direct-slab entry: `[B, QH, D]` in the slab dtype.
    static func slab(
        queries: MLXArray, kSlab: MLXArray, vSlab: MLXArray, tables: MLXArray,
        seqinfo: MLXArray, params: MLXArray, softcap: Bool, pageSize: Int,
        writeFence: MLXArray, hpt: Int, simdgroups: Int, partitionTokens: Int,
        source: String, stream: StreamOrDevice
    ) -> MLXArray {
        let dtype = kSlab.dtype
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let kvh = kSlab.dim(1)
        let gqa = qh / kvh
        let kernelName = name(
            "slab", dtype: dtype, headDim: d, pageSize: pageSize, gqa: gqa, hpt: hpt,
            simdgroups: simdgroups, partitionTokens: partitionTokens, softcap: softcap)
        let kernel = lock.withLock { () -> MLXFast.MLXFastKernel in
            if let existing = kernels[kernelName] { return existing }
            let made = MLXFast.metalKernel(
                name: kernelName,
                inputNames: ["q", "kcache", "vcache", "tables", "seqinfo", "params", "wfence"],
                outputNames: ["out"], source: slabBody, header: source,
                ensureRowContiguous: true)
            kernels[kernelName] = made
            return made
        }
        let tg = 32 * simdgroups * maxPartitions
        return kernel(
            [queries, kSlab, vSlab, tables, seqinfo, params, writeFence],
            template: [
                ("T", dtype), ("D", d), ("S", pageSize), ("GQA", gqa), ("HPT", hpt),
                ("NSG", simdgroups), ("PTOK", partitionTokens), ("MAXPART", maxPartitions),
                ("HAS_SOFTCAP", softcap), ("BLOCK", block),
            ],
            grid: (kvh * (gqa / hpt) * tg, b, 1), threadGroup: (tg, 1, 1),
            outputShapes: [[b, qh, d]], outputDTypes: [dtype], stream: stream)[0]
    }

    nonisolated(unsafe) private static let dummyTreeMeta: MLXArray = {
        let arr = MLXArray.zeros([18], type: Int32.self)
        eval(arr)
        return arr
    }()

    private struct SegLaunchKey: Hashable {
        let dtype: DType
        let b: Int
        let qh: Int
        let d: Int
        let kvHeads: Int
        let pageSize: Int
        let hpt: Int
        let simdgroups: Int
        let partitionTokens: Int
        let maxPart: Int
        let softcap: Bool
        let bindings: Int
        let dynTree: Bool
    }

    nonisolated(unsafe) private static var segLaunchCache: [SegLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

    /// Segmented entry for a single-bucket plan. Returns the next fence and
    /// the attention output.
    static func segments(
        queries: MLXArray, backing: [MLXArray], seqinfo: MLXArray, params: MLXArray,
        previous: MLXArray, valueOffsets: MLXArray, records: MLXArray, dtype: DType,
        kvHeads: Int, pageSize: Int, hpt: Int, simdgroups: Int, partitionTokens: Int,
        softcap: Bool, source: String, probe: Bool = false, treeMeta: MLXArray? = nil,
        maxPart: Int? = nil, forceMaxPart: Bool = false
    ) -> (fence: MLXArray, out: MLXArray) {
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let gqa = qh / kvHeads
        let bindings = backing.count
        let allowDynPart =
            dynamicMaxPartitions
            && (forceMaxPart || !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance)
        let mp =
            (!probe && allowDynPart)
            ? min(maxPartitions, max(1, maxPart ?? maxPartitions))
            : maxPartitions
        let dynTree = treeMeta != nil
        let inputs =
            [queries] + backing + [
                seqinfo, params, previous, valueOffsets, records, treeMeta ?? dummyTreeMeta,
            ]
        if !probe {
            let lkey = SegLaunchKey(
                dtype: dtype, b: b, qh: qh, d: d, kvHeads: kvHeads, pageSize: pageSize,
                hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
                maxPart: mp, softcap: softcap, bindings: bindings, dynTree: dynTree)
            if let hit = lock.withLock({ segLaunchCache[lkey] }) {
                let outputs = hit(inputs)
                return (outputs[0], outputs[1])
            }
        }
        let kernelName =
            name(
                "segments\(bindings)_kvh\(kvHeads)", dtype: dtype, headDim: d, pageSize: pageSize,
                gqa: gqa, hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
                softcap: softcap, maxPart: mp)
        let kernel = lock.withLock { () -> MLXFast.MLXFastKernel in
            if let existing = kernels[kernelName] { return existing }
            if logs {
                FileHandle.standardError.write(Data("[onepass] new kernel \(kernelName)\n".utf8))
            }
            let made = MLXFast.metalKernel(
                name: kernelName,
                inputNames: ["q"] + (0 ..< bindings).map { "segment\($0)" }
                    + ["seqinfo", "params", "previous", "value_offsets", "records", "tree_meta"],
                outputNames: ["fence", "out"], source: segmentBody(bindings: bindings),
                header: source, ensureRowContiguous: true)
            kernels[kernelName] = made
            return made
        }
        let tg = 32 * simdgroups * mp
        let template: [(String, any KernelTemplateArg)] = [
            ("T", dtype), ("D", d), ("S", pageSize), ("KVH", kvHeads), ("GQA", gqa),
            ("HPT", hpt), ("NSG", simdgroups), ("PTOK", partitionTokens),
            ("MAXPART", mp), ("HAS_SOFTCAP", softcap), ("SEGMENTS", bindings),
            ("STRIDE", PagedSegmentDispatchPlan.recordStride), ("BLOCK", block),
            ("DYN_TREE", dynTree),
        ]
        if !probe {
            let lkey = SegLaunchKey(
                dtype: dtype, b: b, qh: qh, d: d, kvHeads: kvHeads, pageSize: pageSize,
                hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
                maxPart: mp, softcap: softcap, bindings: bindings, dynTree: dynTree)
            let launch = kernel.prepare(
                template: template,
                grid: (kvHeads * (gqa / hpt) * tg, b, 1),
                threadGroup: (tg, 1, 1),
                outputShapes: [[1], [b, qh, d]], outputDTypes: [.int32, dtype])
            lock.withLock { segLaunchCache[lkey] = launch }
            let outputs = launch(inputs)
            return (outputs[0], outputs[1])
        }
        let outputs = kernel(
            inputs,
            template: template,
            // The probe asks for the same thread count as one (tg / 2, 2, 1)
            // threadgroup: MLX accepts that shape only when the pipeline
            // holds `tg` threads as compiled, and throws otherwise.
            grid: (forceProbeFailure ? 2 * tg : tg / 2, 2, 1),
            threadGroup: (forceProbeFailure ? 2 * tg : tg / 2, 2, 1),
            outputShapes: [[1], [b, qh, d]], outputDTypes: [.int32, dtype])
        return (outputs[0], outputs[1])
    }

    nonisolated(unsafe) private static let dummyFP8Mu: MLXArray = {
        let a = MLXArray([Float(1), Float(1)])
        eval(a)
        return a
    }()
    nonisolated(unsafe) private static let dummyFP8Bound: MLXArray = {
        let a = MLXArray([Float(448)])
        eval(a)
        return a
    }()

    private static func segmentBodyFusedGate(bindings: Int) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        return """
            const int row = threadgroup_position_in_grid.y;
            int first = 0;
            for (int r = 0; r < row; r++) {
                first += (seqinfo[r * 8 + 1] + PTOK - 1) / PTOK;
            }
            const int own = (seqinfo[row * 8 + 1] + PTOK - 1) / PTOK;
            const int wanted = (int)simdgroup_index_in_threadgroup / NSG;
            const device int32_t* record =
                records + (first + min(wanted, own - 1)) * STRIDE;
            const cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)> cache{
                {\(pointers)}, value_offsets, record, (size_t)KVH * S * D};
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
                fence[0] = previous[0] + 1;
            }
            const float fp8_m0 = fp8_mu[0];
            const float fp8_m1 = fp8_mu[1];
            const float fp8_bound = fp8_bd[0];
            \(scratch)
            cbv2::paged_attention_onepass_impl<
                T, D, S, GQA, HPT, NSG, PTOK, MAXPART, HAS_SOFTCAP,
                cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)>, BLOCK, DYN_TREE,
                true, CAST_FP8>(
                q, cache, seqinfo, seqinfo, params, KVH, 0, q_smem, red_smem, part_smem, out,
                threadgroup_position_in_grid, thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, tree_meta,
                gate, fp8_m0, fp8_m1, fp8_bound);
            """
    }

    private static func segmentBodyFusedGateLowTgem(bindings: Int) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        return """
            const int row = threadgroup_position_in_grid.y;
            int first = 0;
            for (int r = 0; r < row; r++) {
                first += (seqinfo[r * 8 + 1] + PTOK - 1) / PTOK;
            }
            const int own = (seqinfo[row * 8 + 1] + PTOK - 1) / PTOK;
            const int wanted = (int)simdgroup_index_in_threadgroup / NSG;
            const device int32_t* record =
                records + (first + min(wanted, own - 1)) * STRIDE;
            const cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)> cache{
                {\(pointers)}, value_offsets, record, (size_t)KVH * S * D};
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
                fence[0] = previous[0] + 1;
            }
            const float fp8_m0 = fp8_mu[0];
            const float fp8_m1 = fp8_mu[1];
            const float fp8_bound = fp8_bd[0];
            threadgroup float red_smem[MAXPART * HPT * (NSG - 1) * (D + 2)];
            cbv2::paged_attention_onepass_lowtgem_impl<
                T, D, S, GQA, HPT, NSG, PTOK, MAXPART, HAS_SOFTCAP,
                cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)>, BLOCK, DYN_TREE,
                true, CAST_FP8>(
                q, cache, seqinfo, seqinfo, params, KVH, 0, red_smem, out,
                threadgroup_position_in_grid, thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, tree_meta,
                gate, fp8_m0, fp8_m1, fp8_bound);
            """
    }

    private struct SegGateLaunchKey: Hashable {
        let dtype: DType
        let b: Int
        let qh: Int
        let d: Int
        let kvHeads: Int
        let pageSize: Int
        let hpt: Int
        let simdgroups: Int
        let partitionTokens: Int
        let maxPart: Int
        let softcap: Bool
        let bindings: Int
        let dynTree: Bool
        let castFP8: Bool
        let lowTgem: Bool
    }

    nonisolated(unsafe) private static var segGateLaunchCache:
        [SegGateLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

    /// Segmented one-pass attention with fused 0-ULP `output * sigmoid(gate)`
    /// (+ optional ModelOpt FP8 activation cast when `fp8Multipliers != nil`).
    static func segmentsFusedGate(
        queries: MLXArray, backing: [MLXArray], seqinfo: MLXArray, params: MLXArray,
        previous: MLXArray, valueOffsets: MLXArray, records: MLXArray,
        gate: MLXArray, fp8Multipliers: MLXArray?, fp8OverflowBound: MLXArray?,
        dtype: DType, kvHeads: Int, pageSize: Int, hpt: Int, simdgroups: Int,
        partitionTokens: Int, softcap: Bool, source: String, probe: Bool = false,
        treeMeta: MLXArray? = nil, maxPart: Int? = nil, forceMaxPart: Bool = false,
        forceLowTgem: Bool = false
    ) -> (fence: MLXArray, out: MLXArray) {
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let gqa = qh / kvHeads
        let bindings = backing.count
        let allowDynPart =
            dynamicMaxPartitions
            && (forceMaxPart || !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance)
        let mp =
            (!probe && allowDynPart)
            ? min(maxPartitions, max(1, maxPart ?? maxPartitions))
            : maxPartitions
        let dynTree = treeMeta != nil
        let castFP8 = fp8Multipliers != nil && fp8OverflowBound != nil
        let ltHpt =
            (forceLowTgem || lowTgemActive)
            ? lowTgemHeads(headDim: d, gqa: gqa, simdgroups: simdgroups, maxPart: mp)
            : nil
        let useLowTgem = ltHpt != nil
        let effHpt = ltHpt ?? hpt
        let inputs =
            [queries] + backing + [
                seqinfo, params, previous, valueOffsets, records, treeMeta ?? dummyTreeMeta,
                gate, fp8Multipliers ?? dummyFP8Mu, fp8OverflowBound ?? dummyFP8Bound,
            ]
        if !probe {
            let lkey = SegGateLaunchKey(
                dtype: dtype, b: b, qh: qh, d: d, kvHeads: kvHeads, pageSize: pageSize,
                hpt: effHpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
                maxPart: mp, softcap: softcap, bindings: bindings, dynTree: dynTree,
                castFP8: castFP8, lowTgem: useLowTgem)
            if let hit = lock.withLock({ segGateLaunchCache[lkey] }) {
                let outputs = hit(inputs)
                return (outputs[0], outputs[1])
            }
        }
        let kindTag = useLowTgem ? "segments\(bindings)_kvh\(kvHeads)_fgate_lt" : "segments\(bindings)_kvh\(kvHeads)_fgate"
        let kernelName =
            name(
                kindTag, dtype: dtype, headDim: d,
                pageSize: pageSize, gqa: gqa, hpt: effHpt, simdgroups: simdgroups,
                partitionTokens: partitionTokens, softcap: softcap, maxPart: mp)
        let kernel = lock.withLock { () -> MLXFast.MLXFastKernel in
            if let existing = kernels[kernelName] { return existing }
            let made = MLXFast.metalKernel(
                name: kernelName,
                inputNames: ["q"] + (0 ..< bindings).map { "segment\($0)" }
                    + [
                        "seqinfo", "params", "previous", "value_offsets", "records", "tree_meta",
                        "gate", "fp8_mu", "fp8_bd",
                    ],
                outputNames: ["fence", "out"],
                source: useLowTgem
                    ? segmentBodyFusedGateLowTgem(bindings: bindings)
                    : segmentBodyFusedGate(bindings: bindings),
                header: source, ensureRowContiguous: true)
            kernels[kernelName] = made
            return made
        }
        let tg = 32 * simdgroups * mp
        let template: [(String, any KernelTemplateArg)] = [
            ("T", dtype), ("D", d), ("S", pageSize), ("KVH", kvHeads), ("GQA", gqa),
            ("HPT", effHpt), ("NSG", simdgroups), ("PTOK", partitionTokens),
            ("MAXPART", mp), ("HAS_SOFTCAP", softcap), ("SEGMENTS", bindings),
            ("STRIDE", PagedSegmentDispatchPlan.recordStride), ("BLOCK", block),
            ("DYN_TREE", dynTree), ("CAST_FP8", castFP8),
        ]
        if !probe {
            let lkey = SegGateLaunchKey(
                dtype: dtype, b: b, qh: qh, d: d, kvHeads: kvHeads, pageSize: pageSize,
                hpt: effHpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
                maxPart: mp, softcap: softcap, bindings: bindings, dynTree: dynTree,
                castFP8: castFP8, lowTgem: useLowTgem)
            let launch = kernel.prepare(
                template: template,
                grid: (kvHeads * (gqa / effHpt) * tg, b, 1),
                threadGroup: (tg, 1, 1),
                outputShapes: [[1], [b, qh, d]], outputDTypes: [.int32, dtype])
            lock.withLock { segGateLaunchCache[lkey] = launch }
            let outputs = launch(inputs)
            return (outputs[0], outputs[1])
        }
        let outputs = kernel(
            inputs,
            template: template,
            grid: (forceProbeFailure ? 2 * tg : tg / 2, 2, 1),
            threadGroup: (forceProbeFailure ? 2 * tg : tg / 2, 2, 1),
            outputShapes: [[1], [b, qh, d]], outputDTypes: [.int32, dtype])
        return (outputs[0], outputs[1])
    }
}
