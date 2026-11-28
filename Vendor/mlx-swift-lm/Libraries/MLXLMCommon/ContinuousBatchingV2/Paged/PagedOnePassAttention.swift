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
    private static let requestedHeads: Int? = {
        if let raw = environment["MUSE_PAGED_ONEPASS_HPT"], let value = Int(raw), value >= 1 {
            return value
        }
        return nil
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
    static var threadgroupProbeFailed: Bool { probeFailed(record: useRecordKernel) }
    nonisolated(unsafe) private static var threadgroupProbeFailedLow = false
    nonisolated(unsafe) private static var threadgroupProbeFailedRecord = false
    static func probeFailed(record: Bool) -> Bool {
        record ? threadgroupProbeFailedRecord : threadgroupProbeFailedLow
    }
    /// High-acceptance verify rounds (`CBv2DraftRegime.lowOnlyVerifySpeedups &&
    /// !CBv2DraftRegime.lowAcceptance`) launch the record's (c95a27f5) one-pass
    /// kernel (`paged_attention_onepass_impl_rec`: threadgroup q / partition
    /// buffers, the record's heads-per-threadgroup rule); low-acceptance rounds
    /// launch the register-query / in-place-merge kernel. Same bytes either way.
    static var useRecordKernel: Bool {
        CBv2DraftRegime.lowOnlyVerifySpeedups && !CBv2DraftRegime.lowAcceptance
    }
    /// Set by `warm` when the `DYN_TREE = true` one-pass variant fails the threadgroup probe;
    /// leaves `DYN_TREE = false` unaffected and falls back to two-pass only for dynamic-tree rounds.
    static var dynTreeProbeFailed: Bool { dynTreeProbeFailedFor(record: useRecordKernel) }
    nonisolated(unsafe) private static var dynTreeProbeFailedLow = false
    nonisolated(unsafe) private static var dynTreeProbeFailedRecord = false
    static func dynTreeProbeFailedFor(record: Bool) -> Bool {
        record ? dynTreeProbeFailedRecord : dynTreeProbeFailedLow
    }
    /// `CBV2_ONEPASS_APPEND=0`: linear rectangular verify keeps the separate K/V
    /// scatter instead of the write-owning one-pass variant (same bytes either way).
    static let appendEnabled = environment["CBV2_ONEPASS_APPEND"] != "0"
    /// `CBV2_ONEPASS_OUTGATE=0`: keep separate BF16 `output * sigmoid(gate)` launch.
    public static let outputGateEnabled = environment["CBV2_ONEPASS_OUTGATE"] != "0"
    /// True once `warm` qualified at least one write-owning variant.
    static var appendWarmed: Bool { appendEnabled && lock.withLock { !appendVariants.isEmpty } }
    /// `MUSE_PAGED_ONEPASS_PROBE_FAIL=1`: the probe asks for a threadgroup no
    /// pipeline holds, to exercise the two-pass fallback (a self-test).
    private static let forceProbeFailure = environment["MUSE_PAGED_ONEPASS_PROBE_FAIL"] == "1"
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var appendVariants: Set<AppendKey> = []
    nonisolated(unsafe) private static var outGateVariants: Set<OutGateKey> = []
    nonisolated(unsafe) private static var slabLaunchCache:
        [SlabLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static var checked = 0
    nonisolated(unsafe) private static var differing = 0
    nonisolated(unsafe) private static var appendedChecks = 0
    nonisolated(unsafe) private static var loggedSharedLaunches: Set<String> = []

    static func logSharedLaunch(
        record: Bool, partitions: Int, hpt: Int, dynTree: Bool,
        appending: Bool, gated: Bool, fp8Cast: Bool
    ) {
        guard logs else { return }
        let mode = appending ? "append" : (gated ? (fp8Cast ? "outgate_fp8" : "outgate") : "plain")
        let message =
            "[onepass] live shared record=\(record) partitions=\(partitions) hpt=\(hpt) mode=\(mode) dynTree=\(dynTree)\n"
        if lock.withLock({ loggedSharedLaunches.insert(message).inserted }) {
            FileHandle.standardError.write(Data(message.utf8))
        }
    }

    /// Self-test bookkeeping: evaluates both results and counts elements
    /// whose bits differ.
    static func compare(onePass: MLXArray, twoPass: MLXArray, appended: Bool = false) {
        let a = onePass.asType(.float32).asArray(Float.self)
        let b = twoPass.asType(.float32).asArray(Float.self)
        var bad = a.count == b.count ? 0 : max(a.count, b.count)
        if a.count == b.count {
            for i in 0 ..< a.count where a[i].bitPattern != b[i].bitPattern { bad += 1 }
        }
        lock.withLock {
            checked += 1
            if appended { appendedChecks += 1 }
            differing += bad
            if bad != 0 || checked % 500 == 0 {
                FileHandle.standardError.write(
                    Data(
                        "[onepass-check] dispatches=\(checked) rows=\(onePass.dim(0)) elements=\(a.count) differing=\(bad) total_differing=\(differing) appends=\(appendedChecks)\n"
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
        guard let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa)
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
        // The write-owning variant writes its scratch page: its own contiguous storage.
        let appendStorage = contiguous(MLXArray.zeros([2 * pageElements], dtype: dtype))
        let pending = MLXArray.zeros([1, kvHeads, headDim], dtype: dtype)
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
        variants: for useRec in [true, false] {
        guard
            let baselineHeads = eligible(
                headDim: headDim, gqa: gqa, simdgroups: nsg, partitions: 1, rows: 1,
                record: useRec)
        else { continue }
        let partRange = dynamicMaxPartitions ? (1 ... maxPartitions) : (maxPartitions ... maxPartitions)
        let launchShapes = partRange.flatMap { mp -> [(Int, Int)] in
            let baseline = (mp, baselineHeads)
            if let heads = fourHeads(
                headDim: headDim, gqa: gqa, simdgroups: nsg, partitions: mp, record: useRec)
            {
                return [baseline, (mp, heads)]
            }
            return [baseline]
        }
        for bindings in PagedSegmentDispatchPlan.bindingClasses {
            let offsets = MLXArray([Int64](repeating: Int64(pageElements), count: max(8, bindings)))
            let records = MLXArray(record)
            let backing = [MLXArray](repeating: storage, count: bindings)
            // Probe one threadgroup on zero inputs.
            func probe(_ hpt: Int, maxPart: Int? = nil, treeMeta: MLXArray? = nil) -> Bool {
                (try? withError { () -> Bool in
                    let probed = segments(
                        queries: q, backing: backing, seqinfo: seqinfo, params: params,
                        previous: previous, valueOffsets: offsets, records: records,
                        dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                        simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                        source: source, probe: true, treeMeta: treeMeta, maxPart: maxPart,
                        forceMaxPart: maxPart != nil, record: useRec)
                    eval(probed.fence, probed.out)
                    return true
                }) ?? false
            }

            let held = probe(baselineHeads)
            if logs {
                FileHandle.standardError.write(
                    Data(
                    "[onepass] threadgroup probe record=\(useRec) bindings=\(bindings) held=\(held)\n"
                        .utf8))
            }
            guard held else {
                if useRec {
                    threadgroupProbeFailedRecord = true
                } else {
                    threadgroupProbeFailedLow = true
                }
                continue variants
            }
            let warmDynTree = CBv2DraftRegime.verifyTreeMode != 0 && CBv2DraftRegime.dynamicTreeEnabled
            if warmDynTree && !dynTreeProbeFailedFor(record: useRec)
                && !probe(baselineHeads, treeMeta: dummyTreeMeta)
            {
                if useRec {
                    dynTreeProbeFailedRecord = true
                } else {
                    dynTreeProbeFailedLow = true
                }
            }
            for (mp, hpt) in launchShapes {
                let wider = hpt != baselineHeads
                guard !wider || probe(hpt, maxPart: mp) else { continue }
                let result = segments(
                    queries: q, backing: backing,
                    seqinfo: seqinfo, params: params, previous: previous,
                    valueOffsets: offsets, records: records, dtype: dtype,
                    kvHeads: kvHeads, pageSize: pageSize, hpt: hpt, simdgroups: nsg,
                    partitionTokens: ptok, softcap: softcap, source: source, maxPart: mp,
                    forceMaxPart: true, record: useRec)
                outputs += [result.fence, result.out]
                let warmTreeForShape =
                    warmDynTree && !dynTreeProbeFailedFor(record: useRec)
                    && (!wider || probe(hpt, maxPart: mp, treeMeta: dummyTreeMeta))
                if warmTreeForShape {
                    let dynResult = segments(
                        queries: q, backing: backing,
                        seqinfo: seqinfo, params: params, previous: result.fence,
                        valueOffsets: offsets, records: records, dtype: dtype,
                        kvHeads: kvHeads, pageSize: pageSize, hpt: hpt, simdgroups: nsg,
                        partitionTokens: ptok, softcap: softcap, source: source,
                        treeMeta: dummyTreeMeta, maxPart: mp, forceMaxPart: true, record: useRec)
                    outputs += [dynResult.fence, dynResult.out]
                }
                if appendEnabled {
                    outputs += warmAppend(
                        q: q, storage: appendStorage, bindings: bindings, seqinfo: seqinfo,
                        params: params, previous: previous, valueOffsets: offsets,
                        records: records, pending: pending, dtype: dtype,
                        headDim: headDim, kvHeads: kvHeads, gqa: gqa, pageSize: pageSize,
                        hpt: hpt, simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                        source: source, maxPart: mp, record: useRec)
                }
                if outputGateEnabled && !useRec && dtype == .bfloat16 {
                    let dummyMu = MLXArray([Float(1.0), Float(1.0)])
                    let dummyBd = MLXArray([Float(448.0)])
                    for fp8Pair: (MLXArray, MLXArray)? in [nil, (dummyMu, dummyBd)] {
                        outputs += warmOutputGate(
                            q: q, storage: storage, bindings: bindings, seqinfo: seqinfo,
                            params: params, previous: previous, valueOffsets: offsets,
                            records: records, dtype: dtype, headDim: headDim, kvHeads: kvHeads,
                            gqa: gqa, pageSize: pageSize, hpt: hpt, simdgroups: nsg,
                            partitionTokens: ptok, softcap: softcap, source: source, maxPart: mp,
                            treeMeta: nil, fp8Cast: fp8Pair)
                        if warmTreeForShape {
                            outputs += warmOutputGate(
                                q: q, storage: storage, bindings: bindings, seqinfo: seqinfo,
                                params: params, previous: previous, valueOffsets: offsets,
                                records: records, dtype: dtype, headDim: headDim,
                                kvHeads: kvHeads, gqa: gqa, pageSize: pageSize, hpt: hpt,
                                simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                                source: source, maxPart: mp, treeMeta: dummyTreeMeta, fp8Cast: fp8Pair)
                        }
                    }
                }
            }
        }
        }
    }

    private static func warmOutputGate(
        q: MLXArray, storage: MLXArray, bindings: Int, seqinfo: MLXArray, params: MLXArray,
        previous: MLXArray, valueOffsets: MLXArray, records: MLXArray, dtype: DType,
        headDim: Int, kvHeads: Int, gqa: Int, pageSize: Int, hpt: Int, simdgroups: Int,
        partitionTokens: Int, softcap: Bool, source: String, maxPart: Int, treeMeta: MLXArray?,
        fp8Cast: (multipliers: MLXArray, overflowBound: MLXArray)?
    ) -> [MLXArray] {
        let mode = fp8Cast != nil ? 2 : 1
        let backing = [MLXArray](repeating: storage, count: bindings)
        let held =
            (try? withError { () -> Bool in
                let probed = segments(
                    queries: q, backing: backing, seqinfo: seqinfo, params: params,
                    previous: previous, valueOffsets: valueOffsets, records: records,
                    dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                    simdgroups: simdgroups, partitionTokens: partitionTokens, softcap: softcap,
                    source: source, probe: true, treeMeta: treeMeta, maxPart: maxPart,
                    forceMaxPart: true, outputGate: q, fp8Cast: fp8Cast, record: false)
                eval(probed.fence, probed.out)
                return true
            }) ?? false
        if logs {
            FileHandle.standardError.write(
                Data(
                    "[onepass] outgate probe mode=\(mode) dynTree=\(treeMeta != nil) bindings=\(bindings) maxpart=\(maxPart) held=\(held)\n"
                        .utf8))
        }
        guard held else { return [] }
        let result = segments(
            queries: q, backing: backing, seqinfo: seqinfo, params: params,
            previous: previous, valueOffsets: valueOffsets, records: records, dtype: dtype,
            kvHeads: kvHeads, pageSize: pageSize, hpt: hpt, simdgroups: simdgroups,
            partitionTokens: partitionTokens, softcap: softcap, source: source,
            treeMeta: treeMeta, maxPart: maxPart, forceMaxPart: true, outputGate: q,
            fp8Cast: fp8Cast, record: false)
        let key = OutGateKey(
            dtype: dtype, headDim: headDim, kvHeads: kvHeads, gqa: gqa, pageSize: pageSize,
            hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens, softcap: softcap,
            maxPart: maxPart, bindings: bindings, dynTree: treeMeta != nil, mode: mode)
        lock.withLock { _ = outGateVariants.insert(key) }
        return [result.fence, result.out]
    }

    /// Probes and compiles the write-owning (linear-only) variant at one
    /// MAXPART on scratch pages; a variant that fails its probe is not used.
    private static func warmAppend(
        q: MLXArray, storage: MLXArray, bindings: Int, seqinfo: MLXArray, params: MLXArray,
        previous: MLXArray, valueOffsets: MLXArray, records: MLXArray, pending: MLXArray,
        dtype: DType, headDim: Int, kvHeads: Int, gqa: Int, pageSize: Int, hpt: Int,
        simdgroups: Int, partitionTokens: Int, softcap: Bool, source: String, maxPart: Int,
        record: Bool
    ) -> [MLXArray] {
        let backing = [MLXArray](repeating: storage, count: bindings)
        let held =
            (try? withError { () -> Bool in
                let probed = segments(
                    queries: q, backing: backing, seqinfo: seqinfo, params: params,
                    previous: previous, valueOffsets: valueOffsets, records: records,
                    dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                    simdgroups: simdgroups, partitionTokens: partitionTokens, softcap: softcap,
                    source: source, probe: true, maxPart: maxPart, forceMaxPart: true,
                    pending: (pending, pending), record: record)
                eval(probed.fence, probed.out)
                return true
            }) ?? false
        if logs {
            FileHandle.standardError.write(
                Data(
                    "[onepass] append probe record=\(record) bindings=\(bindings) maxpart=\(maxPart) held=\(held)\n"
                        .utf8))
        }
        guard held else { return [] }
        let result = segments(
            queries: q, backing: backing, seqinfo: seqinfo, params: params,
            previous: previous, valueOffsets: valueOffsets, records: records, dtype: dtype,
            kvHeads: kvHeads, pageSize: pageSize, hpt: hpt, simdgroups: simdgroups,
            partitionTokens: partitionTokens, softcap: softcap, source: source,
            maxPart: maxPart, forceMaxPart: true, pending: (pending, pending), record: record)
        let key = AppendKey(
            dtype: dtype, headDim: headDim, kvHeads: kvHeads, gqa: gqa, pageSize: pageSize,
            hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
            softcap: softcap, maxPart: maxPart, bindings: bindings, record: record)
        lock.withLock { _ = appendVariants.insert(key) }
        return [result.fence, result.out]
    }

    private static func fitsThreadgroup(
        _ heads: Int, headDim: Int, simdgroups: Int, partitions: Int, record: Bool
    ) -> Bool {
        let stride = headDim + PagedAttentionKernel.mergeRecordMetaFloats
        let reduction = partitions * simdgroups * heads * stride
        let extra = record ? heads * (headDim + partitions * stride) : 0
        return (reduction + extra) * MemoryLayout<Float32>.size
            <= PagedAttentionKernel.threadgroupMemoryLimit
    }

    /// Heads per threadgroup for the baseline launch and explicit head overrides.
    static func headsPerThreadgroup(
        headDim: Int, gqa: Int, simdgroups: Int, record: Bool = useRecordKernel
    ) -> Int? {
        // Two heads share each K/V row load: at 8 rows the one-head form is
        // throughput-bound (measured 43 vs 31 us per layer), at 1 row equal.
        var hpt = min(gqa, requestedHeads ?? 2)
        while hpt >= 1 {
            if gqa % hpt == 0,
                fitsThreadgroup(
                    hpt, headDim: headDim, simdgroups: simdgroups, partitions: maxPartitions,
                    record: record)
            {
                return hpt
            }
            hpt -= 1
        }
        return nil
    }

    private static func fourHeads(
        headDim: Int, gqa: Int, simdgroups: Int, partitions: Int, record: Bool
    ) -> Int? {
        let heads = 4
        guard !record, dynamicMaxPartitions, requestedHeads == nil, gqa.isMultiple(of: heads),
            fitsThreadgroup(
                heads, headDim: headDim, simdgroups: simdgroups, partitions: partitions,
                record: record)
        else { return nil }
        return heads
    }

    static func eligible(
        headDim: Int, gqa: Int, simdgroups: Int, partitions: Int, rows: Int,
        dynTree: Bool = false, record: Bool = useRecordKernel
    ) -> Int? {
        guard !probeFailed(record: record), !(dynTree && dynTreeProbeFailedFor(record: record)),
            partitions >= 1, partitions <= maxPartitions,
            rows <= 64, 32 * simdgroups * maxPartitions <= 1024
        else { return nil }
        return headsPerThreadgroup(
            headDim: headDim, gqa: gqa, simdgroups: simdgroups, record: record)
    }

    private static let scratch = """
        threadgroup float red_smem[MAXPART * NSG * HPT * (D + 2)];
        """

    private static let scratchRecord = """
        threadgroup float q_smem[HPT * D];
        threadgroup float red_smem[MAXPART * NSG * HPT * (D + 2)];
        threadgroup float part_smem[MAXPART * HPT * (D + 2)];
        """

    private static let slabBodyRecord = """
        const int kvh = kcache_shape[1];
        const int maxp = tables_shape[1];
        \(scratchRecord)
        const cbv2::PagedSlabAccessor<T> cache{kcache, vcache, (size_t)kvh * S * D};
        cbv2::paged_attention_onepass_impl_rec<
            T, D, S, GQA, HPT, NSG, PTOK, MAXPART, HAS_SOFTCAP, cbv2::PagedSlabAccessor<T>,
            BLOCK>(
            q, cache, tables, seqinfo, params, kvh, maxp, q_smem, red_smem, part_smem, out,
            threadgroup_position_in_grid, thread_position_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """

    private static let slabBody = """
        const int kvh = kcache_shape[1];
        const int maxp = tables_shape[1];
        \(scratch)
        const cbv2::PagedSlabAccessor<T> cache{kcache, vcache, (size_t)kvh * S * D};
        cbv2::paged_attention_onepass_impl<
            T, D, S, GQA, HPT, NSG, PTOK, MAXPART, HAS_SOFTCAP, cbv2::PagedSlabAccessor<T>,
            BLOCK>(
            q, cache, tables, seqinfo, params, kvh, maxp, red_smem, out,
            threadgroup_position_in_grid, thread_position_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """

    /// Records of a single-bucket plan are row-major with each row's own
    /// partition count, so (row, partition) is a prefix sum away.
    private static func segmentBody(
        bindings: Int, appending: Bool = false, outGateMode: Int = 0, record: Bool = false
    ) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        // The read-only source stays byte-identical; only the append or outGate variant adds arguments.
        let extraTpl =
            appending ? ", true" : (outGateMode > 0 ? ", false, \(outGateMode)" : "")
        let extraArgs =
            appending
            ? ", knew, vnew"
            : (outGateMode == 2
                ? ", nullptr, nullptr, out_gate, fp8_mu[0], fp8_mu[1], fp8_bd[0]"
                : (outGateMode == 1 ? ", nullptr, nullptr, out_gate" : ""))
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
            \(record ? scratchRecord : scratch)
            cbv2::paged_attention_onepass_impl\(record ? "_rec" : "")<
                T, D, S, GQA, HPT, NSG, PTOK, MAXPART, HAS_SOFTCAP,
                cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)>, BLOCK, DYN_TREE\(extraTpl)>(
                q, cache, seqinfo, seqinfo, params, KVH, 0,
                \(record ? "q_smem, red_smem, part_smem" : "red_smem"), out,
                threadgroup_position_in_grid, thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, tree_meta\(extraArgs));
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
        source: String, stream: StreamOrDevice, record: Bool = useRecordKernel
    ) -> MLXArray {
        let dtype = kSlab.dtype
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let kvh = kSlab.dim(1)
        let gqa = qh / kvh
        let lkey = SlabLaunchKey(
            dtype: dtype, b: b, qh: qh, d: d, kvHeads: kvh, pageSize: pageSize,
            hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
            maxPart: maxPartitions, softcap: softcap, record: record)
        let launch: MLXFast.MLXFastKernel.Launch
        if let hit = lock.withLock({ slabLaunchCache[lkey] }) {
            launch = hit
        } else {
            let kernelName = name(
                "slab", dtype: dtype, headDim: d, pageSize: pageSize, gqa: gqa, hpt: hpt,
                simdgroups: simdgroups, partitionTokens: partitionTokens, softcap: softcap)
                + (record ? "_rec" : "")
            let kernel = lock.withLock { () -> MLXFast.MLXFastKernel in
                if let existing = kernels[kernelName] { return existing }
                let made = MLXFast.metalKernel(
                    name: kernelName,
                    inputNames: ["q", "kcache", "vcache", "tables", "seqinfo", "params", "wfence"],
                    outputNames: ["out"], source: record ? slabBodyRecord : slabBody, header: source,
                    ensureRowContiguous: true)
                kernels[kernelName] = made
                return made
            }
            let tg = 32 * simdgroups * maxPartitions
            let made = kernel.prepare(
                template: [
                    ("T", dtype), ("D", d), ("S", pageSize), ("GQA", gqa), ("HPT", hpt),
                    ("NSG", simdgroups), ("PTOK", partitionTokens), ("MAXPART", maxPartitions),
                    ("HAS_SOFTCAP", softcap), ("BLOCK", block),
                ],
                grid: (kvh * (gqa / hpt) * tg, b, 1), threadGroup: (tg, 1, 1),
                outputShapes: [[b, qh, d]], outputDTypes: [dtype])
            lock.withLock {
                if slabLaunchCache.count < 1024 { slabLaunchCache[lkey] = made }
            }
            launch = made
        }
        return launch(
            [queries, kSlab, vSlab, tables, seqinfo, params, writeFence], stream: stream)[0]
    }

    private struct SlabLaunchKey: Hashable {
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
        let record: Bool
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
        let append: Bool
        let outGateMode: Int
        let record: Bool
    }

    private struct AppendKey: Hashable {
        let dtype: DType
        let headDim: Int
        let kvHeads: Int
        let gqa: Int
        let pageSize: Int
        let hpt: Int
        let simdgroups: Int
        let partitionTokens: Int
        let softcap: Bool
        let maxPart: Int
        let bindings: Int
        let record: Bool
    }

    private struct OutGateKey: Hashable {
        let dtype: DType
        let headDim: Int
        let kvHeads: Int
        let gqa: Int
        let pageSize: Int
        let hpt: Int
        let simdgroups: Int
        let partitionTokens: Int
        let softcap: Bool
        let maxPart: Int
        let bindings: Int
        let dynTree: Bool
        let mode: Int
    }

    /// The MAXPART `segments` compiles for a call (one rule for launch and lookup).
    private static func resolvedMaxPart(
        _ maxPart: Int?, probe: Bool, forceMaxPart: Bool, record: Bool
    ) -> Int {
        let allowDynPart = dynamicMaxPartitions && (forceMaxPart || !record)
        return ((!probe || forceMaxPart) && allowDynPart)
            ? min(maxPartitions, max(1, maxPart ?? maxPartitions))
            : maxPartitions
    }

    /// Select four heads only after the matching Metal pipeline passed warm-up.
    static func warmedSegmentHeads(
        _ baseline: Int, dtype: DType, headDim: Int, kvHeads: Int, gqa: Int, pageSize: Int,
        simdgroups: Int, partitionTokens: Int, softcap: Bool, partitions: Int, rows: Int,
        bindings: Int, dynTree: Bool, appending: Bool, outputGateMode: Int, record: Bool
    ) -> Int {
        let mp = resolvedMaxPart(partitions, probe: false, forceMaxPart: false, record: record)
        guard rows > 1,
            let hpt = fourHeads(
                headDim: headDim, gqa: gqa, simdgroups: simdgroups, partitions: mp, record: record)
        else { return baseline }
        let gateMode =
            !appending && canFuseOutputGate(dtype: dtype, record: record) ? outputGateMode : 0
        // Warm-up uses one row. Metal reuses that pipeline for the live row grid.
        let key = SegLaunchKey(
            dtype: dtype, b: 1, qh: kvHeads * gqa, d: headDim, kvHeads: kvHeads, pageSize: pageSize,
            hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens, maxPart: mp,
            softcap: softcap, bindings: bindings, dynTree: dynTree, append: appending,
            outGateMode: gateMode, record: record)
        return lock.withLock { segLaunchCache[key] != nil } ? hpt : baseline
    }

    /// Whether `warm` qualified the write-owning variant a live `segments` call
    /// with these arguments would launch.
    static func supportsAppend(
        dtype: DType, headDim: Int, kvHeads: Int, gqa: Int, pageSize: Int, hpt: Int,
        simdgroups: Int, partitionTokens: Int, softcap: Bool, maxPart: Int?, bindings: Int,
        record: Bool = useRecordKernel
    ) -> Bool {
        guard appendEnabled else { return false }
        let key = AppendKey(
            dtype: dtype, headDim: headDim, kvHeads: kvHeads, gqa: gqa, pageSize: pageSize,
            hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens, softcap: softcap,
            maxPart: resolvedMaxPart(maxPart, probe: false, forceMaxPart: false, record: record),
            bindings: bindings, record: record)
        return lock.withLock { appendVariants.contains(key) }
    }

    private static func canFuseOutputGate(dtype: DType, record: Bool) -> Bool {
        outputGateEnabled && !record && dtype == .bfloat16
            && (!CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.xnorDraftHeadEligible)
    }

    static func supportsOutputGate(
        dtype: DType, headDim: Int, kvHeads: Int, gqa: Int, pageSize: Int, hpt: Int,
        simdgroups: Int, partitionTokens: Int, softcap: Bool, maxPart: Int?, bindings: Int,
        dynTree: Bool, fp8Cast: Bool = false, record: Bool = useRecordKernel
    ) -> Bool {
        guard canFuseOutputGate(dtype: dtype, record: record) else { return false }
        let key = OutGateKey(
            dtype: dtype, headDim: headDim, kvHeads: kvHeads, gqa: gqa, pageSize: pageSize,
            hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens, softcap: softcap,
            maxPart: resolvedMaxPart(maxPart, probe: false, forceMaxPart: false, record: record),
            bindings: bindings, dynTree: dynTree, mode: fp8Cast ? 2 : 1)
        return lock.withLock { outGateVariants.contains(key) }
    }

    nonisolated(unsafe) private static var segLaunchCache: [SegLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

    /// Segmented entry for a single-bucket plan. Returns the next fence and
    /// the attention output.
    static func segments(
        queries: MLXArray, backing: [MLXArray], seqinfo: MLXArray, params: MLXArray,
        previous: MLXArray, valueOffsets: MLXArray, records: MLXArray, dtype: DType,
        kvHeads: Int, pageSize: Int, hpt: Int, simdgroups: Int, partitionTokens: Int,
        softcap: Bool, source: String, probe: Bool = false, treeMeta: MLXArray? = nil,
        maxPart: Int? = nil, forceMaxPart: Bool = false,
        pending: (keys: MLXArray, values: MLXArray)? = nil,
        outputGate: MLXArray? = nil,
        fp8Cast: (multipliers: MLXArray, overflowBound: MLXArray)? = nil,
        record: Bool = useRecordKernel
    ) -> (fence: MLXArray, out: MLXArray) {
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let gqa = qh / kvHeads
        let bindings = backing.count
        let mp = resolvedMaxPart(maxPart, probe: probe, forceMaxPart: forceMaxPart, record: record)
        let dynTree = treeMeta != nil
        let appending = pending != nil
        let outGateMode = (outputGate != nil) ? (fp8Cast != nil ? 2 : 1) : 0
        precondition(!(appending && dynTree), "the fused append serves linear rows only")
        precondition(!(appending && outGateMode > 0), "fused output gate serves pre-written rows only")
        precondition(!(record && outGateMode > 0), "fused output gate serves low-acceptance kernels only")
        var inputs =
            [queries] + backing + [
                seqinfo, params, previous, valueOffsets, records, treeMeta ?? dummyTreeMeta,
            ]
        if let pending {
            inputs += [pending.keys, pending.values]
        } else if let outputGate {
            inputs.append(outputGate)
            if let fp8Cast {
                inputs += [fp8Cast.multipliers, fp8Cast.overflowBound]
            }
        }
        if !probe {
            let lkey = SegLaunchKey(
                dtype: dtype, b: b, qh: qh, d: d, kvHeads: kvHeads, pageSize: pageSize,
                hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
                maxPart: mp, softcap: softcap, bindings: bindings, dynTree: dynTree,
                append: appending, outGateMode: outGateMode, record: record)
            if let hit = lock.withLock({ segLaunchCache[lkey] }) {
                let outputs = hit(inputs)
                return (outputs[0], outputs[1])
            }
        }
        let gateSuffix = outGateMode == 2 ? "_outgate_fp8" : (outGateMode == 1 ? "_outgate" : "")
        let kernelName =
            name(
                "segments\(bindings)_kvh\(kvHeads)", dtype: dtype, headDim: d, pageSize: pageSize,
                gqa: gqa, hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
                softcap: softcap, maxPart: mp) + (appending ? "_append" : "")
            + gateSuffix + (record ? "_rec" : "")
        let kernel = lock.withLock { () -> MLXFast.MLXFastKernel in
            if let existing = kernels[kernelName] { return existing }
            if logs {
                FileHandle.standardError.write(Data("[onepass] new kernel \(kernelName)\n".utf8))
            }
            let segmentNames = (0 ..< bindings).map { "segment\($0)" }
            let extraNames =
                appending
                ? ["knew", "vnew"]
                : (outGateMode == 2
                    ? ["out_gate", "fp8_mu", "fp8_bd"]
                    : (outGateMode == 1 ? ["out_gate"] : []))
            let made = MLXFast.metalKernel(
                name: kernelName,
                inputNames: ["q"] + segmentNames
                    + ["seqinfo", "params", "previous", "value_offsets", "records", "tree_meta"]
                    + extraNames,
                outputNames: ["fence", "out"],
                source: segmentBody(
                    bindings: bindings, appending: appending, outGateMode: outGateMode,
                    record: record),
                header: source, ensureRowContiguous: true,
                mutableInputs: appending ? segmentNames : [])
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
                maxPart: mp, softcap: softcap, bindings: bindings, dynTree: dynTree,
                append: appending, outGateMode: outGateMode, record: record)
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
}