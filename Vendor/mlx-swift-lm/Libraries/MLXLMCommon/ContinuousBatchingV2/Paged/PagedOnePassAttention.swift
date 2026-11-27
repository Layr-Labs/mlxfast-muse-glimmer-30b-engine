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
    /// `CBV2_ONEPASS_APPEND=0`: linear rectangular verify keeps the separate K/V
    /// scatter instead of the write-owning one-pass variant (same bytes either way).
    static let appendEnabled = environment["CBV2_ONEPASS_APPEND"] != "0"
    /// True once `warm` qualified at least one write-owning variant.
    static var appendWarmed: Bool { appendEnabled && lock.withLock { !appendVariants.isEmpty } }
    /// `MUSE_PAGED_ONEPASS_PROBE_FAIL=1`: the probe asks for a threadgroup no
    /// pipeline holds, to exercise the two-pass fallback (a self-test).
    private static let forceProbeFailure = environment["MUSE_PAGED_ONEPASS_PROBE_FAIL"] == "1"
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var appendVariants: Set<AppendKey> = []
    nonisolated(unsafe) private static var checked = 0
    nonisolated(unsafe) private static var differing = 0
    nonisolated(unsafe) private static var appendedChecks = 0

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
                if appendEnabled {
                    outputs += warmAppend(
                        q: q, storage: appendStorage, bindings: bindings, seqinfo: seqinfo,
                        params: params, previous: previous, valueOffsets: MLXArray(offsets),
                        records: MLXArray(record), pending: pending, dtype: dtype,
                        headDim: headDim, kvHeads: kvHeads, gqa: gqa, pageSize: pageSize,
                        hpt: hpt, simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                        source: source, maxPart: mp)
                }
            }
        }
    }

    /// Probes and compiles the write-owning (linear-only) variant at one
    /// MAXPART on scratch pages; a variant that fails its probe is not used.
    private static func warmAppend(
        q: MLXArray, storage: MLXArray, bindings: Int, seqinfo: MLXArray, params: MLXArray,
        previous: MLXArray, valueOffsets: MLXArray, records: MLXArray, pending: MLXArray,
        dtype: DType, headDim: Int, kvHeads: Int, gqa: Int, pageSize: Int, hpt: Int,
        simdgroups: Int, partitionTokens: Int, softcap: Bool, source: String, maxPart: Int
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
                    pending: (pending, pending))
                eval(probed.fence, probed.out)
                return true
            }) ?? false
        if logs {
            FileHandle.standardError.write(
                Data(
                    "[onepass] append probe bindings=\(bindings) maxpart=\(maxPart) held=\(held)\n"
                        .utf8))
        }
        guard held else { return [] }
        let result = segments(
            queries: q, backing: backing, seqinfo: seqinfo, params: params,
            previous: previous, valueOffsets: valueOffsets, records: records, dtype: dtype,
            kvHeads: kvHeads, pageSize: pageSize, hpt: hpt, simdgroups: simdgroups,
            partitionTokens: partitionTokens, softcap: softcap, source: source,
            maxPart: maxPart, forceMaxPart: true, pending: (pending, pending))
        let key = AppendKey(
            dtype: dtype, headDim: headDim, kvHeads: kvHeads, gqa: gqa, pageSize: pageSize,
            hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
            softcap: softcap, maxPart: maxPart, bindings: bindings)
        lock.withLock { _ = appendVariants.insert(key) }
        return [result.fence, result.out]
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
    private static func segmentBody(bindings: Int, appending: Bool = false) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        // The read-only source stays byte-identical; only the append variant adds arguments.
        let append = appending ? ", true" : ""
        let pending = appending ? ", knew, vnew" : ""
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
                cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)>, BLOCK, DYN_TREE\(append)>(
                q, cache, seqinfo, seqinfo, params, KVH, 0, q_smem, red_smem, part_smem, out,
                threadgroup_position_in_grid, thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, tree_meta\(pending));
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
        let append: Bool
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
    }

    /// The MAXPART `segments` compiles for a call (one rule for launch and lookup).
    private static func resolvedMaxPart(_ maxPart: Int?, probe: Bool, forceMaxPart: Bool) -> Int {
        let allowDynPart =
            dynamicMaxPartitions
            && (forceMaxPart || !CBv2DraftRegime.lowOnlyVerifySpeedups || CBv2DraftRegime.lowAcceptance)
        return ((!probe || forceMaxPart) && allowDynPart)
            ? min(maxPartitions, max(1, maxPart ?? maxPartitions))
            : maxPartitions
    }

    /// Whether `warm` qualified the write-owning variant a live `segments` call
    /// with these arguments would launch.
    static func supportsAppend(
        dtype: DType, headDim: Int, kvHeads: Int, gqa: Int, pageSize: Int, hpt: Int,
        simdgroups: Int, partitionTokens: Int, softcap: Bool, maxPart: Int?, bindings: Int
    ) -> Bool {
        guard appendEnabled else { return false }
        let key = AppendKey(
            dtype: dtype, headDim: headDim, kvHeads: kvHeads, gqa: gqa, pageSize: pageSize,
            hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens, softcap: softcap,
            maxPart: resolvedMaxPart(maxPart, probe: false, forceMaxPart: false),
            bindings: bindings)
        return lock.withLock { appendVariants.contains(key) }
    }

    nonisolated(unsafe) private static var segLaunchCache: [SegLaunchKey: MLXFast.MLXFastKernel.Launch] = [:]

    /// Segmented entry for a single-bucket plan. Returns the next fence and
    /// the attention output.
    static func segments(
        queries: MLXArray, backing: some Collection<MLXArray>, seqinfo: MLXArray, params: MLXArray,
        previous: MLXArray, valueOffsets: MLXArray, records: MLXArray, dtype: DType,
        kvHeads: Int, pageSize: Int, hpt: Int, simdgroups: Int, partitionTokens: Int,
        softcap: Bool, source: String, probe: Bool = false, treeMeta: MLXArray? = nil,
        maxPart: Int? = nil, forceMaxPart: Bool = false, bindingClass: Int? = nil,
        pending: (keys: MLXArray, values: MLXArray)? = nil
    ) -> (fence: MLXArray, out: MLXArray) {
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let gqa = qh / kvHeads
        let backingCount = backing.count
        let bindings = bindingClass ?? backingCount
        let mp = resolvedMaxPart(maxPart, probe: probe, forceMaxPart: forceMaxPart)
        let dynTree = treeMeta != nil
        let appending = pending != nil
        precondition(!(appending && dynTree), "the fused append serves linear rows only")
        var inputs: [MLXArray] = []
        inputs.reserveCapacity(9 + bindings)
        inputs.append(queries)
        inputs.append(contentsOf: backing)
        if bindings > backingCount {
            let firstBinding = inputs[1]
            inputs.append(contentsOf: repeatElement(firstBinding, count: bindings - backingCount))
        }
        inputs.append(seqinfo)
        inputs.append(params)
        inputs.append(previous)
        inputs.append(valueOffsets)
        inputs.append(records)
        inputs.append(treeMeta ?? dummyTreeMeta)
        if let pending {
            inputs.append(pending.keys)
            inputs.append(pending.values)
        }
        let launchKey = probe ? nil : SegLaunchKey(
            dtype: dtype, b: b, qh: qh, d: d, kvHeads: kvHeads, pageSize: pageSize,
            hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
            maxPart: mp, softcap: softcap, bindings: bindings, dynTree: dynTree,
            append: appending)
        if let launchKey, let hit = lock.withLock({ segLaunchCache[launchKey] }) {
            let outputs = hit(inputs)
            return (outputs[0], outputs[1])
        }
        let kernelName =
            name(
                "segments\(bindings)_kvh\(kvHeads)", dtype: dtype, headDim: d, pageSize: pageSize,
                gqa: gqa, hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
                softcap: softcap, maxPart: mp) + (appending ? "_append" : "")
        let kernel = lock.withLock { () -> MLXFast.MLXFastKernel in
            if let existing = kernels[kernelName] { return existing }
            if logs {
                FileHandle.standardError.write(Data("[onepass] new kernel \(kernelName)\n".utf8))
            }
            let segmentNames = (0 ..< bindings).map { "segment\($0)" }
            let made = MLXFast.metalKernel(
                name: kernelName,
                inputNames: ["q"] + segmentNames
                    + ["seqinfo", "params", "previous", "value_offsets", "records", "tree_meta"]
                    + (appending ? ["knew", "vnew"] : []),
                outputNames: ["fence", "out"],
                source: segmentBody(bindings: bindings, appending: appending),
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
        if let launchKey {
            let launch = kernel.prepare(
                template: template,
                grid: (kvHeads * (gqa / hpt) * tg, b, 1),
                threadGroup: (tg, 1, 1),
                outputShapes: [[1], [b, qh, d]], outputDTypes: [.int32, dtype])
            lock.withLock { segLaunchCache[launchKey] = launch }
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
