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
    private static let requestedHeads: Int = {
        if let raw = environment["MUSE_PAGED_ONEPASS_HPT"], let value = Int(raw), value >= 1 {
            return value
        }
        // Two heads share each K/V row load: at 8 rows the one-head form is
        // throughput-bound (measured 43 vs 31 us per layer), at 1 row equal.
        return 2
    }()
    /// `MUSE_PAGED_ONEPASS_CHECK=1`: every one-pass dispatch also runs the
    /// two-pass kernels and compares raw bits (slow; a self-test, not a mode).
    static let checks = environment["MUSE_PAGED_ONEPASS_CHECK"] == "1"
    private static let logs = environment["MUSE_PAGED_ONEPASS_LOG"] == "1"
    /// Set by the load-time probe in `warm` when a one-pass pipeline cannot
    /// hold the kernel's threadgroup directly. MLX would then re-resolve a
    /// constrained pipeline on every dispatch (host string work per layer);
    /// the two-pass kernels are used instead. Same bytes either way.
    nonisolated(unsafe) private(set) static var threadgroupProbeFailed = false
    static var appendWarmed: Bool { lock.withLock { !appendVariants.isEmpty } }
    /// `MUSE_PAGED_ONEPASS_PROBE_FAIL=1`: the probe asks for a threadgroup no
    /// pipeline holds, to exercise the two-pass fallback (a self-test).
    private static let forceProbeFailure = environment["MUSE_PAGED_ONEPASS_PROBE_FAIL"] == "1"
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var appendVariants: Set<String> = []
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
            eligible(
                headDim: headDim, gqa: gqa, simdgroups: nsg, partitions: 1, rows: 1) != nil
        else { return }
        let ptok = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokens, pageSize: pageSize)
        let pageElements = kvHeads * pageSize * headDim
        let storage = contiguous(MLXArray.zeros([2 * pageElements], dtype: dtype))
        let q = MLXArray.zeros([1, queryHeads, headDim], dtype: dtype)
        let seqinfo = MLXArray([Int32(0), 1, 1, 0, 0, 0, 0, 0], [1, 8])
        let params = MLXArray([Float(1), 1, 0, 0, 0, 0, 0, 0])
        let previous = MLXArray.zeros([1], type: Int32.self)
        let pending = MLXArray.zeros([1, kvHeads, headDim], dtype: dtype)
        var record = [Int32](repeating: 0, count: PagedSegmentDispatchPlan.recordStride)
        record[3] = 1
        record[6] = 1
        var outputs = [MLXArray]()
        for capacity in 1 ... maxPartitions {
            var warmedHeads = Set<Int>()
            for rows in 1 ... 8 {
                guard let variant = eligible(
                    headDim: headDim, gqa: gqa, simdgroups: nsg, partitions: capacity,
                    rows: rows)
                else { continue }
                let heads = variant.heads
                guard warmedHeads.insert(heads).inserted else { continue }
                for bindings in PagedSegmentDispatchPlan.bindingClasses {
                    let backing = [MLXArray](repeating: storage, count: bindings)
                    let valueOffsets = MLXArray(
                        [Int64](repeating: Int64(pageElements), count: max(8, bindings)))
                    let records = MLXArray(record)
                    if capacity == maxPartitions, rows == 1 {
                        let held =
                            (try? withError { () -> Bool in
                                let probed = segments(
                                    queries: q,
                                    backing: backing, seqinfo: seqinfo, params: params,
                                    previous: previous, valueOffsets: valueOffsets, records: records,
                                    dtype: dtype, kvHeads: kvHeads, pageSize: pageSize,
                                    hpt: heads, simdgroups: nsg, partitionTokens: ptok,
                                    softcap: softcap, source: source, probe: true,
                                    capacity: capacity)
                                eval(probed.fence, probed.out)
                                return true
                            }) ?? false
                        if logs {
                            FileHandle.standardError.write(
                                Data(
                                    "[onepass] threadgroup probe bindings=\(bindings) held=\(held)\n"
                                        .utf8))
                        }
                        guard held else {
                            threadgroupProbeFailed = true
                            return
                        }
                    }
                    let result = segments(
                        queries: q, backing: backing, seqinfo: seqinfo, params: params,
                        previous: previous, valueOffsets: valueOffsets, records: records,
                        dtype: dtype, kvHeads: kvHeads, pageSize: pageSize,
                        hpt: heads, simdgroups: nsg,
                        partitionTokens: ptok, softcap: softcap, source: source,
                        capacity: capacity)
                    outputs += [result.fence, result.out]
                    let held = (try? withError { () -> Bool in
                        let appended = segments(
                            queries: q, backing: backing, seqinfo: seqinfo, params: params,
                            previous: result.fence, valueOffsets: valueOffsets, records: records,
                            dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: heads,
                            simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                            source: source, probe: true, capacity: capacity,
                            pending: (pending, pending))
                        eval(appended.fence, appended.out)
                        return true
                    }) ?? false
                    if held {
                        let key = segmentName(
                            dtype: dtype, headDim: headDim, kvHeads: kvHeads, gqa: gqa,
                            pageSize: pageSize, hpt: heads, simdgroups: nsg,
                            partitionTokens: ptok, softcap: softcap, capacity: capacity,
                            bindings: bindings, appending: true)
                        lock.withLock { _ = appendVariants.insert(key) }
                    }
                }
            }
        }
        eval(outputs)
    }

    /// Heads per threadgroup: the largest divisor of `gqa` at or under the
    static func headsPerThreadgroup(
        headDim: Int, gqa: Int, simdgroups: Int, capacity: Int, rows: Int
    ) -> Int? {
        let wanted = rows >= 5 ? max(requestedHeads, 4) : requestedHeads
        var hpt = min(gqa, wanted)
        while hpt >= 1 {
            let floats =
                hpt * headDim + capacity * simdgroups * hpt * (headDim + 2)
                + capacity * hpt * (headDim + 2)
            if gqa % hpt == 0, floats * 4 <= PagedAttentionKernel.threadgroupMemoryLimit {
                return hpt
            }
            hpt -= 1
        }
        return nil
    }

    static func headsPerThreadgroup(headDim: Int, gqa: Int, simdgroups: Int) -> Int? {
        headsPerThreadgroup(
            headDim: headDim, gqa: gqa, simdgroups: simdgroups, capacity: maxPartitions,
            rows: 1)
    }

    static func eligible(
        headDim: Int, gqa: Int, simdgroups: Int, partitions: Int, rows: Int
    ) -> (heads: Int, capacity: Int)? {
        let capacity = min(max(1, partitions), maxPartitions)
        guard !threadgroupProbeFailed, partitions >= 1, partitions <= maxPartitions,
            rows <= 64, 32 * simdgroups * capacity <= 1024,
            let heads = headsPerThreadgroup(
                headDim: headDim, gqa: gqa, simdgroups: simdgroups, capacity: capacity,
                rows: rows)
        else { return nil }
        return (heads, capacity)
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
    private static func segmentBody(bindings: Int, appending: Bool) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        let append = appending ? "true" : "false"
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
                cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)>, BLOCK, \(append)>(
                q, cache, seqinfo, seqinfo, params, KVH, 0, q_smem, red_smem, part_smem, out,
                threadgroup_position_in_grid, thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup\(pending));
            """
    }

    private static func name(
        _ kind: String, dtype: DType, headDim: Int, pageSize: Int, gqa: Int, hpt: Int,
        simdgroups: Int, partitionTokens: Int, softcap: Bool, capacity: Int
    ) -> String {
        let d: String
        switch dtype {
        case .float16: d = "f16"
        case .float32: d = "f32"
        case .bfloat16: d = "bf16"
        default: d = "dt\(dtype)"
        }
        return "cbv2_paged_onepass_\(kind)_\(d)_d\(headDim)_s\(pageSize)_g\(gqa)_h\(hpt)"
            + "_n\(simdgroups)_t\(partitionTokens)_p\(capacity)_cap\(softcap ? 1 : 0)"
            + "_b\(block)"
    }

    private static func segmentName(
        dtype: DType, headDim: Int, kvHeads: Int, gqa: Int, pageSize: Int, hpt: Int,
        simdgroups: Int, partitionTokens: Int, softcap: Bool, capacity: Int, bindings: Int,
        appending: Bool
    ) -> String {
        name(
            "segments\(bindings)_kvh\(kvHeads)", dtype: dtype, headDim: headDim, pageSize: pageSize,
            gqa: gqa, hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
            softcap: softcap, capacity: capacity) + (appending ? "_append" : "")
    }

    static func supportsAppend(
        dtype: DType, headDim: Int, kvHeads: Int, gqa: Int, pageSize: Int, hpt: Int,
        simdgroups: Int, partitionTokens: Int, softcap: Bool, capacity: Int, bindings: Int
    ) -> Bool {
        let key = segmentName(
            dtype: dtype, headDim: headDim, kvHeads: kvHeads, gqa: gqa, pageSize: pageSize,
            hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
            softcap: softcap, capacity: capacity, bindings: bindings, appending: true)
        return lock.withLock { appendVariants.contains(key) }
    }

    /// Direct-slab entry: `[B, QH, D]` in the slab dtype.
    static func slab(
        queries: MLXArray, kSlab: MLXArray, vSlab: MLXArray, tables: MLXArray,
        seqinfo: MLXArray, params: MLXArray, softcap: Bool, pageSize: Int,
        writeFence: MLXArray, hpt: Int, simdgroups: Int, partitionTokens: Int,
        source: String, stream: StreamOrDevice, capacity: Int
    ) -> MLXArray {
        let dtype = kSlab.dtype
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let kvh = kSlab.dim(1)
        let gqa = qh / kvh
        let kernelName = name(
            "slab", dtype: dtype, headDim: d, pageSize: pageSize, gqa: gqa, hpt: hpt,
            simdgroups: simdgroups, partitionTokens: partitionTokens, softcap: softcap,
            capacity: capacity)
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
        let tg = 32 * simdgroups * capacity
        return kernel(
            [queries, kSlab, vSlab, tables, seqinfo, params, writeFence],
            template: [
                ("T", dtype), ("D", d), ("S", pageSize), ("GQA", gqa), ("HPT", hpt),
                ("NSG", simdgroups), ("PTOK", partitionTokens), ("MAXPART", capacity),
                ("HAS_SOFTCAP", softcap), ("BLOCK", block),
            ],
            grid: (kvh * (gqa / hpt) * tg, b, 1), threadGroup: (tg, 1, 1),
            outputShapes: [[b, qh, d]], outputDTypes: [dtype], stream: stream)[0]
    }

    /// Segmented entry for a single-bucket plan. Returns the next fence and
    /// the attention output.
    static func segments(
        queries: MLXArray, backing: [MLXArray], seqinfo: MLXArray, params: MLXArray,
        previous: MLXArray, valueOffsets: MLXArray, records: MLXArray, dtype: DType,
        kvHeads: Int, pageSize: Int, hpt: Int, simdgroups: Int, partitionTokens: Int,
        softcap: Bool, source: String, probe: Bool = false, capacity: Int,
        pending: (keys: MLXArray, values: MLXArray)? = nil
    ) -> (fence: MLXArray, out: MLXArray) {
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let gqa = qh / kvHeads
        let bindings = backing.count
        let appending = pending != nil
        let kernelName = segmentName(
            dtype: dtype, headDim: d, kvHeads: kvHeads, gqa: gqa, pageSize: pageSize, hpt: hpt,
            simdgroups: simdgroups, partitionTokens: partitionTokens, softcap: softcap,
            capacity: capacity, bindings: bindings, appending: appending)
        let kernel = lock.withLock { () -> MLXFast.MLXFastKernel in
            if let existing = kernels[kernelName] { return existing }
            if logs {
                FileHandle.standardError.write(Data("[onepass] new kernel \(kernelName)\n".utf8))
            }
            let segments = (0 ..< bindings).map { "segment\($0)" }
            let made = MLXFast.metalKernel(
                name: kernelName,
                inputNames: ["q"] + segments
                    + ["seqinfo", "params", "previous", "value_offsets", "records"]
                    + (appending ? ["knew", "vnew"] : []),
                outputNames: ["fence", "out"],
                source: segmentBody(bindings: bindings, appending: appending),
                header: source, ensureRowContiguous: true,
                mutableInputs: appending ? segments : [])
            kernels[kernelName] = made
            return made
        }
        let tg = 32 * simdgroups * capacity
        var inputs = [queries] + backing + [seqinfo, params, previous, valueOffsets, records]
        if let pending {
            inputs.append(pending.keys)
            inputs.append(pending.values)
        }
        let outputs = kernel(
            inputs,
            template: [
                ("T", dtype), ("D", d), ("S", pageSize), ("KVH", kvHeads), ("GQA", gqa),
                ("HPT", hpt), ("NSG", simdgroups), ("PTOK", partitionTokens),
                ("MAXPART", capacity), ("HAS_SOFTCAP", softcap), ("SEGMENTS", bindings),
                ("STRIDE", PagedSegmentDispatchPlan.recordStride), ("BLOCK", block),
            ],
            // The probe asks for the same thread count as one (tg / 2, 2, 1)
            // threadgroup: MLX accepts that shape only when the pipeline
            // holds `tg` threads as compiled, and throws otherwise.
            grid: probe
                ? (forceProbeFailure ? 2 * tg : tg / 2, 2, 1)
                : (kvHeads * (gqa / hpt) * tg, b, 1),
            threadGroup: probe ? (forceProbeFailure ? 2 * tg : tg / 2, 2, 1) : (tg, 1, 1),
            outputShapes: [[1], [b, qh, d]], outputDTypes: [.int32, dtype])
        return (outputs[0], outputs[1])
    }
}
