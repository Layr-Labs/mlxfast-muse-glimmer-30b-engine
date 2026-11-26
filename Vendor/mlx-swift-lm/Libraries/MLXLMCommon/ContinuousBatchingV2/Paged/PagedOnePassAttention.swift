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
    /// Kill switch: `MUSE_PAGED_ONEPASS=0` (two-pass path).
    static let enabled = environment["MUSE_PAGED_ONEPASS"] != "0"
    /// Kill switch for fused sigmoid(gate) epilogue in one-pass verify attention.
    static let gatedEnvEnabled = environment["MUSE_PAGED_ONEPASS_GATE"] != "0"
    nonisolated(unsafe) private static var gatedParityPassed: Bool?
    public static var gatedEnabled: Bool {
        guard enabled, gatedEnvEnabled, !threadgroupProbeFailed else { return false }
        return lock.withLock { gatedParityPassed ?? false }
    }
    /// Opt-in switch for shared-KV multi-row gated one-pass verify attention (default off:
    /// stock `HPT=2` single-row gated one-pass preserves optimal KV reuse across odd/even spans).
    static let sharedKVGatedEnvEnabled = environment["MUSE_PAGED_ONEPASS_SHARED_KV"] == "1"
    static let sharedKVRowsPerTile = 2
    nonisolated(unsafe) private static var sharedKVGatedParityPassed: Bool?
    public static var sharedKVGatedEnabled: Bool {
        guard gatedEnabled, sharedKVGatedEnvEnabled else { return false }
        return lock.withLock { sharedKVGatedParityPassed ?? false }
    }
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
    /// `MUSE_PAGED_ONEPASS_PROBE_FAIL=1`: the probe asks for a threadgroup no
    /// pipeline holds, to exercise the two-pass fallback (a self-test).
    private static let forceProbeFailure = environment["MUSE_PAGED_ONEPASS_PROBE_FAIL"] == "1"
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
            let result = segments(
                queries: q, backing: [MLXArray](repeating: storage, count: bindings),
                seqinfo: seqinfo, params: params, previous: previous,
                valueOffsets: MLXArray(offsets), records: MLXArray(record), dtype: dtype,
                kvHeads: kvHeads, pageSize: pageSize, hpt: hpt, simdgroups: nsg,
                partitionTokens: ptok, softcap: softcap, source: source)
            outputs += [result.fence, result.out]
        }
        eval(outputs)
        if gatedEnvEnabled, dtype == .bfloat16, let firstBindings = PagedSegmentDispatchPlan.bindingClasses.first {
            // One-time 0-ULP parity check of the gated one-pass kernel on
            // non-zero inputs spanning negative, zero, and positive gate values.
            // Guarded with `withError` so any Metal JIT/runtime error cleanly
            // falls back to stock without aborting the engine.
            let passed: Bool = (try? withError { () -> Bool in
                let qVals = (0 ..< (queryHeads * headDim)).map { idx -> Float in
                    Float(((idx * 5) % 29) - 14) * 0.125
                }
                let gVals = (0 ..< (queryHeads * headDim)).map { idx -> Float in
                    Float(((idx * 7) % 33) - 16) * 0.25
                }
                let kvVals = (0 ..< (2 * pageElements)).map { idx -> Float in
                    Float(((idx * 3) % 23) - 11) * 0.125
                }
                let qTest = MLXArray(qVals, [1, queryHeads, headDim]).asType(dtype)
                let gTest = MLXArray(gVals, [1, queryHeads, headDim]).asType(dtype)
                let sTest = MLXArray(kvVals).asType(dtype)
                let offsets = [Int64](repeating: Int64(pageElements), count: max(8, firstBindings))
                let backing = [MLXArray](repeating: sTest, count: firstBindings)
                let ungated = segments(
                    queries: qTest, backing: backing, seqinfo: seqinfo, params: params,
                    previous: previous, valueOffsets: MLXArray(offsets), records: MLXArray(record),
                    dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt, simdgroups: nsg,
                    partitionTokens: ptok, softcap: softcap, source: source)
                let gated = segmentsGated(
                    queries: qTest, gate: gTest, backing: backing, seqinfo: seqinfo, params: params,
                    previous: previous, valueOffsets: MLXArray(offsets), records: MLXArray(record),
                    dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt, simdgroups: nsg,
                    partitionTokens: ptok, softcap: softcap, source: source)
                let ref = ungated.out * sigmoid(gTest)
                let differing = (
                    contiguous(gated.out).view(dtype: .uint16)
                        .!= contiguous(ref).view(dtype: .uint16)
                ).asType(.int32).sum().item(Int.self)
                guard differing == 0 else { return false }
                var gatedOutputs = [MLXArray]()
                let zeroGate = MLXArray.zeros([1, queryHeads, headDim], dtype: dtype)
                for bindings in PagedSegmentDispatchPlan.bindingClasses {
                    let bOffsets = [Int64](repeating: Int64(pageElements), count: max(8, bindings))
                    let r = segmentsGated(
                        queries: q, gate: zeroGate,
                        backing: [MLXArray](repeating: storage, count: bindings),
                        seqinfo: seqinfo, params: params, previous: previous,
                        valueOffsets: MLXArray(bOffsets), records: MLXArray(record), dtype: dtype,
                        kvHeads: kvHeads, pageSize: pageSize, hpt: hpt, simdgroups: nsg,
                        partitionTokens: ptok, softcap: softcap, source: source)
                    gatedOutputs += [r.fence, r.out]
                }
                eval(gatedOutputs)
                return true
            }) ?? false
            lock.withLock { gatedParityPassed = passed }
            if passed, sharedKVGatedEnvEnabled,
                let sharedHpt = eligibleSharedKV(
                    headDim: headDim, gqa: gqa, simdgroups: nsg, partitions: 2, rows: 5,
                    rpt: sharedKVRowsPerTile)
            {
                let allSharedPassed: Bool = (try? withError { () -> Bool in
                    // 0-ULP parity check on 5 and 4 causal rows spanning both a
                    // 16-token page boundary and a 256-token partition boundary
                    // (attend_len = 254..258, crossing parts 1 -> 2 and testing
                    // both odd tail-1 and full 2-row tiles).
                    let pagesInStorage = 2
                    let twoPageVals = (0 ..< (pagesInStorage * 2 * pageElements)).map { idx -> Float in
                        Float(((idx * 5) % 29) - 14) * 0.125
                    }
                    let twoPageStorage = MLXArray(twoPageVals).asType(dtype)
                    let twoPageOffsets = [Int64](
                        repeating: Int64(pagesInStorage * pageElements),
                        count: max(8, firstBindings))
                    let sharedBacking = [MLXArray](repeating: twoPageStorage, count: firstBindings)
                    let pagesPerPart = ptok / pageSize
                    func makePartRecord(row: Int, part: Int, pageCount: Int, maxParts: Int) -> [Int32] {
                        var rec = [Int32](repeating: 0, count: PagedSegmentDispatchPlan.recordStride)
                        rec[0] = Int32(row)
                        rec[1] = Int32(part)
                        rec[2] = Int32(part * pagesPerPart)
                        rec[3] = Int32(pageCount)
                        rec[6] = Int32(maxParts)
                        for p in 0 ..< pageCount {
                            rec[PagedSegmentDispatchPlan.recordHeader + 2 * p] = 0
                            rec[PagedSegmentDispatchPlan.recordHeader + 2 * p + 1] = Int32(p & 1)
                        }
                        return rec
                    }
                    for testRows in [5, 4] {
                        let baseLen = ptok - 2
                        let lens = (0 ..< testRows).map { baseLen + $0 }
                        var seqData = [Int32]()
                        seqData.reserveCapacity(testRows * 8)
                        var recData = [Int32]()
                        let maxP = lens.map { ($0 + ptok - 1) / ptok }.max() ?? 1
                        for (rIdx, len) in lens.enumerated() {
                            let numPages = (len + pageSize - 1) / pageSize
                            seqData += [0, Int32(len), Int32(numPages), 0, 0, 0, 0, 0]
                            let parts = (len + ptok - 1) / ptok
                            for p in 0 ..< parts {
                                let remPages = min(pagesPerPart, numPages - p * pagesPerPart)
                                recData += makePartRecord(
                                    row: rIdx, part: p, pageCount: remPages, maxParts: maxP)
                            }
                        }
                        let qMultiVals = (0 ..< (testRows * queryHeads * headDim)).map { idx -> Float in
                            Float(((idx * 7 + 3) % 31) - 15) * 0.125
                        }
                        let gMultiVals = (0 ..< (testRows * queryHeads * headDim)).map { idx -> Float in
                            Float(((idx * 11 + 5) % 37) - 18) * 0.25
                        }
                        let qMulti = MLXArray(qMultiVals, [testRows, queryHeads, headDim]).asType(dtype)
                        let gMulti = MLXArray(gMultiVals, [testRows, queryHeads, headDim]).asType(dtype)
                        let seqMulti = MLXArray(seqData, [testRows, 8])
                        let recMulti = MLXArray(recData)
                        let refSingle = segmentsGated(
                            queries: qMulti, gate: gMulti, backing: sharedBacking,
                            seqinfo: seqMulti, params: params, previous: previous,
                            valueOffsets: MLXArray(twoPageOffsets), records: recMulti,
                            dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: hpt,
                            simdgroups: nsg, partitionTokens: ptok, softcap: softcap, source: source)
                        let candShared = segmentsSharedKVGated(
                            queries: qMulti, gate: gMulti, backing: sharedBacking,
                            seqinfo: seqMulti, params: params, previous: previous,
                            valueOffsets: MLXArray(twoPageOffsets), records: recMulti,
                            dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: sharedHpt,
                            rpt: sharedKVRowsPerTile, simdgroups: nsg, partitionTokens: ptok,
                            softcap: softcap, source: source)
                        let diff = (
                            contiguous(candShared.out).view(dtype: .uint16)
                                .!= contiguous(refSingle.out).view(dtype: .uint16)
                        ).asType(.int32).sum().item(Int.self)
                        guard diff == 0 else { return false }
                    }
                    let zeroGate = MLXArray.zeros([1, queryHeads, headDim], dtype: dtype)
                    var warmShared = [MLXArray]()
                    for bindings in PagedSegmentDispatchPlan.bindingClasses {
                        let bOffsets = [Int64](
                            repeating: Int64(pageElements), count: max(8, bindings))
                        let r = segmentsSharedKVGated(
                            queries: q, gate: zeroGate,
                            backing: [MLXArray](repeating: storage, count: bindings),
                            seqinfo: seqinfo, params: params, previous: previous,
                            valueOffsets: MLXArray(bOffsets), records: MLXArray(record),
                            dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: sharedHpt,
                            rpt: sharedKVRowsPerTile, simdgroups: nsg, partitionTokens: ptok,
                            softcap: softcap, source: source)
                        warmShared += [r.fence, r.out]
                    }
                    eval(warmShared)
                    return true
                }) ?? false
                lock.withLock { sharedKVGatedParityPassed = allSharedPassed }
            }
        }
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
        headDim: Int, gqa: Int, simdgroups: Int, partitions: Int, rows: Int
    ) -> Int? {
        guard !threadgroupProbeFailed, partitions >= 1, partitions <= maxPartitions,
            rows <= 64, 32 * simdgroups * maxPartitions <= 1024
        else { return nil }
        return headsPerThreadgroup(headDim: headDim, gqa: gqa, simdgroups: simdgroups)
    }

    static func eligibleSharedKV(
        headDim: Int, gqa: Int, simdgroups: Int, partitions: Int, rows: Int, rpt: Int
    ) -> Int? {
        guard !threadgroupProbeFailed, partitions >= 1, partitions <= maxPartitions,
            rows >= 2, rows <= 64, (2 ... 8).contains(rpt),
            32 * simdgroups * maxPartitions <= 1024
        else { return nil }
        let hpt = 1
        guard gqa % hpt == 0 else { return nil }
        let floats =
            rpt * hpt * headDim + maxPartitions * simdgroups * hpt * (headDim + 2)
            + maxPartitions * hpt * (headDim + 2)
        guard floats * 4 <= PagedAttentionKernel.threadgroupMemoryLimit else { return nil }
        return hpt
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
                cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)>, BLOCK>(
                q, cache, seqinfo, seqinfo, params, KVH, 0, q_smem, red_smem, part_smem, out,
                threadgroup_position_in_grid, thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            """
    }

    private static func name(
        _ kind: String, dtype: DType, headDim: Int, pageSize: Int, gqa: Int, hpt: Int,
        simdgroups: Int, partitionTokens: Int, softcap: Bool
    ) -> String {
        let d: String
        switch dtype {
        case .float16: d = "f16"
        case .float32: d = "f32"
        case .bfloat16: d = "bf16"
        default: d = "dt\(dtype)"
        }
        return "cbv2_paged_onepass_\(kind)_\(d)_d\(headDim)_s\(pageSize)_g\(gqa)_h\(hpt)"
            + "_n\(simdgroups)_t\(partitionTokens)_p\(maxPartitions)_cap\(softcap ? 1 : 0)"
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

    /// Segmented entry for a single-bucket plan. Returns the next fence and
    /// the attention output.
    static func segments(
        queries: MLXArray, backing: [MLXArray], seqinfo: MLXArray, params: MLXArray,
        previous: MLXArray, valueOffsets: MLXArray, records: MLXArray, dtype: DType,
        kvHeads: Int, pageSize: Int, hpt: Int, simdgroups: Int, partitionTokens: Int,
        softcap: Bool, source: String, probe: Bool = false
    ) -> (fence: MLXArray, out: MLXArray) {
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let gqa = qh / kvHeads
        let bindings = backing.count
        let kernelName =
            name(
                "segments\(bindings)_kvh\(kvHeads)", dtype: dtype, headDim: d, pageSize: pageSize,
                gqa: gqa, hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
                softcap: softcap)
        let kernel = lock.withLock { () -> MLXFast.MLXFastKernel in
            if let existing = kernels[kernelName] { return existing }
            if logs {
                FileHandle.standardError.write(Data("[onepass] new kernel \(kernelName)\n".utf8))
            }
            let made = MLXFast.metalKernel(
                name: kernelName,
                inputNames: ["q"] + (0 ..< bindings).map { "segment\($0)" }
                    + ["seqinfo", "params", "previous", "value_offsets", "records"],
                outputNames: ["fence", "out"], source: segmentBody(bindings: bindings),
                header: source, ensureRowContiguous: true)
            kernels[kernelName] = made
            return made
        }
        let tg = 32 * simdgroups * maxPartitions
        let outputs = kernel(
            [queries] + backing + [seqinfo, params, previous, valueOffsets, records],
            template: [
                ("T", dtype), ("D", d), ("S", pageSize), ("KVH", kvHeads), ("GQA", gqa),
                ("HPT", hpt), ("NSG", simdgroups), ("PTOK", partitionTokens),
                ("MAXPART", maxPartitions), ("HAS_SOFTCAP", softcap), ("SEGMENTS", bindings),
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

    private static func segmentGatedBody(bindings: Int) -> String {
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
                cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)>, BLOCK, true>(
                q, cache, seqinfo, seqinfo, params, KVH, 0, q_smem, red_smem, part_smem, out,
                threadgroup_position_in_grid, thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, gate);
            """
    }

    /// Segmented entry with fused `output * sigmoid(gate)` epilogue.
    static func segmentsGated(
        queries: MLXArray, gate: MLXArray, backing: [MLXArray], seqinfo: MLXArray, params: MLXArray,
        previous: MLXArray, valueOffsets: MLXArray, records: MLXArray, dtype: DType,
        kvHeads: Int, pageSize: Int, hpt: Int, simdgroups: Int, partitionTokens: Int,
        softcap: Bool, source: String
    ) -> (fence: MLXArray, out: MLXArray) {
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let gqa = qh / kvHeads
        let bindings = backing.count
        let kernelName =
            name(
                "segments_gated\(bindings)_kvh\(kvHeads)", dtype: dtype, headDim: d,
                pageSize: pageSize, gqa: gqa, hpt: hpt, simdgroups: simdgroups,
                partitionTokens: partitionTokens, softcap: softcap)
        let kernel = lock.withLock { () -> MLXFast.MLXFastKernel in
            if let existing = kernels[kernelName] { return existing }
            let made = MLXFast.metalKernel(
                name: kernelName,
                inputNames: ["q", "gate"] + (0 ..< bindings).map { "segment\($0)" }
                    + ["seqinfo", "params", "previous", "value_offsets", "records"],
                outputNames: ["fence", "out"], source: segmentGatedBody(bindings: bindings),
                header: source, ensureRowContiguous: true)
            kernels[kernelName] = made
            return made
        }
        let tg = 32 * simdgroups * maxPartitions
        let outputs = kernel(
            [queries, gate] + backing + [seqinfo, params, previous, valueOffsets, records],
            template: [
                ("T", dtype), ("D", d), ("S", pageSize), ("KVH", kvHeads), ("GQA", gqa),
                ("HPT", hpt), ("NSG", simdgroups), ("PTOK", partitionTokens),
                ("MAXPART", maxPartitions), ("HAS_SOFTCAP", softcap), ("SEGMENTS", bindings),
                ("STRIDE", PagedSegmentDispatchPlan.recordStride), ("BLOCK", block),
            ],
            grid: (kvHeads * (gqa / hpt) * tg, b, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[1], [b, qh, d]], outputDTypes: [.int32, dtype])
        return (outputs[0], outputs[1])
    }

    private static func segmentSharedKVGatedBody(bindings: Int) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        return """
            const int total_rows = q_shape[0];
            const int group = threadgroup_position_in_grid.y;
            const int row0 = group * RPT;
            const int row_last = min(total_rows - 1, row0 + RPT - 1);
            int first = 0;
            for (int r = 0; r < row_last; r++) {
                first += (seqinfo[r * 8 + 1] + PTOK - 1) / PTOK;
            }
            const int own = (seqinfo[row_last * 8 + 1] + PTOK - 1) / PTOK;
            const int wanted = (int)simdgroup_index_in_threadgroup / NSG;
            const device int32_t* record =
                records + (first + min(wanted, own - 1)) * STRIDE;
            const cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)> cache{
                {\(pointers)}, value_offsets, record, (size_t)KVH * S * D};
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
                fence[0] = previous[0] + 1;
            }
            threadgroup float q_smem[RPT * HPT * D];
            threadgroup float red_smem[MAXPART * NSG * HPT * (D + 2)];
            threadgroup float part_smem[MAXPART * HPT * (D + 2)];
            cbv2::paged_attention_onepass_shared_kv_impl<
                T, D, S, GQA, HPT, NSG, PTOK, MAXPART, HAS_SOFTCAP,
                cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)>, BLOCK, true, RPT>(
                q, cache, seqinfo, seqinfo, params, KVH, 0, total_rows,
                q_smem, red_smem, part_smem, out,
                threadgroup_position_in_grid, thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup, gate);
            """
    }

    /// Segmented entry sharing KV loads across up to `rpt` causal verify rows
    /// per threadgroup with fused `output * sigmoid(gate)` epilogue.
    static func segmentsSharedKVGated(
        queries: MLXArray, gate: MLXArray, backing: [MLXArray], seqinfo: MLXArray, params: MLXArray,
        previous: MLXArray, valueOffsets: MLXArray, records: MLXArray, dtype: DType,
        kvHeads: Int, pageSize: Int, hpt: Int, rpt: Int, simdgroups: Int, partitionTokens: Int,
        softcap: Bool, source: String
    ) -> (fence: MLXArray, out: MLXArray) {
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let gqa = qh / kvHeads
        let bindings = backing.count
        let kernelName =
            name(
                "segments_sharedkv_r\(rpt)_gated\(bindings)_kvh\(kvHeads)", dtype: dtype,
                headDim: d, pageSize: pageSize, gqa: gqa, hpt: hpt, simdgroups: simdgroups,
                partitionTokens: partitionTokens, softcap: softcap)
        let kernel = lock.withLock { () -> MLXFast.MLXFastKernel in
            if let existing = kernels[kernelName] { return existing }
            let made = MLXFast.metalKernel(
                name: kernelName,
                inputNames: ["q", "gate"] + (0 ..< bindings).map { "segment\($0)" }
                    + ["seqinfo", "params", "previous", "value_offsets", "records"],
                outputNames: ["fence", "out"],
                source: segmentSharedKVGatedBody(bindings: bindings),
                header: source, ensureRowContiguous: true)
            kernels[kernelName] = made
            return made
        }
        let tg = 32 * simdgroups * maxPartitions
        let groups = (b + rpt - 1) / rpt
        let outputs = kernel(
            [queries, gate] + backing + [seqinfo, params, previous, valueOffsets, records],
            template: [
                ("T", dtype), ("D", d), ("S", pageSize), ("KVH", kvHeads), ("GQA", gqa),
                ("HPT", hpt), ("NSG", simdgroups), ("PTOK", partitionTokens),
                ("MAXPART", maxPartitions), ("HAS_SOFTCAP", softcap), ("SEGMENTS", bindings),
                ("STRIDE", PagedSegmentDispatchPlan.recordStride), ("BLOCK", block),
                ("RPT", rpt),
            ],
            grid: (kvHeads * (gqa / hpt) * tg, groups, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[1], [b, qh, d]], outputDTypes: [.int32, dtype])
        return (outputs[0], outputs[1])
    }
}
