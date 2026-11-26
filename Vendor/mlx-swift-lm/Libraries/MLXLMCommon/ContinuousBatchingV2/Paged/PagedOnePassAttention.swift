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
    nonisolated(unsafe) private static var treeProbeFailed = false
    nonisolated(unsafe) private static var treeWarmedClasses = 0
    nonisolated(unsafe) private static var deviceHolesWarmedClasses = 0
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
            if CBv2VerifyTree.enabled,
                let treeHpt = eligible(
                    headDim: headDim, gqa: gqa, simdgroups: nsg, partitions: 1, rows: 1,
                    holes: true)
            {
                // The draft-tree variants (branch rows skip a range): holes
                // from seqinfo, and for GPU-built trees from a separate
                // seed-relative `[rows, 2]` buffer. They hold more registers
                // and threadgroup memory than the chain's, so each gets the
                // same probe: a pipeline that cannot hold the threadgroup
                // would make MLX re-resolve a constrained one on every dispatch.
                let deviceHoles = MLXArray.zeros([4, 2], type: Int32.self)
                var variants: [MLXArray?] = [nil]
                if CBv2VerifyTree.zeroSync { variants.append(deviceHoles) }
                var treeHeld = true
                for variant in variants {
                    let held =
                        (try? withError { () -> Bool in
                            let probed = segments(
                                queries: q,
                                backing: [MLXArray](repeating: storage, count: bindings),
                                seqinfo: seqinfo, params: params, previous: previous,
                                valueOffsets: MLXArray(offsets), records: MLXArray(record),
                                dtype: dtype, kvHeads: kvHeads, pageSize: pageSize, hpt: treeHpt,
                                simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                                source: source, probe: true, holes: true, deviceHoles: variant)
                            eval(probed.fence, probed.out)
                            return true
                        }) ?? false
                    if logs {
                        FileHandle.standardError.write(
                            Data(
                                "[onepass] tree threadgroup probe bindings=\(bindings) device=\(variant != nil) held=\(held)\n"
                                    .utf8))
                    }
                    if variant == nil { treeHeld = held }
                    guard held else { break }
                    let tree = segments(
                        queries: q, backing: [MLXArray](repeating: storage, count: bindings),
                        seqinfo: seqinfo, params: params, previous: previous,
                        valueOffsets: MLXArray(offsets), records: MLXArray(record), dtype: dtype,
                        kvHeads: kvHeads, pageSize: pageSize, hpt: treeHpt, simdgroups: nsg,
                        partitionTokens: ptok, softcap: softcap, source: source, holes: true,
                        deviceHoles: variant)
                    outputs += [tree.fence, tree.out]
                    if variant != nil { deviceHolesWarmedClasses += 1 }
                }
                guard treeHeld else {
                    treeProbeFailed = true
                    CBv2VerifyTree.disable()
                    continue
                }
                treeWarmedClasses += 1
            } else if CBv2VerifyTree.enabled {
                CBv2VerifyTree.disable()
            }
        }
        // A tree window attends through the one-pass kernel only (the
        // two-pass kernels have no branch variant): ready when every binding
        // class compiled and held its threadgroup, for contexts the one-pass
        // kernel covers.
        if CBv2VerifyTree.enabled, !treeProbeFailed,
            treeWarmedClasses >= PagedSegmentDispatchPlan.bindingClasses.count
        {
            CBv2VerifyTree.setContextLimit(maxPartitions * ptok)
            // GPU-built trees also need the device-hole variant of every class.
            CBv2VerifyTree.setZeroSyncReady(
                deviceHolesWarmedClasses >= PagedSegmentDispatchPlan.bindingClasses.count)
        } else {
            CBv2VerifyTree.disable()
        }
        eval(outputs)
    }

    /// Heads per threadgroup: the largest divisor of `gqa` at or under the
    /// requested count whose threadgroup memory fits Metal's limit.
    /// The draft-tree hole variant also holds a K/V copy of its leaf row
    /// (`2 * headDim` floats); no other variant reserves it.
    static func headsPerThreadgroup(
        headDim: Int, gqa: Int, simdgroups: Int, holes: Bool = false
    ) -> Int? {
        var hpt = min(gqa, requestedHeads)
        while hpt >= 1 {
            let floats =
                hpt * headDim + maxPartitions * simdgroups * hpt * (headDim + 2)
                + maxPartitions * hpt * (headDim + 2) + (holes ? 2 * headDim : 0)
            if gqa % hpt == 0, floats * 4 <= PagedAttentionKernel.threadgroupMemoryLimit {
                return hpt
            }
            hpt -= 1
        }
        return nil
    }

    static func eligible(
        headDim: Int, gqa: Int, simdgroups: Int, partitions: Int, rows: Int, holes: Bool = false
    ) -> Int? {
        guard !threadgroupProbeFailed, partitions >= 1, partitions <= maxPartitions,
            rows <= 64, 32 * simdgroups * maxPartitions <= 1024
        else { return nil }
        return headsPerThreadgroup(
            headDim: headDim, gqa: gqa, simdgroups: simdgroups, holes: holes)
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
            q, cache, tables, seqinfo, params, kvh, maxp, q_smem, red_smem, part_smem,
            (threadgroup float*)nullptr, out,
            threadgroup_position_in_grid, thread_position_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """

    /// Records of a single-bucket plan are row-major with each row's own
    /// partition count, so (row, partition) is a prefix sum away.
    private static func segmentBody(bindings: Int, deviceHoles: Bool = false) -> String {
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
            #ifdef CBV2_TREE_HOLES
            threadgroup float leaf_smem[2 * D];
            threadgroup float* leaf = leaf_smem;
            #else
            threadgroup float* leaf = nullptr;
            #endif
            cbv2::paged_attention_onepass_impl<
                T, D, S, GQA, HPT, NSG, PTOK, MAXPART, HAS_SOFTCAP,
                cbv2::PagedSegmentAccessor<T, SEGMENTS, (BLOCK > 1)>, BLOCK>(
                q, cache, seqinfo, seqinfo, params, KVH, 0, q_smem, red_smem, part_smem, leaf, out,
                threadgroup_position_in_grid, thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup\(deviceHoles ? ", hole_rel" : ""));
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
        CBv2KernelLog.observe(
            kernelName, template: "PTOK\(partitionTokens),HPT\(hpt)",
            inputs: [queries, kSlab, vSlab, tables, seqinfo, params, writeFence],
            outputs: [dtype])
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
        softcap: Bool, source: String, probe: Bool = false, holes: Bool = false,
        deviceHoles: MLXArray? = nil
    ) -> (fence: MLXArray, out: MLXArray) {
        let (b, qh, d) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let gqa = qh / kvHeads
        let bindings = backing.count
        // GPU-built tree: holes from `deviceHoles` (seed-relative, >= 8 int32).
        let holeRel = holes ? deviceHoles : nil
        let extra = holeRel.map { [$0] } ?? []
        let kernelName =
            name(
                "segments\(bindings)_kvh\(kvHeads)", dtype: dtype, headDim: d, pageSize: pageSize,
                gqa: gqa, hpt: hpt, simdgroups: simdgroups, partitionTokens: partitionTokens,
                softcap: softcap) + (holes ? (holeRel != nil ? "_dholes" : "_holes") : "")
        let kernel = lock.withLock { () -> MLXFast.MLXFastKernel in
            if let existing = kernels[kernelName] { return existing }
            if logs {
                FileHandle.standardError.write(Data("[onepass] new kernel \(kernelName)\n".utf8))
            }
            let made = MLXFast.metalKernel(
                name: kernelName,
                inputNames: ["q"] + (0 ..< bindings).map { "segment\($0)" }
                    + ["seqinfo", "params", "previous", "value_offsets", "records"]
                    + (holeRel != nil ? ["hole_rel"] : []),
                outputNames: ["fence", "out"],
                source: segmentBody(bindings: bindings, deviceHoles: holeRel != nil),
                header: (holes ? "#define CBV2_TREE_HOLES 1\n" : "")
                    + (holeRel != nil ? "#define CBV2_TREE_DEVICE_HOLES 1\n" : "") + source,
                ensureRowContiguous: true)
            kernels[kernelName] = made
            return made
        }
        let tg = 32 * simdgroups * maxPartitions
        CBv2KernelLog.observe(
            kernelName, template: "PTOK\(partitionTokens),HPT\(hpt),probe\(probe)",
            inputs: [queries] + backing + [seqinfo, params, previous, valueOffsets, records] + extra,
            outputs: [.int32, dtype])
        let outputs = kernel(
            [queries] + backing + [seqinfo, params, previous, valueOffsets, records] + extra,
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
}
