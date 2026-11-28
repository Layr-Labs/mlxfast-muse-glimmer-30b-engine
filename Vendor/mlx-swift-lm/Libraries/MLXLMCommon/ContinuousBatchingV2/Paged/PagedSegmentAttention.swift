import Foundation
import MLX
import MLXFast

/// Decode over any number of native segments using bounded binding buckets.
/// Only address resolution differs from the direct-slab numerical reference.
enum PagedSegmentAttention {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    /// Front caches keyed by the structured variant identity, so a decode
    /// call (one part bucket plus one merge per layer per step) never
    /// formats a kernel name string to find an already-built kernel. Each
    /// entry is the very object `kernels[name]` holds for that identity's
    /// name; a miss falls through to the name-keyed path unchanged.
    private struct PartKernelID: Hashable {
        let key: PagedAttentionKernelKey
        let bindings: Int
        let kvHeads: Int
        let allocatesScratch: Bool
    }
    nonisolated(unsafe) private static var partKernels: [PartKernelID: MLXFast.MLXFastKernel] =
        [:]
    nonisolated(unsafe) private static var mergeKernels:
        [PagedAttentionKernelKey: MLXFast.MLXFastKernel] = [:]

    /// `allocatesScratch` is the first bucket's variant: `partials` and `meta`
    /// are its uninitialized outputs instead of mutable inputs. The body and
    /// arithmetic are identical; only the buffers' origin differs.
    static func inputNames(bindings: Int, allocatesScratch: Bool = false) -> [String] {
        ["q", "knew", "vnew"] + (0 ..< bindings).map { "segment\($0)" }
            + ["seqinfo", "params", "previous", "value_offsets", "records", "tree_meta"]
            + (allocatesScratch ? [] : ["partials", "meta"])
    }

    static func body(bindings: Int) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        return """
            const device int32_t* record = records + threadgroup_position_in_grid.y * STRIDE;
            const cbv2::PagedSegmentAccessor<T, SEGMENTS> cache{
                {\(pointers)}, value_offsets, record, (size_t)KVH * S * D};
            const uint3 position(threadgroup_position_in_grid.x, record[0], record[1]);
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
                fence[0] = previous[0] + 1;
            }
            threadgroup float q_smem[HPT * D];
            threadgroup float red_smem[NSG * HPT * (D + 2)];
            cbv2::paged_attention_part_cached_impl<T, D, S, GQA, HPT, NSG, PTOK, HAS_WRITE, HAS_SOFTCAP, cbv2::PagedSegmentAccessor<T, SEGMENTS>, const device bool*, DYN_TREE>(
                q, knew, vnew, cache, seqinfo, seqinfo, params, KVH, 0, record[6],
                q_smem, red_smem, partials, meta,
                position, thread_position_in_threadgroup, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup,
                0, 0, -1, 0, 1, false, nullptr, false, nullptr, 0, 0, tree_meta);
            """
    }

    private static func kernel(
        key: PagedAttentionKernelKey, bindings: Int, kvHeads: Int,
        source: String, allocatesScratch: Bool = false
    ) -> MLXFast.MLXFastKernel {
        let id = PartKernelID(
            key: key, bindings: bindings, kvHeads: kvHeads, allocatesScratch: allocatesScratch)
        return lock.withLock {
            if let cached = partKernels[id] { return cached }
            let name =
                key.kernelName + "_segments\(bindings)_kvh\(kvHeads)"
                + (allocatesScratch ? "_scratch" : "")
            if let existing = kernels[name] {
                partKernels[id] = existing
                return existing
            }
            let made = MLXFast.metalKernel(
                name: name,
                inputNames: inputNames(bindings: bindings, allocatesScratch: allocatesScratch),
                outputNames: ["fence"] + (allocatesScratch ? ["partials", "meta"] : []),
                source: body(bindings: bindings), header: source, ensureRowContiguous: true,
                mutableInputs: (allocatesScratch ? [] : ["partials", "meta"])
                    + (key.hasWrite ? (0 ..< bindings).map { "segment\($0)" } : []))
            kernels[name] = made
            partKernels[id] = made
            return made
        }
    }

    nonisolated(unsafe) private static var unusedSinks: [Int: MLXArray] = [:]

    /// The merge kernel's `sinks` binding when HAS_SINKS is false: the
    /// shader never reads it (`if (HAS_SINKS)` guards the only load). A fresh
    /// `zeros` per call was a broadcast that `ensureRowContiguous` copied, one
    /// fill dispatch per layer per step. One materialized, never-mutated
    /// buffer per size serves every call (same discipline as
    /// `PagedAttentionKernel.zeroSinks`).
    static func unusedMergeSinks(count: Int) -> MLXArray {
        lock.withLock {
            if let existing = unusedSinks[count] { return existing }
            let made = MLXArray.zeros([count], dtype: .float32)
            eval(made)
            unusedSinks[count] = made
            return made
        }
    }

    nonisolated(unsafe) static let dummyTreeMeta: MLXArray = {
        let arr = MLXArray.zeros([18], type: Int32.self)
        eval(arr)
        return arr
    }()

    /// Warms the two-pass segmented decode kernels (part + merge, both linear and dynamic-tree)
    /// for `bindings = 1, hasWrite = false, allocatesScratch = true` so contexts exceeding
    /// `PagedOnePassAttention.maxPartitions * PTOK` (or any probe fallback) pay zero live JIT.
    static func warmTwoPass(
        dtype: DType, queryHeads: Int, kvHeads: Int, headDim: Int, pageSize: Int,
        softcap: Bool, source: String
    ) -> [MLXArray] {
        guard kvHeads > 0, queryHeads % kvHeads == 0,
            let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(
                headDim: headDim, gqa: queryHeads / kvHeads)
        else { return [] }
        let gqa = queryHeads / kvHeads
        let hpt = PagedAttentionKernel.headsPerThreadgroup(headDim: headDim, gqa: gqa)
        let splits = gqa / hpt
        let ptok = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokens, pageSize: pageSize)
        let pageElements = kvHeads * pageSize * headDim
        let storage = MLXArray.zeros([2 * pageElements], dtype: dtype)
        let q = MLXArray.zeros([1, queryHeads, headDim], dtype: dtype)
        let seqinfo = MLXArray([Int32(0), 1, 1, 0, 0, 0, 0, 0], [1, 8])
        let params = MLXArray([Float(1), 1, 0, 0, 0, 0, 0, 0])
        var fence = MLXArray.zeros([1], type: Int32.self)
        let offsets = MLXArray([Int64](repeating: Int64(pageElements), count: 8))
        var record = [Int32](repeating: 0, count: PagedSegmentDispatchPlan.recordStride)
        record[3] = 1
        record[6] = 1
        let records = MLXArray(record)
        let partKey = PagedAttentionKernelKey(
            pass: .part, dtype: dtype, headDim: headDim, pageSize: pageSize, gqa: gqa,
            simdgroups: nsg, hasSinks: false, hasSoftcap: softcap,
            partitionTokens: ptok, hasWrite: false)
        let mergeKey = PagedAttentionKernelKey(
            pass: .merge, dtype: dtype, headDim: headDim, pageSize: pageSize, gqa: gqa,
            simdgroups: 1, hasSinks: false, hasSoftcap: false, partitionTokens: ptok)
        let mergeSinks = unusedMergeSinks(count: max(8, queryHeads))
        let partialsShape = [1, queryHeads, 1, headDim]
        let metaCount = max(8, queryHeads * 2)
        let warmDynTree = CBv2DraftRegime.verifyTreeMode != 0 && CBv2DraftRegime.dynamicTreeEnabled
        var out: [MLXArray] = []
        for dynTree in (warmDynTree ? [false, true] : [false]) {
            let template: [(String, any KernelTemplateArg)] = [
                ("T", dtype), ("D", headDim), ("S", pageSize),
                ("KVH", kvHeads), ("GQA", gqa), ("HPT", hpt), ("NSG", nsg),
                ("PTOK", ptok), ("HAS_WRITE", false), ("HAS_SOFTCAP", softcap),
                ("SEGMENTS", 1), ("STRIDE", PagedSegmentDispatchPlan.recordStride),
                ("DYN_TREE", dynTree),
            ]
            let partOutputs = kernel(
                key: partKey, bindings: 1, kvHeads: kvHeads, source: source,
                allocatesScratch: true)(
                    [q, q, q, storage, seqinfo, params, fence, offsets, records, dummyTreeMeta],
                    template: template,
                    grid: (kvHeads * splits * 32 * nsg, 1, 1),
                    threadGroup: (32 * nsg, 1, 1),
                    outputShapes: [[1], partialsShape, [metaCount]],
                    outputDTypes: [.int32, .float32, .float32])
            fence = partOutputs[0]
            // Also warm `allocatesScratch: false` (used by `zeroFilledScratch` / `mergeScratch`).
            fence = kernel(
                key: partKey, bindings: 1, kvHeads: kvHeads, source: source,
                allocatesScratch: false)(
                    [
                        q, q, q, storage, seqinfo, params, fence, offsets, records, dummyTreeMeta,
                        partOutputs[1], partOutputs[2],
                    ],
                    template: template,
                    grid: (kvHeads * splits * 32 * nsg, 1, 1),
                    threadGroup: (32 * nsg, 1, 1),
                    outputShapes: [[1]],
                    outputDTypes: [.int32])[0]
            let merged = mergeKernel(key: mergeKey, source: source)(
                [partOutputs[1], partOutputs[2], seqinfo, mergeSinks, fence, dummyTreeMeta],
                template: [
                    ("T", dtype), ("D", headDim), ("PTOK", ptok), ("HAS_SINKS", false),
                    ("DYN_TREE", dynTree),
                ],
                grid: (queryHeads * 32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[1, queryHeads, headDim]], outputDTypes: [dtype])[0]
            out += [fence, merged]
        }
        return out
    }

    static let orderedMergeBody: String = """
        const int heads = partials_shape[1];
        const int maxpart = partials_shape[2];
        cbv2::paged_attention_merge_impl<T, D, PTOK, HAS_SINKS, DYN_TREE>(
            partials, meta, seqinfo, sinks, heads, maxpart, out,
            threadgroup_position_in_grid,
            thread_index_in_simdgroup, 0, tree_meta);
        """

    static func mergeKernel(key: PagedAttentionKernelKey, source: String)
        -> MLXFast.MLXFastKernel
    {
        return lock.withLock {
            if let cached = mergeKernels[key] { return cached }
            let name = key.kernelName + "_segments_ordered"
            if let existing = kernels[name] {
                mergeKernels[key] = existing
                return existing
            }
            // previous is a binding-only dependency. CustomKernel registers
            // every input with the Metal encoder, which inserts a full buffer
            // barrier when this prior kernel output is consumed. An alias-only
            // Depends node cannot provide that barrier for hidden writes.
            let made = MLXFast.metalKernel(
                name: name, inputNames: ["partials", "meta", "seqinfo", "sinks", "previous", "tree_meta"],
                outputNames: ["out"], source: orderedMergeBody,
                header: source, ensureRowContiguous: true)
            kernels[name] = made
            mergeKernels[key] = made
            return made
        }
    }

    static func decode(
        queries: MLXArray, newKeys: MLXArray?, newValues: MLXArray?,
        group: PagedKVGroup, rows: [PagedSegmentDispatchPlan.Row],
        sinks: MLXArray?, params: MLXArray, softcap: Bool, source: String,
        dispatchCache: PagedSegmentDispatchCache? = nil,
        zeroFilledScratch: Bool = false,
        seqinfoCache: PagedSegmentSeqinfoCache? = nil,
        mergeScratch: PagedSegmentMergeScratch? = nil,
        treeMeta: MLXArray? = nil,
        pending: (keys: MLXArray, values: MLXArray)? = nil
    ) -> MLXArray {
        decodeWithOutputGate(
            queries: queries, newKeys: newKeys, newValues: newValues, group: group, rows: rows,
            sinks: sinks, params: params, softcap: softcap, source: source,
            dispatchCache: dispatchCache, zeroFilledScratch: zeroFilledScratch,
            seqinfoCache: seqinfoCache, mergeScratch: mergeScratch, treeMeta: treeMeta,
            pending: pending, outputGate: nil).out
    }

    static func decodeWithOutputGate(
        queries: MLXArray,
        newKeys: MLXArray?,
        newValues: MLXArray?,
        group: PagedKVGroup,
        rows: [PagedSegmentDispatchPlan.Row],
        sinks: MLXArray?,
        params: MLXArray,
        softcap: Bool,
        source: String,
        dispatchCache: PagedSegmentDispatchCache? = nil,
        zeroFilledScratch: Bool = false,
        seqinfoCache: PagedSegmentSeqinfoCache? = nil,
        mergeScratch: PagedSegmentMergeScratch? = nil,
        treeMeta: MLXArray? = nil,
        pending: (keys: MLXArray, values: MLXArray)? = nil,
        outputGate: MLXArray?,
        fp8Cast: (multipliers: MLXArray, overflowBound: MLXArray)? = nil
    ) -> (out: MLXArray, gateApplied: Bool) {
        var q = queries.ndim == 4 ? queries.squeezed(axis: 2) : queries
        guard !group.writeValidation.isFaulted else { return (q, false) }
        precondition(pending == nil || treeMeta == nil, "the fused append serves linear rows only")
        if let newKeys, let newValues {
            guard
                group.writeValidation.validate(
                    keys: newKeys, values: newValues, expected: group.dtype)
            else { return (q, false) }
        }
        precondition(q.ndim == 3 && q.dim(0) == rows.count)
        precondition((newKeys == nil) == (newValues == nil))
        let dtype = group.dtype
        let b = q.dim(0)
        let qh = q.dim(1)
        let d = q.dim(2)
        let kvh = group.key.kvHeads
        precondition(d == group.key.headDim && qh % kvh == 0)
        let gqa = qh / kvh
        let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: d, gqa: gqa)!
        let hpt = PagedAttentionKernel.headsPerThreadgroup(headDim: d, gqa: gqa)
        let splits = gqa / hpt
        let (seqinfo, maxLength) =
            seqinfoCache?.seqinfo(rows.map(\.info))
            ?? PagedAttentionKernel.seqinfo(rows.map(\.info))
        let ptok = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokensForDispatch(
                maxAttendLength: maxLength, batch: b, kvHeads: kvh,
                headSplits: splits, pageSize: group.pageSize), pageSize: group.pageSize)
        let prepared =
            dispatchCache?.prepare(
                rows: rows, group: group, partitionTokens: ptok, hasWrite: newKeys != nil)
            ?? PagedSegmentPreparedDispatch(
                rows: rows, group: group, partitionTokens: ptok, hasWrite: newKeys != nil)
        let plan = prepared.plan
        if q.dtype != dtype { q = q.asType(dtype) }
        let k = newKeys ?? q
        let v = newValues ?? q
        if newKeys != nil {
            precondition(k.shape == [b, kvh, d] && v.shape == k.shape)
        }
        // `pending` columns `[KVH, B, D]`: row r's terminal position is its slot.
        func scatterPending(_ pending: (keys: MLXArray, values: MLXArray)) {
            let slots = rows.map {
                let position = $0.info.attendStart + $0.info.attendLength - 1
                let page = $0.pages[(position / group.pageSize) % $0.info.tableLength]
                return page * Int32(group.pageSize) + Int32(position % group.pageSize)
            }
            PagedSegmentTransfers.write(
                group: group, slots: slots, keys: pending.keys, values: pending.values)
        }
        if newKeys == nil, sinks == nil, dispatchCache != nil, PagedOnePassAttention.enabled,
            plan.buckets.count == 1,
            let heads = PagedOnePassAttention.eligible(
                headDim: d, gqa: gqa, simdgroups: nsg, partitions: plan.maxPartitions, rows: b,
                dynTree: treeMeta != nil)
        {
            let bucket = plan.buckets[0]
            var fusedPending = pending
            if let pending,
                !PagedOnePassAttention.supportsAppend(
                    dtype: dtype, headDim: d, kvHeads: kvh, gqa: gqa, pageSize: group.pageSize,
                    hpt: heads, simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                    maxPart: plan.maxPartitions, bindings: bucket.bindingClass)
            {
                scatterPending(pending)
                guard !group.writeValidation.isFaulted else { return (q, false) }
                fusedPending = nil
            }
            var fusedOutGate: MLXArray? = nil
            var fusedFP8Cast: (multipliers: MLXArray, overflowBound: MLXArray)? = nil
            if fusedPending == nil, !PagedOnePassAttention.checks, let outputGate,
                outputGate.dtype == dtype, outputGate.shape == [b, qh, d],
                PagedOnePassAttention.supportsOutputGate(
                    dtype: dtype, headDim: d, kvHeads: kvh, gqa: gqa, pageSize: group.pageSize,
                    hpt: heads, simdgroups: nsg, partitionTokens: ptok, softcap: softcap,
                    maxPart: plan.maxPartitions, bindings: bucket.bindingClass,
                    dynTree: treeMeta != nil, fp8Cast: fp8Cast != nil)
            {
                fusedOutGate = outputGate
                fusedFP8Cast = fp8Cast
            }
            var backing = bucket.segmentIDs.map { group.segments[$0]!.storage }
            backing.append(
                contentsOf: repeatElement(backing[0], count: bucket.bindingClass - backing.count))
            let result = PagedOnePassAttention.segments(
                queries: q, backing: backing, seqinfo: seqinfo, params: params,
                previous: group.writeFence, valueOffsets: prepared.metadata[0].valueOffsets,
                records: prepared.metadata[0].records, dtype: dtype, kvHeads: kvh,
                pageSize: group.pageSize, hpt: heads, simdgroups: nsg, partitionTokens: ptok,
                softcap: softcap, source: source, treeMeta: treeMeta,
                maxPart: plan.maxPartitions,
                pending: fusedPending.map {
                    ($0.keys.transposed(1, 0, 2), $0.values.transposed(1, 0, 2))
                },
                outputGate: fusedOutGate, fp8Cast: fusedFP8Cast)
            if PagedOnePassAttention.checks {
                // The reference reads the pages this dispatch wrote.
                if fusedPending != nil { group.writeFence = result.fence }
                // Self-test: the two-pass kernels on the same inputs, raw bits.
                let reference = decode(
                    queries: q, newKeys: nil, newValues: nil, group: group, rows: rows,
                    sinks: nil, params: params, softcap: softcap, source: source,
                    dispatchCache: nil, zeroFilledScratch: true, treeMeta: treeMeta)
                PagedOnePassAttention.compare(
                    onePass: result.out, twoPass: reference, appended: fusedPending != nil)
                return (reference, false)
            }
            group.writeFence = result.fence
            return (result.out, fusedOutGate != nil)
        }
        if let pending {
            scatterPending(pending)
            guard !group.writeValidation.isFaulted else { return (q, false) }
        }
        let partialsShape = [b, qh, plan.maxPartitions, d]
        // Padding preserves a device pointer even for a one-head/one-partition probe.
        let metaCount = max(8, b * qh * plan.maxPartitions * 2)
        // No fill dispatch: the first bucket allocates the merge scratch as
        // uninitialized kernel outputs and later buckets write their disjoint
        // slots in place. The merge reads only partitions [0, ceil(len / PTOK))
        // of each row, the plan issues exactly one work record per such
        // (row, partition), and each record's threadgroups write every query
        // head's full partial and (m, l) — so no unwritten slot is ever read
        // and the output bytes cannot change. `zeroFilledScratch` keeps the
        // original zero fills as the exactness reference for tests.
        // `mergeScratch` serves the same zero-initialised mutable-input pair
        // from the caller's previous call instead of two fresh fills. Only
        // the slots the merge reads must hold this call's values, and pass A
        // writes every one of them (above), so earlier contents are dead.
        let reused = zeroFilledScratch
            ? mergeScratch?.buffers(partialsShape: partialsShape, metaCount: metaCount) : nil
        var partials: MLXArray? =
            reused?.partials
            ?? (zeroFilledScratch ? MLXArray.zeros(partialsShape, dtype: .float32) : nil)
        var meta: MLXArray? =
            reused?.meta
            ?? (zeroFilledScratch ? MLXArray.zeros([metaCount], dtype: .float32) : nil)
        let partKey = PagedAttentionKernelKey(
            pass: .part, dtype: dtype, headDim: d, pageSize: group.pageSize, gqa: gqa,
            simdgroups: nsg, hasSinks: false, hasSoftcap: softcap,
            partitionTokens: ptok, hasWrite: newKeys != nil)
        for (bucket, metadata) in zip(plan.buckets, prepared.metadata) {
            let segments = bucket.segmentIDs.map { group.segments[$0]! }
            let bindingCount = bucket.bindingClass
            var backing = segments.map(\.storage)
            // Empty records never select padded bindings. They alias an input,
            // never a separately declared writable output.
            backing.append(
                contentsOf: repeatElement(backing[0], count: bindingCount - backing.count))
            let inputs =
                [q, k, v] + backing + [
                    seqinfo, params, group.writeFence, metadata.valueOffsets, metadata.records,
                    treeMeta ?? dummyTreeMeta,
                ]
            let template: [(String, any KernelTemplateArg)] = [
                ("T", dtype), ("D", d), ("S", group.pageSize),
                ("KVH", kvh), ("GQA", gqa), ("HPT", hpt), ("NSG", nsg),
                ("PTOK", ptok), ("HAS_WRITE", newKeys != nil),
                ("HAS_SOFTCAP", softcap), ("SEGMENTS", bindingCount),
                ("STRIDE", PagedSegmentDispatchPlan.recordStride),
                ("DYN_TREE", treeMeta != nil),
            ]
            let grid = (kvh * splits * 32 * nsg, bucket.workCount, 1)
            if let partials, let meta {
                group.writeFence =
                    kernel(
                        key: partKey, bindings: bindingCount, kvHeads: kvh, source: source)(
                        inputs + [partials, meta], template: template, grid: grid,
                        threadGroup: (32 * nsg, 1, 1), outputShapes: [[1]], outputDTypes: [.int32])[
                        0]
            } else {
                let outputs = kernel(
                    key: partKey, bindings: bindingCount, kvHeads: kvh, source: source,
                    allocatesScratch: true)(
                        inputs, template: template, grid: grid, threadGroup: (32 * nsg, 1, 1),
                        outputShapes: [[1], partialsShape, [metaCount]],
                        outputDTypes: [.int32, .float32, .float32])
                group.writeFence = outputs[0]
                partials = outputs[1]
                meta = outputs[2]
            }
        }
        let mergeKey = PagedAttentionKernelKey(
            pass: .merge, dtype: dtype, headDim: d, pageSize: group.pageSize, gqa: gqa,
            simdgroups: 1, hasSinks: sinks != nil, hasSoftcap: false, partitionTokens: ptok)
        let mergeSinks = sinks ?? unusedMergeSinks(count: max(8, qh))
        let out = mergeKernel(key: mergeKey, source: source)(
            [partials!, meta!, seqinfo, mergeSinks, group.writeFence, treeMeta ?? dummyTreeMeta],
            template: [
                ("T", dtype), ("D", d), ("PTOK", ptok), ("HAS_SINKS", sinks != nil),
                ("DYN_TREE", treeMeta != nil),
            ],
            grid: (qh * 32, b, 1), threadGroup: (32, 1, 1),
            outputShapes: [[b, qh, d]], outputDTypes: [dtype])[0]
        return (out, false)
    }
}

/// One merge-scratch pair per caller, zero-filled when its shape is first
/// seen and passed to pass A as mutable inputs on every later call.
///
/// Exact: the merge reads, per row `b` and head, partitions `[0,
/// ceil(len_b / PTOK))` at `maxPartitions` stride; the plan issues one work
/// record per such (row, partition) with `maxPartitions` in `record[6]`; that
/// record's threadgroups (one per kv head and head split) write every query
/// head's full partial and `(m, l)` for it before the merge runs. So no slot
/// the merge reads keeps a value from an earlier call, and the bytes it reads
/// equal those it reads from a fresh zero fill.
///
/// Only for a caller with ONE decode per graph that depends on the previous
/// call's output (a layer's verify window: the next round's queries derive
/// from this round's logits), so the next pass A is encoded after this
/// merge and MLX orders the write after the read (the scratch is registered
/// as an output of pass A and an input of the merge). Never share one across
/// independent decodes of one graph (serial verify columns).
final class PagedSegmentMergeScratch {
    private var shape: [Int] = []
    private var metaCount = 0
    private var pair: (partials: MLXArray, meta: MLXArray)?
    /// Zero-filled pairs made (test hook).
    private(set) var allocations = 0

    func buffers(partialsShape: [Int], metaCount: Int) -> (partials: MLXArray, meta: MLXArray) {
        if let pair, shape == partialsShape, self.metaCount == metaCount { return pair }
        let made = (
            partials: MLXArray.zeros(partialsShape, dtype: .float32),
            meta: MLXArray.zeros([metaCount], dtype: .float32)
        )
        pair = made
        shape = partialsShape
        self.metaCount = metaCount
        allocations += 1
        return made
    }
}

/// One decode seqinfo upload per distinct value instead of one per layer.
///
/// The segmented kernels read a row's attend start, length, table length and
/// write slot from seqinfo, but never its write page: the part kernel takes
/// the fused write's destination from its work record (`write_page` ignores
/// its argument) and nothing else reads that word. With the page zeroed, the
/// layers of one group present identical seqinfo in a decode step (same
/// rows' positions, ring length and slot), so the pool shares one immutable
/// array per value. Every value the kernels read is unchanged.
final class PagedSegmentSeqinfoCache {
    private var entries: [(rows: [PagedAttentionKernel.SeqInfoRow], array: MLXArray)] = []
    /// Arrays built from host values (test hook).
    private(set) var uploads = 0
    static let capacity = 4

    func clear() { entries.removeAll() }

    func seqinfo(_ rows: [PagedAttentionKernel.SeqInfoRow]) -> (
        array: MLXArray, maxAttendLength: Int
    ) {
        var visible = rows
        for index in visible.indices { visible[index].writePage = 0 }
        let maxAttendLength = max(1, visible.map(\.attendLength).max() ?? 1)
        if let index = entries.firstIndex(where: { $0.rows == visible }) {
            let hit = entries.remove(at: index)
            entries.insert(hit, at: 0)
            return (hit.array, maxAttendLength)
        }
        let made = PagedAttentionKernel.seqinfo(visible)
        precondition(made.maxAttendLength == maxAttendLength)
        uploads += 1
        entries.insert((visible, made.array), at: 0)
        if entries.count > Self.capacity { entries.removeLast() }
        return made
    }
}

/// Experimental dispatch batching, not a new attention reduction. A single
/// global-cache sequence becomes one independent native decode row per query.
/// Fixed PTOK is required: adaptive batch sizing would change arithmetic.
enum PagedMTPBatchedColumns {
    static let enabled =
        ProcessInfo.processInfo.environment[
            "DARKBLOOM_NEMOTRON35_MTP_BATCHED_ATTENTION"] != "0"

    static func eligible(queries: MLXArray, row: PagedSequenceKV, group: PagedKVGroup) -> Bool {
        queries.dim(0) == 1 && (2 ... 8).contains(queries.dim(2))
            && row.windowSize == nil && group.segmentLayout != nil
            && row.absoluteOffset >= row.frozenHighWater
            && PagedAttentionKernel.partitionTargetThreadgroups == 0
    }

    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        row: PagedSequenceKV, group: PagedKVGroup,
        sinks: MLXArray?, params: MLXArray, softcap: Bool,
        source: String, dispatchCache: PagedSegmentDispatchCache
    ) -> MLXArray {
        precondition(eligible(queries: queries, row: row, group: group))
        let count = queries.dim(2)
        let heads = queries.dim(1)
        let width = queries.dim(3)
        // One ordinary chunk write, followed by independent causal query rows.
        row.write(keys: keys.squeezed(axis: 0), values: values.squeezed(axis: 0))
        if PagedMultiQueryAttention.enabled,
            PagedMultiQueryAttention.eligible(
                columns: count, queryHeads: heads, headDim: width, group: group)
        {
            // Item 24: the window's columns share ONE pass over the context.
            let batch = queries.transposed(0, 2, 1, 3).reshaped(count, heads, width)
            let result = PagedMultiQueryAttention.attend(
                queries: batch, row: row, group: group, sinks: sinks, params: params,
                softcap: softcap, source: source, dispatchCache: dispatchCache)
            let output = result.reshaped(1, count, heads, width).transposed(0, 2, 1, 3)
            return output.dtype == queries.dtype ? output : output.asType(queries.dtype)
        }
        let descriptors = (0 ..< count).map { position in
            let end = row.absoluteOffset - count + position + 1
            return PagedSegmentDispatchPlan.Row(
                pages: row.table,
                info: row.seqInfoRow(
                    attending: (
                        start: row.baseOffset,
                        length: end - row.baseOffset
                    )),
                identity: .init(serial: row.serial, tableVersion: row.tableVersion))
        }
        let batch = queries.transposed(0, 2, 1, 3).reshaped(count, heads, width)
        let result = PagedSegmentAttention.decode(
            queries: batch, newKeys: nil, newValues: nil,
            group: group, rows: descriptors, sinks: sinks, params: params, softcap: softcap,
            source: source, dispatchCache: dispatchCache)
        let output = result.reshaped(1, count, heads, width).transposed(0, 2, 1, 3)
        return output.dtype == queries.dtype ? output : output.asType(queries.dtype)
    }
}

/// Item 24: multi-query paged verification. A speculative window's `1 + k`
/// columns of ONE row attend the same pages, column `c` over the first
/// `length - (count - 1 - c)` positions. `PagedMTPBatchedColumns` used to run
/// them as `count` independent decode rows, so every column re-read the whole
/// context (5x the KV bytes per layer per round at depth 4). Here one
/// threadgroup owns one (kv head, partition) and runs one SIMDGROUP per
/// (column, query head) pair, as MLX's `sdpa_vector_2pass` does for `qL > 1`:
/// every simdgroup streams the same K/V rows, so each byte comes from memory
/// once per window. Each simdgroup walks its partition's tokens in order with
/// the same online softmax as pass A, writes the same unnormalised partial and
/// `(m, l)` meta, and the ordinary merge pass reduces them per column. Each
/// simdgroup keeps the per-column kernel's NSG strided online-softmax states
/// (token `i` to state `(i - lo) % NSG`) and merges them in the same order,
/// so a column's partial replays the one-token decode kernel's arithmetic for
/// that column: the same operations in the same order (the hardware runs and
/// the near-tie rule are the proof, not this comment). Kill switch:
/// `DARKBLOOM_PAGED_MTP_MULTI_QUERY=0` (falls back to the per-column rows).
enum PagedMultiQueryAttention {
    static let enabled =
        ProcessInfo.processInfo.environment[
            "DARKBLOOM_PAGED_MTP_MULTI_QUERY"] != "0"
    /// Dispatches issued (test/telemetry hook; process-wide).
    nonisolated(unsafe) private(set) static var dispatches = 0
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    private struct MQPartLaunchID: Hashable {
        let kernelName: String
        let workCount: Int
    }

    private struct MQMergeLaunchID: Hashable {
        let key: PagedAttentionKernelKey
        let columns: Int
        let queryHeads: Int
    }

    nonisolated(unsafe) private static var partLaunches:
        [MQPartLaunchID: MLXFast.MLXFastKernel.Launch] = [:]
    nonisolated(unsafe) private static var mergeLaunches:
        [MQMergeLaunchID: MLXFast.MLXFastKernel.Launch] = [:]
    private static let launchLimit = 1024

    static func eligible(columns: Int, queryHeads: Int, headDim: Int, group: PagedKVGroup) -> Bool {
        let kvh = group.key.kvHeads
        guard group.segmentLayout != nil, !group.key.isAsymmetric, kvh > 0,
            queryHeads % kvh == 0, headDim == group.key.headDim, headDim % 32 == 0,
            PagedAttentionKernel.supportedHeadDims.contains(headDim)
        else { return false }
        // One simdgroup per (column, query head of the kv head), at least one
        // column per threadgroup.
        return (2 ... 8).contains(columns)
            && queryHeads / kvh
                <= maximumPairs(
                    headDim: headDim,
                    gqa: queryHeads / kvh)
    }

    /// (column, head) simdgroups one threadgroup may hold. Each simdgroup
    /// keeps NSG online-softmax states of `headDim / 32` floats; register
    /// pressure lowers a pipeline's thread limit below 1,024, and a dispatch
    /// over it fails, so heavy states get 512 threads. Override:
    /// `DARKBLOOM_PAGED_MTP_MQ_THREADS` (64...1024).
    static func maximumPairs(headDim: Int, gqa: Int) -> Int {
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_PAGED_MTP_MQ_THREADS"],
            let threads = Int(raw), (64 ... 1024).contains(threads)
        {
            return threads / 32
        }
        let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa) ?? 8
        return nsg * headDim / 32 <= 32 ? 32 : 16
    }

    static func body(bindings: Int) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        return """
            const device int32_t* record = records + threadgroup_position_in_grid.y * STRIDE;
            const cbv2::PagedSegmentAccessor<T, SEGMENTS> cache{
                {\(pointers)}, value_offsets, record, (size_t)KVH * S * D};
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
                fence[0] = previous[0] + 1;
            }
            constexpr int EPT = D / 32;
            const int kv_head = threadgroup_position_in_grid.x / GROUPS;
            const int part = record[1];
            const int maxpart = record[6];
            const int pair = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const int col = (threadgroup_position_in_grid.x % GROUPS) * CPT + pair / GQA;
            if (col >= COLS) {
                return;
            }
            const int head = kv_head * GQA + pair % GQA;
            const int qh = KVH * GQA;
            const device int32_t* info = seqinfo + col * 8;
            const int attend_start = info[0];
            const int attend_len = info[1];
            const int lo = part * PTOK;
            if (lo >= attend_len) {
                return;
            }
            const int hi = min(attend_len, lo + PTOK);
            const float softcap = params[0];
            const float scale = params[1];
            float qv[EPT];
            const device T* qrow = q + ((size_t)col * (size_t)qh + (size_t)head) * D + lane * EPT;
            for (int e = 0; e < EPT; e++) {
                qv[e] = float(qrow[e]) * scale;
            }
            // NSG online-softmax states, token i to state (i - lo) % NSG, merged
            // in state order: the per-column kernel's simdgroup split and
            // merge, replayed by one simdgroup, so every column's partial is
            // the one-token decode kernel's for that column.
            float m[NSG];
            float l[NSG];
            float acc[NSG][EPT];
            for (int s = 0; s < NSG; s++) {
                m[s] = -FLT_MAX;
                l[s] = 0.0f;
                for (int e = 0; e < EPT; e++) {
                    acc[s][e] = 0.0f;
                }
            }
            float kf[EPT];
            float vf[EPT];
            // Unrolled by NSG so each state index is a compile-time constant
            // (no dynamically indexed register arrays).
            for (int first = lo; first < hi; first += NSG) {
                #pragma unroll
                for (int s = 0; s < NSG; s++) {
                    const int i = first + s;
                    if (i < hi) {
                        const int pos = attend_start + i;
                        const auto page = cache.read_page(seqinfo, pos / S, 0);
                        const size_t base = ((size_t)kv_head * S + (size_t)(pos % S)) * D;
                        cbv2::load_row<T, EPT>(page.key + base, lane, kf);
                        cbv2::load_row<T, EPT>(page.value + base, lane, vf);
                        float partial = 0.0f;
                        for (int e = 0; e < EPT; e++) {
                            partial += qv[e] * kf[e];
                        }
                        float qk = simd_sum(partial);
                        if (HAS_SOFTCAP) {
                            qk = softcap * precise::tanh(qk / softcap);
                        }
                        qk *= M_LOG2E_F;
                        const float m_new = max(m[s], qk);
                        const float corr = exp2(m[s] - m_new);
                        const float p = exp2(qk - m_new);
                        l[s] = l[s] * corr + p;
                        for (int e = 0; e < EPT; e++) {
                            acc[s][e] = acc[s][e] * corr + p * vf[e];
                        }
                        m[s] = m_new;
                    }
                }
            }
            float pm = -FLT_MAX;
            for (int t = 0; t < NSG; t++) {
                pm = max(pm, m[t]);
            }
            float pl = 0.0f;
            for (int t = 0; t < NSG; t++) {
                pl += l[t] * exp2(m[t] - pm);
            }
            const size_t slot = ((size_t)col * (size_t)qh + (size_t)head) * (size_t)maxpart + (size_t)part;
            device float* prow = partials + slot * D + lane * EPT;
            for (int e = 0; e < EPT; e++) {
                float v = 0.0f;
                for (int t = 0; t < NSG; t++) {
                    v += acc[t][e] * exp2(m[t] - pm);
                }
                prow[e] = v;
            }
            if (lane == 0) {
                meta[slot * 2] = pm;
                meta[slot * 2 + 1] = pl;
            }
            """
    }

    private static func kernel(name: String, bindings: Int, source: String) -> MLXFast.MLXFastKernel
    {
        return lock.withLock {
            if let existing = kernels[name] { return existing }
            let made = MLXFast.metalKernel(
                name: name, inputNames: PagedSegmentAttention.inputNames(bindings: bindings),
                outputNames: ["fence"], source: body(bindings: bindings), header: source,
                ensureRowContiguous: true, mutableInputs: ["partials", "meta"])
            kernels[name] = made
            return made
        }
    }

    /// `queries` is `[count, queryHeads, D]` (column-major batch, as the
    /// per-column path lays it out); `row` already holds the window's
    /// columns (written by the caller), so column `c` attends
    /// `[base, absoluteOffset - count + c + 1)`.
    static func attend(
        queries: MLXArray, row: PagedSequenceKV, group: PagedKVGroup,
        sinks: MLXArray?, params: MLXArray, softcap: Bool, source: String,
        dispatchCache: PagedSegmentDispatchCache
    ) -> MLXArray {
        var q = queries
        let count = q.dim(0)
        let qh = q.dim(1)
        let d = q.dim(2)
        guard !group.writeValidation.isFaulted else { return q }
        let kvh = group.key.kvHeads
        let gqa = qh / kvh
        guard let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: d, gqa: gqa) else {
            return q
        }
        // Columns per threadgroup, balanced over the fewest groups: each group
        // reads the context once, so the window reads it `groups` times.
        let groups =
            (count + max(1, maximumPairs(headDim: d, gqa: gqa) / gqa) - 1)
            / max(1, maximumPairs(headDim: d, gqa: gqa) / gqa)
        let cpt = (count + groups - 1) / groups
        let dtype = group.dtype
        let last = (start: row.baseOffset, length: row.absoluteOffset - row.baseOffset)
        // The partition grid is the LONGEST column's; shorter columns skip
        // the tokens past their own end (at most `count - 1` of the last
        // partition) and the merge counts each column's own partitions.
        let ptok = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokens, pageSize: group.pageSize)
        let descriptor = PagedSegmentDispatchPlan.Row(
            pages: row.table, info: row.seqInfoRow(attending: last),
            identity: .init(serial: row.serial, tableVersion: row.tableVersion))
        let prepared = dispatchCache.prepare(
            rows: [descriptor], group: group, partitionTokens: ptok, hasWrite: false)
        let plan = prepared.plan
        let (seqinfo, _) = PagedAttentionKernel.seqinfo(
            (0 ..< count).map { c in
                row.seqInfoRow(
                    attending: (start: last.start, length: last.length - (count - 1 - c)))
            })
        if q.dtype != dtype { q = q.asType(dtype) }
        let partials = MLXArray.zeros([count, qh, plan.maxPartitions, d], dtype: .float32)
        let meta = MLXArray.zeros([max(8, count * qh * plan.maxPartitions * 2)], dtype: .float32)
        let dname: String
        switch dtype {
        case .float16: dname = "f16"
        case .float32: dname = "f32"
        case .bfloat16: dname = "bf16"
        default: dname = "dt\(dtype)"
        }
        for (bucket, metadata) in zip(plan.buckets, prepared.metadata) {
            let segments = bucket.segmentIDs.map { group.segments[$0]! }
            let bindingCount = bucket.bindingClass
            var backing = segments.map(\.storage)
            backing.append(
                contentsOf: repeatElement(backing[0], count: bindingCount - backing.count))
            let inputs =
                [q, q, q] + backing + [
                    seqinfo, params, group.writeFence, metadata.valueOffsets,
                    metadata.records, PagedSegmentAttention.dummyTreeMeta, partials, meta,
                ]
            let name =
                "cbv2_paged_mq_\(dname)_d\(d)_s\(group.pageSize)_g\(gqa)_n\(nsg)_c\(count)_x\(cpt)"
                + "_t\(ptok)_kvh\(kvh)_cap\(softcap ? 1 : 0)_segments\(bindingCount)"
            let partID = MQPartLaunchID(kernelName: name, workCount: bucket.workCount)
            if let launch = lock.withLock({ partLaunches[partID] }) {
                group.writeFence = launch(inputs)[0]
            } else {
                let made = kernel(name: name, bindings: bindingCount, source: source).prepare(
                    template: [
                        ("T", dtype), ("D", d), ("S", group.pageSize), ("KVH", kvh),
                        ("GQA", gqa), ("NSG", nsg), ("PTOK", ptok), ("HAS_SOFTCAP", softcap),
                        ("GROUPS", groups), ("CPT", cpt), ("COLS", count),
                        ("SEGMENTS", bindingCount),
                        ("STRIDE", PagedSegmentDispatchPlan.recordStride),
                    ],
                    grid: (kvh * groups * 32 * gqa * cpt, bucket.workCount, 1),
                    threadGroup: (32 * gqa * cpt, 1, 1), outputShapes: [[1]],
                    outputDTypes: [.int32])
                let launch = lock.withLock {
                    if let hit = partLaunches[partID] { return hit }
                    if partLaunches.count < launchLimit { partLaunches[partID] = made }
                    return made
                }
                group.writeFence = launch(inputs)[0]
            }
        }
        lock.withLock { dispatches += 1 }
        let mergeKey = PagedAttentionKernelKey(
            pass: .merge, dtype: dtype, headDim: d, pageSize: group.pageSize, gqa: gqa,
            simdgroups: 1, hasSinks: sinks != nil, hasSoftcap: false, partitionTokens: ptok)
        let zeroSinks = sinks ?? PagedSegmentAttention.unusedMergeSinks(count: max(8, qh))
        let mergeInputs = [
            partials, meta, seqinfo, zeroSinks, group.writeFence,
            PagedSegmentAttention.dummyTreeMeta,
        ]
        let mergeID = MQMergeLaunchID(key: mergeKey, columns: count, queryHeads: qh)
        if let launch = lock.withLock({ mergeLaunches[mergeID] }) {
            return launch(mergeInputs)[0]
        }
        let madeMerge = PagedSegmentAttention.mergeKernel(key: mergeKey, source: source).prepare(
            template: [
                ("T", dtype), ("D", d), ("PTOK", ptok), ("HAS_SINKS", sinks != nil),
                ("DYN_TREE", false),
            ],
            grid: (qh * 32, count, 1), threadGroup: (32, 1, 1),
            outputShapes: [[count, qh, d]], outputDTypes: [dtype])
        let mergeLaunch = lock.withLock {
            if let hit = mergeLaunches[mergeID] { return hit }
            if mergeLaunches.count < launchLimit { mergeLaunches[mergeID] = madeMerge }
            return madeMerge
        }
        return mergeLaunch(mergeInputs)[0]
    }
}