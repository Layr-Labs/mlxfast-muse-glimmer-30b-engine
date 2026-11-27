import Foundation
import MLX
import MLXFast

/// Bounded page transfers. A dispatch binds one segment and the request's input
/// or final output, never a concatenation of the pool's native backing.
enum PagedSegmentTransfers {
    private static let writeBody = """
        const int h = int(thread_position_in_grid.y);
        const int d = int(thread_position_in_grid.x);
        const int r = int(thread_position_in_grid.z);
        const int token = records[r * 3];
        const int page = records[r * 3 + 1];
        const int slot = records[r * 3 + 2];
        const int n = keys_shape[1];
        if (page > 0) {
            const size_t source = ((size_t)h * n + token) * D + d;
            const size_t target = (((size_t)page * H + h) * S + slot) * D + d;
            device T* destination = storage;
            destination[target] = keys[source];
            destination[VBASE + target] = values[source];
        }
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """

    private static let readBody = """
        const int h = int(thread_position_in_grid.y);
        const int d = int(thread_position_in_grid.x);
        const int r = int(thread_position_in_grid.z);
        const int token = records[r * 3];
        const int page = records[r * 3 + 1];
        const int slot = records[r * 3 + 2];
        const int count = output_shape[3];
        const size_t source = (((size_t)page * H + h) * S + slot) * D + d;
        const size_t target = ((size_t)h * count + token) * D + d;
        device T* destination = output;
        destination[target] = storage[source];
        destination[(size_t)H * count * D + target] = storage[VBASE + source];
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """

    /// Depends is a buffer alias, not an encoded Metal dependency. A later
    /// Slice may copy that alias, so force one real GPU consumer of the final
    /// transfer fence first. Its read-after-write barrier covers the hidden
    /// in-place destination writes; only four bytes of new output are needed.
    private static let completionKernel = MLXFast.metalKernel(
        name: "cbv2_segment_transfer_complete", inputNames: ["previous"],
        outputNames: ["witness"], source: "witness[0] = previous[0] + 1;",
        ensureRowContiguous: true)

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    private static func kernel(
        reading: Bool, dtype: DType,
        quantization: PagedKVQuantizationConfig? = nil
    ) -> MLXFast.MLXFastKernel {
        let key =
            "cbv2_segment_\(reading ? "read" : "write")_\(dtype)"
            + (quantization.map {
                "_k\($0.keyBits)v\($0.valueBits)g\($0.groupSize)r\($0.rotationBlockSize)"
            } ?? "")
        return lock.withLock {
            if let existing = kernels[key] { return existing }
            let made = MLXFast.metalKernel(
                name: key,
                inputNames: (reading
                    ? ["storage", "records", "output", "previous"]
                    : ["keys", "values", "storage", "records", "previous"])
                    + (quantization == nil ? [] : ["value_base"]),
                outputNames: ["fence"],
                source: quantization == nil
                    ? (reading ? readBody : writeBody)
                    : (reading
                        ? PagedQuantizedTransfers.readBody : PagedQuantizedTransfers.writeBody),
                header: quantization == nil ? "" : PagedQuantizedMetal.header,
                ensureRowContiguous: quantization == nil,
                mutableInputs: [reading ? "output" : "storage"])
            kernels[key] = made
            return made
        }
    }

    static func records(_ triples: [Int32]) -> MLXArray {
        precondition(triples.count % 3 == 0)
        return MLXArray(triples + Array(repeating: 0, count: max(0, 24 - triples.count)))
    }

    /// Runtime byte metadata follows the transfer fence's input ownership.
    /// Allocation length must never become a packed shader specialization.
    static func quantizedValueBase(_ segment: PagedKVSegment) -> MLXArray {
        precondition(segment.valueOffset >= 0 && segment.valueOffset <= segment.storage.nbytes)
        return MLXArray([Int64(segment.valueOffset)])
    }

    /// Each record is (input/output token index, local page, slot).
    /// Pages are resolved once per page, not once per token, and records are
    /// appended in place: a whole-history gather at a 30K context used to
    /// allocate a temporary array per token, on every layer of every prompt
    /// chunk. Same records, same order (ascending tokens within a segment).
    static func buckets(group: PagedKVGroup, slots: [Int32]) -> [(PagedKVSegment, [Int32])] {
        let layout = group.segmentLayout!
        let pageSize = Int32(group.pageSize)
        var records: [Int: [Int32]] = [:]
        var run: [Int32] = []
        run.reserveCapacity(slots.count * 3)
        var index = -1
        var local: Int32 = 0
        var lastPage: Int32 = -1
        func flush() {
            guard index >= 0, !run.isEmpty else { return }
            records[index, default: []].append(contentsOf: run)
            run.removeAll(keepingCapacity: true)
        }
        for (token, position) in slots.enumerated() {
            precondition(position >= 0)
            let page = position / pageSize
            if page != lastPage {
                precondition(group.isAllocatable(page))
                let next = layout.segmentIndex(page: page)
                if next != index {
                    flush()
                    index = next
                }
                local = Int32(layout.localPage(page))
                lastPage = page
            }
            run.append(Int32(token))
            run.append(local)
            run.append(position % pageSize)
        }
        flush()
        return records.keys.sorted().map { index in
            guard let segment = group.segments[index] else {
                preconditionFailure("transfer names an uncommitted segment")
            }
            return (segment, records[index]!)
        }
    }

    private static func quantizationTemplate(_ group: PagedKVGroup) -> [(
        String, any KernelTemplateArg
    )] {
        guard let q = group.key.quantization else { return [] }
        return [
            ("G", q.groupSize), ("KB", q.keyBits), ("VB", q.valueBits),
            ("R", q.resolvedRotationBlockSize(headDim: group.key.headDim)),
        ]
    }

    static func write(group: PagedKVGroup, slots: [Int32], keys: MLXArray, values: MLXArray) {
        guard group.writeValidation.validate(keys: keys, values: values, expected: group.dtype)
        else { return }
        guard
            group.writeValidation.validateShape(
                keys: keys, values: values, group: group.key,
                rank: 3, batch: nil, tokens: slots.count)
        else { return }
        if group.key.isAsymmetric {
            PagedAsymmetricTransfers.writeSegmented(
                group: group, slots: slots, keys: keys, values: values)
            return
        }
        precondition(keys.shape == [group.key.kvHeads, slots.count, group.key.headDim])
        precondition(values.shape == keys.shape)
        let k = keys
        let v = values
        for (segment, triples) in buckets(group: group, slots: slots) {
            let offset = group.key.quantization == nil ? [] : [quantizedValueBase(segment)]
            group.writeFence =
                kernel(reading: false, dtype: group.dtype, quantization: group.key.quantization)(
                    [k, v, segment.storage, records(triples), group.writeFence] + offset,
                    template: [
                        ("T", group.dtype), ("H", group.key.kvHeads),
                        ("D", group.key.headDim), ("S", group.pageSize),
                    ] + (group.key.quantization == nil ? [("VBASE", segment.valueOffset)] : [])
                        + quantizationTemplate(group),
                    grid: (group.key.headDim, group.key.kvHeads, triples.count / 3),
                    threadGroup: (
                        group.key.quantization == nil
                            ? min(256, group.key.headDim) : group.key.headDim, 1, 1
                    ),
                    outputShapes: [[1]], outputDTypes: [.int32])[0]
        }
    }

    // MARK: Draft tree (CBV2_VERIFY_TREE)

    /// In-segment slot copy: record `r` = (source local page, source slot,
    /// destination local page, destination slot); every head, every lane of
    /// key and value. Sources and destinations are distinct slots.
    private static let copyBody = """
        const int h = int(thread_position_in_grid.y);
        const int d = int(thread_position_in_grid.x);
        const int r = int(thread_position_in_grid.z);
        const size_t source =
            (((size_t)records[r * 4] * H + h) * S + records[r * 4 + 1]) * D + d;
        const size_t target =
            (((size_t)records[r * 4 + 2] * H + h) * S + records[r * 4 + 3]) * D + d;
        device T* destination = storage;
        destination[target] = storage[source];
        destination[VBASE + target] = storage[VBASE + source];
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """

    private static func copyKernel(dtype: DType) -> MLXFast.MLXFastKernel {
        let key = "cbv2_segment_tree_copy_\(dtype)"
        return lock.withLock {
            if let existing = kernels[key] { return existing }
            let made = MLXFast.metalKernel(
                name: key, inputNames: ["storage", "records", "previous"],
                outputNames: ["fence"], source: copyBody, ensureRowContiguous: true,
                mutableInputs: ["storage"])
            kernels[key] = made
            return made
        }
    }

    /// Per segment, the (source, destination) pool slot pairs of one group,
    /// as copy records; nil when a pair's two pages sit in different segments
    /// (the copy is then not expressible here and the caller commits no leaf).
    static func copyPlan(group: PagedKVGroup, pairs: [(source: Int32, destination: Int32)])
        -> [(PagedKVSegment, [Int32])]?
    {
        guard let layout = group.segmentLayout, group.key.quantization == nil,
            !group.key.isAsymmetric
        else { return nil }
        let pageSize = Int32(group.pageSize)
        var records: [Int: [Int32]] = [:]
        for pair in pairs {
            let sourcePage = pair.source / pageSize
            let destinationPage = pair.destination / pageSize
            guard pair.source >= 0, pair.destination >= 0, pair.source != pair.destination,
                layout.isUsable(sourcePage), layout.isUsable(destinationPage)
            else { return nil }
            let segment = layout.segmentIndex(page: sourcePage)
            guard layout.segmentIndex(page: destinationPage) == segment,
                group.segments[segment] != nil
            else { return nil }
            records[segment, default: []].append(contentsOf: [
                Int32(layout.localPage(sourcePage)), pair.source % pageSize,
                Int32(layout.localPage(destinationPage)), pair.destination % pageSize,
            ])
        }
        return records.keys.sorted().map { (group.segments[$0]!, records[$0]!) }
    }

    /// Load time: compiles the copy kernel and checks one in-place slot copy
    /// on a scratch page (slot 3 onto slot 5, keys and values, every head).
    static func warmTreeCopy(dtype: DType, kvHeads: Int, headDim: Int, pageSize: Int) -> Bool {
        guard kvHeads > 0, headDim > 0, pageSize > 5 else { return false }
        let pageElements = kvHeads * pageSize * headDim
        let storage = MLXRandom.normal([2 * pageElements], key: MLXRandom.key(21)).asType(dtype)
        eval(storage)
        let records = MLXArray([Int32(0), 3, 0, 5] + [Int32](repeating: 0, count: 20))
        let fence = copyKernel(dtype: dtype)(
            [storage, records, MLXArray.zeros([1], type: Int32.self)],
            template: [
                ("T", dtype), ("H", kvHeads), ("D", headDim), ("S", pageSize),
                ("VBASE", pageElements),
            ],
            grid: (headDim, kvHeads, 1), threadGroup: (min(256, headDim), 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32])[0]
        eval(fence)
        let page = storage.reshaped([2, kvHeads, pageSize, headDim]).asType(.float32)
        let same = all(page[0..., 0..., 3 ..< 4, 0...] .== page[0..., 0..., 5 ..< 6, 0...])
        eval(same)
        return same.item(Bool.self)
    }

    /// Runs a `copyPlan`: one dispatch per segment, ordered on the group's
    /// write fence like every other in-place write.
    static func copy(group: PagedKVGroup, plan: [(PagedKVSegment, [Int32])]) {
        for (segment, quads) in plan where !quads.isEmpty {
            let padded = quads + Array(repeating: 0, count: max(0, 24 - quads.count))
            group.writeFence = copyKernel(dtype: group.dtype)(
                [segment.storage, MLXArray(padded), group.writeFence],
                template: [
                    ("T", group.dtype), ("H", group.key.kvHeads), ("D", group.key.headDim),
                    ("S", group.pageSize), ("VBASE", segment.valueOffset),
                ],
                grid: (group.key.headDim, group.key.kvHeads, quads.count / 4),
                threadGroup: (min(256, group.key.headDim), 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
        }
        // Submit now: the source slot's page may go back to the free list at
        // this round's commit, before anything else would evaluate the copy.
        asyncEval([group.writeFence])
    }

    static func gather(group: PagedKVGroup, pages: [Int32], firstSlot: Int, count: Int)
        -> (keys: MLXArray, values: MLXArray)
    {
        if group.key.isAsymmetric {
            return PagedAsymmetricTransfers.gatherSegmented(
                group: group, pages: pages, firstSlot: firstSlot, count: count)
        }
        let ordered = gatherCombined(group: group, pages: pages, firstSlot: firstSlot, count: count)
        return (ordered[0], ordered[1])
    }

    /// Historical owners retain this single output, avoiding two potentially
    /// materializing Slice views. The caller reserves every output/record/fence
    /// before calling and keeps its charge until the captured owner retires.
    /// `publishReadFence: false` requires the owning engine step to block all
    /// successor writes until the private read has completed or drained.
    static func gatherCombined(
        group: PagedKVGroup, pages: [Int32], firstSlot: Int, count: Int,
        publishReadFence: Bool = true, stream: StreamOrDevice = .default
    ) -> MLXArray {
        precondition(
            !group.key.isAsymmetric, "legacy combined gather cannot represent asymmetric roles")
        let h = group.key.kvHeads
        let d = group.key.headDim
        guard count > 0 else {
            return MLXArray.zeros([2, 1, h, 0, d], dtype: group.dtype, stream: stream)
        }
        precondition(firstSlot >= 0 && firstSlot < group.pageSize)
        precondition(pages.count * group.pageSize >= firstSlot + count)
        let slots = (0 ..< count).map { token -> Int32 in
            let offset = firstSlot + token
            return pages[offset / group.pageSize] * Int32(group.pageSize)
                + Int32(offset % group.pageSize)
        }
        let output = MLXArray.zeros([2, 1, h, count, d], dtype: group.dtype, stream: stream)
        var fence = group.writeFence
        for (segment, triples) in buckets(group: group, slots: slots) {
            let offset = group.key.quantization == nil ? [] : [quantizedValueBase(segment)]
            fence =
                kernel(reading: true, dtype: group.dtype, quantization: group.key.quantization)(
                    [segment.storage, records(triples), output, fence] + offset,
                    template: [
                        ("T", group.dtype), ("H", h), ("D", d),
                        ("S", group.pageSize),
                    ] + (group.key.quantization == nil ? [("VBASE", segment.valueOffset)] : [])
                        + quantizationTemplate(group),
                    grid: (d, h, triples.count / 3),
                    threadGroup: (group.key.quantization == nil ? min(256, d) : d, 1, 1),
                    outputShapes: [[1]], outputDTypes: [.int32], stream: stream)[0]
        }
        fence =
            completionKernel(
                [fence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32], stream: stream)[0]
        // Keep the final destination; no second whole-gather allocation.
        if publishReadFence { group.writeFence = fence }
        // MLX Depends inherits its input primitive stream (ops.cpp); host
        // Int32 record arrays have no primitive or stream to drain. Source
        // storage/fence dependencies cross streams through MLX events.
        return depends(input: output, dependencies: [fence])
    }
}
