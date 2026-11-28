import Foundation
import MLX
import MLXRandom

public enum PagedOnePassBench {
    public struct Result {
        public let microsecondsPerCall: Double
        public let out: MLXArray
        public let nsg: Int
        public let hpt: Int
    }

    public static func run(
        ctx: Int, rows: Int, tree: Bool, seed: UInt64, layers: Int = 16, calls: Int = 52,
        reps: Int = 9, multiRow: (Int, Int, Int)? = nil, gate: Int = 0,
        kvHeads: Int = 2, queryHeads: Int = 32, headDim: Int = 128, timed: Bool = true
    ) -> Result {
        let pageSize = 16
        let ptok = 256
        let dtype = DType.bfloat16
        let gqa = queryHeads / kvHeads
        let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa)!
        let source: String
        if let path = ProcessInfo.processInfo.environment["ATTN_METAL_SRC"] {
            source = try! String(contentsOfFile: path, encoding: .utf8)
        } else {
            source = try! PagedAttentionResources.loadSourceForCurrentProcess()
        }
        let pe = kvHeads * pageSize * headDim
        let maxLen = tree ? ctx + 8 : ctx + rows
        let numPages = (maxLen + pageSize - 1) / pageSize + 2
        MLXRandom.seed(seed)
        var rng = SystemRandomNumberGenerator()
        _ = rng
        var lens = [Int](), depths = [Int](), ancs = [UInt32]()
        if tree {
            var parent = [Int](repeating: 0, count: rows)
            var d = [Int](repeating: 0, count: rows)
            var anc = [[Int]](repeating: [0], count: rows)
            var g = SplitMix(seed: seed &* 7919 &+ 13)
            for s in 1 ..< max(rows, 1) {
                parent[s] = Int(g.next() % UInt64(s))
                d[s] = d[parent[s]] + 1
                anc[s] = anc[parent[s]] + [s]
            }
            for s in 0 ..< rows {
                depths.append(d[s])
                lens.append(ctx + d[s] + 1)
                var packed: UInt32 = 0
                for (k, a) in anc[s].enumerated() { packed |= UInt32(a) << UInt32(4 * k) }
                ancs.append(packed)
            }
        } else {
            for r in 0 ..< rows { lens.append(ctx + r + 1) }
        }
        let maxPartitions = lens.map { ($0 + ptok - 1) / ptok }.max()!
        guard
            let heads0 = PagedOnePassAttention.eligible(
                headDim: headDim, gqa: gqa, simdgroups: nsg, partitions: maxPartitions,
                rows: rows, dynTree: tree, record: false)
        else {
            fatalError("one-pass not eligible at ctx \(ctx) (maxPartitions \(maxPartitions))")
        }
        var perm = Array(0 ..< numPages)
        var sm = SplitMix(seed: seed &+ 99)
        for i in stride(from: numPages - 1, to: 0, by: -1) {
            perm.swapAt(i, Int(sm.next() % UInt64(i + 1)))
        }
        var seq = [Int32]()
        for r in 0 ..< rows {
            seq += [0, Int32(lens[r]), 1000, 0, 0, 0, tree ? Int32(ctx) : 0, 0]
        }
        var records = [Int32]()
        for r in 0 ..< rows {
            let len = lens[r]
            let parts = (len + ptok - 1) / ptok
            for p in 0 ..< parts {
                let start = p * ptok
                let end = min(len, (p + 1) * ptok)
                let first = start / pageSize
                var last = (end - 1) / pageSize
                if tree { last = min((len - 1) / pageSize, (end - 1 + 7) / pageSize) }
                var rec: [Int32] = [
                    Int32(r), Int32(p), Int32(first), Int32(last - first + 1), 0, 0,
                    Int32(maxPartitions), 0,
                ]
                for lp in first ... last {
                    rec.append(0)
                    rec.append(Int32(perm[lp]))
                }
                rec += [Int32](
                    repeating: 0, count: PagedSegmentDispatchPlan.recordStride - rec.count)
                records += rec
            }
        }
        var treeMetaValues = [Int32](repeating: 0, count: 18)
        if tree {
            for s in 0 ..< rows {
                treeMetaValues[1 + s] = Int32(bitPattern: ancs[s])
                treeMetaValues[9 + s] = Int32(depths[s])
            }
        }
        let treeMeta: MLXArray? = tree ? MLXArray(treeMetaValues) : nil
        let seqinfo = MLXArray(seq, [rows, 8])
        let recordsArray = MLXArray(records)
        let valueOffsets = MLXArray([Int64](repeating: Int64(numPages * pe), count: 8))
        let params = MLXArray([Float(1), Float(1.0 / Double(headDim).squareRoot()), 0, 0, 0, 0, 0, 0])
        var storages = [MLXArray]()
        var queries = [MLXArray]()
        for _ in 0 ..< layers {
            storages.append((MLXRandom.normal([2 * numPages * pe]) * 1.5).asType(dtype))
            queries.append((MLXRandom.normal([rows, queryHeads, headDim]) * 2).asType(dtype))
        }
        let gateArray = (MLXRandom.normal([rows, queryHeads, headDim])).asType(dtype)
        let fp8: (multipliers: MLXArray, overflowBound: MLXArray)? =
            gate == 2 ? (MLXArray([Float(1.0), Float(1.0)]), MLXArray([Float(448.0)])) : nil
        eval(storages + queries + [gateArray, seqinfo, recordsArray, valueOffsets, params])
        if let treeMeta { eval(treeMeta) }
        let previous0 = MLXArray.zeros([1], type: Int32.self)
        eval(previous0)

        func chain(_ n: Int) -> [MLXArray] {
            var prev = previous0
            var outs = [MLXArray]()
            for i in 0 ..< n {
                let li = i % layers
                let result = PagedOnePassAttention.segments(
                    queries: queries[li], backing: [storages[li]], seqinfo: seqinfo,
                    params: params, previous: prev, valueOffsets: valueOffsets,
                    records: recordsArray, dtype: dtype, kvHeads: kvHeads,
                    pageSize: pageSize, hpt: heads0, simdgroups: nsg, partitionTokens: ptok,
                    softcap: false, source: source, treeMeta: treeMeta,
                    maxPart: maxPartitions, outputGate: gate > 0 ? gateArray : nil,
                    fp8Cast: fp8, record: false,
                    multiRow: multiRow.map {
                        PagedOnePassAttention.MultiRowConfig(rows: $0.0, heads: $0.1, wave: $0.2)
                    })
                prev = result.fence
                outs.append(result.out)
            }
            eval(outs + [prev])
            return outs
        }
        let first = chain(layers)
        var us = 0.0
        if timed {
            _ = chain(calls)
            var samples = [Double]()
            for _ in 0 ..< reps {
                let t0 = DispatchTime.now().uptimeNanoseconds
                _ = chain(calls)
                let t1 = DispatchTime.now().uptimeNanoseconds
                samples.append(Double(t1 - t0) / 1000.0 / Double(calls))
            }
            samples.sort()
            us = samples[samples.count / 2]
        }
        let all = concatenated(first.map { $0.reshaped(-1) })
        eval(all)
        return Result(microsecondsPerCall: us, out: all, nsg: nsg, hpt: heads0)
    }

    struct SplitMix {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }
}
