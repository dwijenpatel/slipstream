import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Qwen3.6's full-attention layers (head dim 256, 16 query heads over 2
/// key-value heads) used to fall to the scalar causal kernel, which was 55
/// percent of a 12k prefill. The tensor-ops kernel ported from gpu-kernel's
/// prefill-attention champion takes that shape on Apple10 GPUs. These tests
/// pin the selection rule and the numerics against a CPU reference.
@Suite struct PrefillAttentionHD256Tests {
    private static func qwenParams(start: Int, chunk: Int,
                                   window: UInt32? = nil,
                                   headDim: Int = 256) -> PrefillAttentionParams {
        PrefillAttentionParams(
            startPosition: UInt32(start),
            queryCount: UInt32(chunk),
            headDim: UInt32(headDim),
            numQHeads: 16,
            numKVHeads: 2,
            kvValidCount: UInt32(start + chunk),
            slidingWindow: window ?? UInt32(start + chunk),
            kvTokenStrideElements: UInt32(2 * headDim),
            qTokenStrideElements: UInt32(16 * headDim),
            oTokenStrideElements: UInt32(16 * headDim),
            scale: 0.0625)
    }

    @Test func selectsTheHD256KernelForTheQwenShape() {
        #expect(PrefillAttention.selectsHD256TensorOps(
            params: Self.qwenParams(start: 100, chunk: 64),
            kvRingCapacity: 0, path: .fullTensorOps2DPreferred))
    }

    @Test func fallsBackWhenTheChunkIsSmallerThanAQueryTile() {
        #expect(!PrefillAttention.selectsHD256TensorOps(
            params: Self.qwenParams(start: 100, chunk: 31),
            kvRingCapacity: 0, path: .fullTensorOps2DPreferred))
    }

    @Test func fallsBackWhenFewerKeysThanAKeyTileAreVisible() {
        #expect(!PrefillAttention.selectsHD256TensorOps(
            params: Self.qwenParams(start: 0, chunk: 100),
            kvRingCapacity: 0, path: .fullTensorOps2DPreferred))
    }

    @Test func fallsBackForARingAWindowAnotherHeadDimOrTheScalarPath() {
        let p = Self.qwenParams(start: 100, chunk: 64)
        #expect(!PrefillAttention.selectsHD256TensorOps(params: p, kvRingCapacity: 1024,
                                                       path: .fullTensorOps2DPreferred))
        #expect(!PrefillAttention.selectsHD256TensorOps(
            params: Self.qwenParams(start: 100, chunk: 64, window: 50),
            kvRingCapacity: 0, path: .fullTensorOps2DPreferred))
        #expect(!PrefillAttention.selectsHD256TensorOps(
            params: Self.qwenParams(start: 100, chunk: 64, headDim: 512),
            kvRingCapacity: 0, path: .fullTensorOps2DPreferred))
        #expect(!PrefillAttention.selectsHD256TensorOps(params: p, kvRingCapacity: 0,
                                                       path: .causalTiled))
    }

    /// The correctness cases would also pass through the scalar fallback, so
    /// the pipeline's existence is pinned on Apple10 hardware.
    @Test func theHD256PipelineBuildsOnApple10() throws {
        let context = try MetalContext()
        guard context.device.supportsFamily(.apple10) else { return }
        let prefill = try PrefillAttention(context: context)
        #expect(prefill.hd256TensorOpsAvailable)
    }

    /// Tile geometry: BQ 32 queries, BK 128 keys. Cases cover exact tiles,
    /// a ragged query count (overlapping final query tile), a ragged key
    /// count (overlapping final key tile), and a chunk below both tiles that
    /// must fall back and still be right.
    @Test(arguments: [
        (start: 0, chunk: 128),
        (start: 100, chunk: 33),
        (start: 300, chunk: 132),
        (start: 640, chunk: 96),
        (start: 0, chunk: 64),
        (start: 127, chunk: 1),
    ])
    func hd256MatchesTheCPUReferenceAndIsByteStable(_ c: (start: Int, chunk: Int)) throws {
        let context = try MetalContext()
        guard context.device.supportsFamily(.apple10) else { return }
        let fixture = Fixture(start: c.start, chunk: c.chunk, seed: 0xB256 + UInt64(c.start))
        let first = try fixture.run(context: context, path: .fullTensorOps2DPreferred)
        let second = try fixture.run(context: context, path: .fullTensorOps2DPreferred)
        let reference = fixture.reference()
        let maxAbs = RelError.maxAbsDiff(first, reference)
        let rel = RelError.compute(actual: first, reference: reference)
        #expect(first == second, "not byte-stable at start=\(c.start) chunk=\(c.chunk)")
        #expect(maxAbs <= 2e-2, "maxAbs=\(maxAbs) rel=\(rel) start=\(c.start) chunk=\(c.chunk)")
        #expect(rel <= 2e-2, "rel=\(rel) maxAbs=\(maxAbs) start=\(c.start) chunk=\(c.chunk)")
    }

    /// Long chunks are dispatched in bounded query spans so no single
    /// dispatch can run past the driver's interactivity watchdog. A small
    /// cap makes the split exercisable at test size.
    @Test func boundedQuerySpansAreExact() throws {
        let context = try MetalContext()
        guard context.device.supportsFamily(.apple10) else { return }
        let fixture = Fixture(start: 100, chunk: 200, seed: 0xB257)
        let split = try fixture.run(context: context, path: .fullTensorOps2DPreferred,
                                    maxQueriesPerDispatch: 64)
        let whole = try fixture.run(context: context, path: .fullTensorOps2DPreferred)
        let reference = fixture.reference()
        #expect(RelError.maxAbsDiff(split, reference) <= 2e-2)
        #expect(RelError.maxAbsDiff(split, whole) <= 1e-3)
    }

    private struct Fixture {
        let headDim = 256
        let qHeads = 16
        let kvHeads = 2
        let start: Int
        let chunk: Int
        let kvValid: Int
        let qStride: Int
        let kvStride: Int
        let oStride: Int
        let scale: Float = 0.0625
        var q: [Float]
        var k: [Float]
        var v: [Float]

        init(start: Int, chunk: Int, seed: UInt64) {
            self.start = start
            self.chunk = chunk
            self.kvValid = start + chunk
            self.qStride = qHeads * headDim
            self.kvStride = kvHeads * headDim
            self.oStride = qHeads * headDim
            var rng = SeedTree(seed).key("hd256-start\(start)-chunk\(chunk)")
            q = [Float](repeating: 0, count: chunk * qStride)
            k = [Float](repeating: 0, count: kvValid * kvStride)
            v = [Float](repeating: 0, count: kvValid * kvStride)
            for i in q.indices { q[i] = rng.uniform(-0.35, 0.35) }
            for i in k.indices { k[i] = rng.uniform(-0.35, 0.35) }
            for i in v.indices { v[i] = rng.uniform(-0.35, 0.35) }
        }

        var params: PrefillAttentionParams {
            PrefillAttentionParams(
                startPosition: UInt32(start), queryCount: UInt32(chunk),
                headDim: UInt32(headDim), numQHeads: UInt32(qHeads),
                numKVHeads: UInt32(kvHeads), kvValidCount: UInt32(kvValid),
                slidingWindow: UInt32(kvValid),
                kvTokenStrideElements: UInt32(kvStride),
                qTokenStrideElements: UInt32(qStride),
                oTokenStrideElements: UInt32(oStride),
                scale: scale)
        }

        func run(context: MetalContext,
                 path: RuntimePrefillAttentionPath,
                 maxQueriesPerDispatch: Int? = nil) throws -> [Float] {
            let prefill = try PrefillAttention(context: context)
            if let maxQueriesPerDispatch { prefill.maxQueriesPerDispatch = maxQueriesPerDispatch }
            // The key/value buffers are over-allocated past kvValid, as the
            // runtime's cache is, and the slack is filled with NaN so any
            // read past the valid length shows up in the output.
            let slack = 256
            let kPadded = k + [Float](repeating: .nan, count: slack * kvStride)
            let vPadded = v + [Float](repeating: .nan, count: slack * kvStride)
            let qPadded = q + [Float](repeating: .nan, count: 64 * qStride)
            guard let qBuf = Fp16Buffer.make(context.device, values: qPadded),
                  let kBuf = Fp16Buffer.make(context.device, values: kPadded),
                  let vBuf = Fp16Buffer.make(context.device, values: vPadded),
                  let outBuf = Fp16Buffer.make(context.device, count: chunk * oStride) else {
                Issue.record("alloc failed")
                return []
            }
            let cb = context.queue.makeCommandBuffer()!
            prefill.encodeCausal(commandBuffer: cb, q: qBuf, k: kBuf, v: vBuf, out: outBuf,
                                 params: params, kvRingCapacity: 0, path: path)
            cb.commit()
            cb.waitUntilCompleted()
            return Fp16Buffer.read(outBuf, count: chunk * oStride)
        }

        func reference() -> [Float] {
            var out = [Float](repeating: 0, count: chunk * oStride)
            let qPerKV = qHeads / kvHeads
            for t in 0..<chunk {
                let last = start + t + 1
                for qh in 0..<qHeads {
                    let kvh = qh / qPerKV
                    var scores = [Float](repeating: 0, count: last)
                    var maxScore: Float = -.infinity
                    for key in 0..<last {
                        var s: Float = 0
                        for d in 0..<headDim {
                            s += q[t * qStride + qh * headDim + d]
                                * k[key * kvStride + kvh * headDim + d]
                        }
                        scores[key] = s * scale
                        maxScore = max(maxScore, scores[key])
                    }
                    var denom: Float = 0
                    for key in 0..<last {
                        scores[key] = Foundation.exp(scores[key] - maxScore)
                        denom += scores[key]
                    }
                    for d in 0..<headDim {
                        var acc: Float = 0
                        for key in 0..<last {
                            acc += scores[key] * v[key * kvStride + kvh * headDim + d]
                        }
                        out[t * oStride + qh * headDim + d] = acc / denom
                    }
                }
            }
            return out
        }
    }
}
