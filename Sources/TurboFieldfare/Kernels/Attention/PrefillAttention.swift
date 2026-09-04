import Foundation
import Metal

struct PrefillAttentionParams: Sendable, Equatable {
    var startPosition: UInt32
    var queryCount: UInt32
    var headDim: UInt32
    var numQHeads: UInt32
    var numKVHeads: UInt32
    var kvValidCount: UInt32
    var slidingWindow: UInt32
    var kvTokenStrideElements: UInt32
    var qTokenStrideElements: UInt32
    var oTokenStrideElements: UInt32
    var scale: Float

    init(startPosition: UInt32,
                queryCount: UInt32,
                headDim: UInt32,
                numQHeads: UInt32,
                numKVHeads: UInt32,
                kvValidCount: UInt32,
                slidingWindow: UInt32,
                kvTokenStrideElements: UInt32,
                qTokenStrideElements: UInt32,
                oTokenStrideElements: UInt32,
                scale: Float) {
        self.startPosition = startPosition
        self.queryCount = queryCount
        self.headDim = headDim
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.kvValidCount = kvValidCount
        self.slidingWindow = slidingWindow
        self.kvTokenStrideElements = kvTokenStrideElements
        self.qTokenStrideElements = qTokenStrideElements
        self.oTokenStrideElements = oTokenStrideElements
        self.scale = scale
    }
}


final class PrefillAttention {
    /// Query tile and key tile of the head-dim-256 tensor-ops kernel. The
    /// kernel overlaps its final tiles instead of guarding reads past the
    /// end, so it needs at least one full tile of each.
    static let hd256QueryTile = 32
    static let hd256KeyTile = 128

    private let context: MetalContext
    private let psoCausalTiled: MTLComputePipelineState
    private let psoFullTensorOps2DValidityV2: MTLComputePipelineState?
    private let psoCausalTensorOpsHD256: MTLComputePipelineState?

    /// Longest query span handed to one tensor-ops dispatch. Bounded so no
    /// single dispatch can run long enough to trip the driver's interactivity
    /// watchdog on a long prompt; the spans are exact by construction.
    var maxQueriesPerDispatch = 2048

    var hd256TensorOpsAvailable: Bool { psoCausalTensorOpsHD256 != nil }

    init(context: MetalContext) throws {
        self.context = context
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
        let tensorOps = context.device.supportsFamily(.apple10)
        self.psoFullTensorOps2DValidityV2 = tensorOps
            ? try? context.pipeline("attention_prefill_full_tensorops_2d_validity_v2")
            : nil
        self.psoCausalTensorOpsHD256 = tensorOps
            ? try? context.pipeline("attention_prefill_causal_tensorops_hd256")
            : nil
    }

    /// Whether the head-dim-256 tensor-ops kernel applies. It is a causal,
    /// full-visibility, linear-KV kernel for grouped-query attention, and the
    /// chunk must be at least one query tile against at least one key tile.
    static func selectsHD256TensorOps(params: PrefillAttentionParams,
                                      kvRingCapacity: UInt32,
                                      path: RuntimePrefillAttentionPath) -> Bool {
        let requestsTensorOps = path == .fullTensorOps2DPreferred
            || path == .fullTensorOps2DValidityV2
        let windowNeverClips = params.slidingWindow == 0
            || params.slidingWindow >= params.kvValidCount
        return requestsTensorOps
            && kvRingCapacity == 0
            && params.headDim == 256
            && params.numKVHeads > 0
            && params.numQHeads % params.numKVHeads == 0
            && windowNeverClips
            && params.startPosition + params.queryCount == params.kvValidCount
            && params.queryCount >= UInt32(hd256QueryTile)
            && params.kvValidCount >= UInt32(hd256KeyTile)
    }

    func encodeCausal(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int = 0,
                             k: MTLBuffer, kOffset: Int = 0,
                             v: MTLBuffer, vOffset: Int = 0,
                             out: MTLBuffer, outOffset: Int = 0,
                             params: PrefillAttentionParams,
                             kvRingCapacity: UInt32 = 0,
                             path: RuntimePrefillAttentionPath = .causalTiled) {
        validate(params)

        if let pso = psoCausalTensorOpsHD256,
           Self.selectsHD256TensorOps(params: params, kvRingCapacity: kvRingCapacity, path: path) {
            encodeHD256Spans(pipeline: pso, commandBuffer: commandBuffer,
                             q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                             v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                             params: params)
            return
        }

        let requestsTensorOps = path == .fullTensorOps2DPreferred
            || path == .fullTensorOps2DValidityV2
        // The pinned model uses 512/16/2 only for full attention; its
        // sliding-window layers use 256/16/8. A future model that reuses this
        // shape for sliding attention must add a full-visibility check here.
        let tensorOpsShape = requestsTensorOps
            && kvRingCapacity == 0
            && params.headDim == 512
            && params.numQHeads == 16
            && params.numKVHeads == 2
            && params.scale == 1.0
        let tensorOpsPipeline = tensorOpsShape ? psoFullTensorOps2DValidityV2 : nil
        let useTensorOps = tensorOpsPipeline != nil
        let pipeline: MTLComputePipelineState
        if let tensorOpsPipeline {
            pipeline = tensorOpsPipeline
        } else if tensorOpsShape && path == .fullTensorOps2DValidityV2 {
            preconditionFailure(
                "TensorOps 2D prefill attention requires Apple10 MPP tensor support")
        } else {
            // Explicit mode also falls back for incompatible shapes. Benchmark
            // fixtures must use 512/16/2 to prove that TensorOps ran.
            pipeline = causalTiledPipeline(kvRingCapacity: kvRingCapacity)
        }
        let headDim = Int(params.headDim)
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        let threadCount = useTensorOps
            ? 128
            : roundUp(max(threadWidth, headDim), toMultipleOf: threadWidth)
        precondition(threadCount <= pipeline.maxTotalThreadsPerThreadgroup,
                     "tiled prefill attention requires headDim <= maxTotalThreadsPerThreadgroup")

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        let groups = useTensorOps
            ? MTLSize(width: Int(params.queryCount),
                      height: Int(params.numQHeads) / 8,
                      depth: 1)
            : MTLSize(width: Int(params.queryCount),
                      height: Int(params.numQHeads),
                      depth: 1)
        enc.dispatchThreadgroups(
            groups,
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        enc.endEncoding()
    }


    /// Splits the chunk into query spans of at most `maxQueriesPerDispatch`
    /// rows. Each span is a complete causal problem of its own: its queries
    /// are the last `n` of the `startPosition + offset + n` visible keys, so
    /// the kernel's position arithmetic holds and no key is masked twice. A
    /// trailing span shorter than a query tile goes to the scalar kernel,
    /// which is exact for any size.
    private func encodeHD256Spans(pipeline: MTLComputePipelineState,
                                  commandBuffer: MTLCommandBuffer,
                                  q: MTLBuffer, qOffset: Int,
                                  k: MTLBuffer, kOffset: Int,
                                  v: MTLBuffer, vOffset: Int,
                                  out: MTLBuffer, outOffset: Int,
                                  params: PrefillAttentionParams) {
        let total = Int(params.queryCount)
        let cap = max(Self.hd256QueryTile, maxQueriesPerDispatch)
        let elementSize = MemoryLayout<Float16>.size
        var offset = 0
        while offset < total {
            let n = min(cap, total - offset)
            var span = params
            span.startPosition = params.startPosition + UInt32(offset)
            span.queryCount = UInt32(n)
            span.kvValidCount = span.startPosition + span.queryCount
            span.slidingWindow = span.kvValidCount
            let spanQOffset = qOffset + offset * Int(params.qTokenStrideElements) * elementSize
            let spanOOffset = outOffset + offset * Int(params.oTokenStrideElements) * elementSize
            if n < Self.hd256QueryTile {
                encodeScalar(pipeline: psoCausalTiled, commandBuffer: commandBuffer,
                             q: q, qOffset: spanQOffset, k: k, kOffset: kOffset,
                             v: v, vOffset: vOffset, out: out, outOffset: spanOOffset,
                             params: span)
            } else {
                guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(q, offset: spanQOffset, index: 0)
                enc.setBuffer(k, offset: kOffset, index: 1)
                enc.setBuffer(v, offset: vOffset, index: 2)
                enc.setBuffer(out, offset: spanOOffset, index: 3)
                var p = span
                enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
                let queryTiles = (n + Self.hd256QueryTile - 1) / Self.hd256QueryTile
                enc.dispatchThreadgroups(
                    MTLSize(width: queryTiles * Int(params.numQHeads), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }
            offset += n
        }
    }

    private func encodeScalar(pipeline: MTLComputePipelineState,
                              commandBuffer: MTLCommandBuffer,
                              q: MTLBuffer, qOffset: Int,
                              k: MTLBuffer, kOffset: Int,
                              v: MTLBuffer, vOffset: Int,
                              out: MTLBuffer, outOffset: Int,
                              params: PrefillAttentionParams) {
        let headDim = Int(params.headDim)
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        let threadCount = roundUp(max(threadWidth, headDim), toMultipleOf: threadWidth)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(params.queryCount), height: Int(params.numQHeads), depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func validate(_ params: PrefillAttentionParams) {
        precondition(params.headDim > 0, "headDim must be positive")
        precondition(params.queryCount > 0, "queryCount must be positive")
        precondition(params.numQHeads > 0, "numQHeads must be positive")
        precondition(params.numKVHeads > 0, "numKVHeads must be positive")
        precondition(params.numQHeads % params.numKVHeads == 0,
                     "numQHeads must be divisible by numKVHeads")
        precondition(params.qTokenStrideElements >= params.numQHeads * params.headDim,
                     "q token stride is too small")
        precondition(params.oTokenStrideElements >= params.numQHeads * params.headDim,
                     "output token stride is too small")
        precondition(params.kvTokenStrideElements >= params.numKVHeads * params.headDim,
                     "KV token stride is too small")
        precondition(params.startPosition + params.queryCount <= params.kvValidCount,
                     "kvValidCount must include all in-flight query rows")
    }


    private func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }

    private func causalTiledPipeline(kvRingCapacity: UInt32) -> MTLComputePipelineState {
        guard kvRingCapacity > 0 else { return psoCausalTiled }
        do {
            return try context.pipeline(
                "attention_prefill_causal_tiled",
                constants: [MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity))])
        } catch {
            preconditionFailure("failed to build FP16 KV ring prefill attention pipeline: \(error)")
        }
    }
}
