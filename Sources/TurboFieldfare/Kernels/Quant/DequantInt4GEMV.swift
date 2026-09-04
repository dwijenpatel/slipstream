import Metal

/// MLX-affine INT4 matrix-vector multiplication.
/// Eight SIMD groups process eight output rows per threadgroup.
final class DequantInt4GEMV {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
    }

    private static let rowsPerThreadgroup = 8
    private static let realDecodeShapes: [Shape] = [
        Shape(m: 4096, n: 2816),
        Shape(m: 2048, n: 2816),
        Shape(m: 2816, n: 4096),
        Shape(m: 8192, n: 2816),
        Shape(m: 1024, n: 2816),
        Shape(m: 2816, n: 8192),
    ]

    private let pipeline: MTLComputePipelineState
    private let specializedPipelines: [Shape: MTLComputePipelineState]
    private let multiRowPipeline: MTLComputePipelineState
    private let multiRowSpecializedPipelines: [Shape: MTLComputePipelineState]
    /// Activation rows one multi-row dispatch handles; the kernel keeps this
    /// many accumulators per lane.
    static let maxMultiRows = 8

    /// `additionalShapes` compiles extra constant-folded variants for a
    /// non-Gemma model's decode shapes. Measured: an unspecialized
    /// 4096x2048 GEMV runs at 102 GB/s, the specialized one at 141 GB/s.
    init(context: MetalContext,
         additionalShapes: [(m: Int, n: Int)] = []) throws {
        self.pipeline = try context.pipeline(
            "dequant_int4_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        let shapes = Self.realDecodeShapes
            + additionalShapes.map { Shape(m: UInt32($0.m), n: UInt32($0.n)) }
        var specializedPipelines: [Shape: MTLComputePipelineState] = [:]
        var multiRowSpecialized: [Shape: MTLComputePipelineState] = [:]
        for shape in shapes {
            let constants = [
                MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
                MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
                MetalFunctionConstant(index: 22, value: .bool(true)),
            ]
            specializedPipelines[shape] = try context.pipeline(
                "dequant_int4_gemv_simd",
                constants: constants,
                maxTotalThreadsPerThreadgroup: 512)
            multiRowSpecialized[shape] = try context.pipeline(
                "dequant_int4_gemv_multirow_simd",
                constants: constants,
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPipelines = specializedPipelines
        self.multiRowPipeline = try context.pipeline(
            "dequant_int4_gemv_multirow_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        self.multiRowSpecializedPipelines = multiRowSpecialized
    }

    /// `rows` activation rows at `xStrideElements` apart, one weight read per
    /// row of W, outputs at `yStrideElements` apart. Dispatches in groups of
    /// `maxMultiRows`; each output row is bit-identical to `encode` on that
    /// row alone.
    func encodeMultiRow(commandBuffer: MTLCommandBuffer,
                        weights: MTLBuffer,
                        weightsOffset: Int = 0,
                        scales: MTLBuffer,
                        scalesOffset: Int = 0,
                        biases: MTLBuffer,
                        biasesOffset: Int = 0,
                        x: MTLBuffer,
                        xOffset: Int = 0,
                        xStrideElements: Int,
                        y: MTLBuffer,
                        yOffset: Int = 0,
                        yStrideElements: Int,
                        rows: Int,
                        m: UInt32,
                        n: UInt32) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        precondition(weightsOffset % 2 == 0,
                     "dequant_int4_gemv_multirow_simd needs a 2-aligned weightsOffset, got \(weightsOffset)")
        precondition(rows > 0)
        let elementSize = MemoryLayout<Float16>.stride
        var first = 0
        while first < rows {
            let count = min(Self.maxMultiRows, rows - first)
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
            encoder.setComputePipelineState(
                multiRowSpecializedPipelines[Shape(m: m, n: n)] ?? multiRowPipeline)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(scales, offset: scalesOffset, index: 1)
            encoder.setBuffer(biases, offset: biasesOffset, index: 2)
            encoder.setBuffer(x, offset: xOffset + first * xStrideElements * elementSize, index: 3)
            encoder.setBuffer(y, offset: yOffset + first * yStrideElements * elementSize, index: 4)
            var mValue = m
            var nValue = n
            var tValue = UInt32(count)
            var xStride = UInt32(xStrideElements)
            var yStride = UInt32(yStrideElements)
            encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
            encoder.setBytes(&tValue, length: MemoryLayout<UInt32>.size, index: 7)
            encoder.setBytes(&xStride, length: MemoryLayout<UInt32>.size, index: 8)
            encoder.setBytes(&yStride, length: MemoryLayout<UInt32>.size, index: 9)
            encoder.dispatchThreadgroups(
                MTLSize(width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
                        height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                               height: 1, depth: 1))
            encoder.endEncoding()
            first += count
        }
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                biases: MTLBuffer,
                biasesOffset: Int = 0,
                x: MTLBuffer,
                xOffset: Int = 0,
                y: MTLBuffer,
                yOffset: Int = 0,
                m: UInt32,
                n: UInt32) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        // The kernel reads packed weights through a `ushort*`; the repacker
        // guarantees two-byte sub-tensor alignment but not four-byte alignment.
        precondition(weightsOffset % 2 == 0,
                     "dequant_int4_gemv_simd needs a 2-aligned weightsOffset, got \(weightsOffset)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            specializedPipelines[Shape(m: m, n: n)] ?? pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = m
        var nValue = n
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)

        let threadgroupSize = MTLSize(
            width: 32 * Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        encoder.dispatchThreadgroups(threadgroupCount,
                                     threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}
