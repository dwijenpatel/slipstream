import Foundation
import Metal

/// Keeps the GPU clocked up across decode's expert-read gaps.
///
/// Measured 2026-09-05 on the M5 at 16 slots with every expert read from the
/// SSD: each layer's 1.5 ms read gap let the chip lower the GPU clock, and the
/// same decode kernels took 28 ms per token instead of the 13 ms they take on
/// a warm machine. One threadgroup of 32 threads looping on register
/// arithmetic, kept queued on a second command queue, restored 13 ms and
/// lifted decode from 9.5 to 12.6 tokens per second with byte-identical
/// output. Logs: `bench-results/gpu-heater-20260905`.
///
/// The hold runs while a deadline lies in the future. The decode step calls
/// `touch` to extend the deadline, so the hold covers the token loop of every
/// surface and stops on its own a fraction of a second after the last token,
/// including on a client disconnect or a stall. Prefill never touches it.
public final class GPUClockHold: @unchecked Sendable {
    /// About 0.7 ms of GPU time per command buffer on the M5: long enough
    /// that the hold thread commits about 1,500 buffers per second rather
    /// than 5,000, short enough that a passed deadline is noticed within a
    /// couple of milliseconds. 15,000 and 60,000 iterations measured the
    /// same decode gain on 2026-09-05.
    static let iterationsPerThread: UInt32 = 50_000
    static let commandBuffersInFlight = 2

    private let worker: GPUClockHoldWorker

    public init(device: MTLDevice, library: MTLLibrary) throws {
        worker = try GPUClockHoldWorker(device: device, library: library)
    }

    deinit {
        worker.stop()
    }

    /// Keep the GPU clocked for at least `seconds` from now. Later calls only
    /// ever push the deadline out.
    public func touch(seconds: Double) { worker.touch(seconds: seconds) }

    /// Ends the hold for good. Idempotent; also called when the owner is released.
    public func stop() { worker.stop() }

    public var isActive: Bool { worker.isActive }
    public var submittedCommandBuffers: UInt64 { worker.submittedCommandBuffers }

    /// Wall time during which the hold had a deadline ahead of it.
    public var activeSeconds: Double { worker.activeSeconds }
}

/// The thread retains only this state, so releasing GPUClockHold can always
/// stop and wake it, even while it waits indefinitely with no active deadline.
private final class GPUClockHoldWorker: @unchecked Sendable {
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let sink: MTLBuffer
    private let inFlight = DispatchSemaphore(value: GPUClockHold.commandBuffersInFlight)
    private let condition = NSCondition()
    private var deadlineNanos: UInt64 = 0
    private var stopped = false
    private var submitted: UInt64 = 0
    private var activeNanos: UInt64 = 0
    private var activeSinceNanos: UInt64?

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let queue = device.makeCommandQueue() else {
            throw MetalError.noQueue
        }
        queue.label = "gpu-clock-hold"
        guard let function = library.makeFunction(name: "gpu_clock_hold") else {
            throw MetalError.missingFunction("gpu_clock_hold")
        }
        guard let sink = device.makeBuffer(length: 16, options: .storageModePrivate) else {
            throw MetalError.missingShaderResource("gpu_clock_hold sink buffer")
        }
        self.queue = queue
        self.pipeline = try device.makeComputePipelineState(function: function)
        self.sink = sink
        let thread = Thread { [self] in
            while self.submitOneOrWait() {}
        }
        thread.name = "gpu-clock-hold"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    /// Keep the GPU clocked for at least `seconds` from now. Later calls only
    /// ever push the deadline out.
    func touch(seconds: Double) {
        let now = Self.now()
        let requested = now &+ UInt64(max(0, seconds) * 1e9)
        condition.lock()
        if !stopped {
            deadlineNanos = max(deadlineNanos, requested)
            if activeSinceNanos == nil { activeSinceNanos = now }
            condition.signal()
        }
        condition.unlock()
    }

    /// Ends the hold for good. Idempotent; called by the owner's `deinit`.
    func stop() {
        condition.lock()
        stopped = true
        closeActiveWindowLocked(at: Self.now())
        condition.broadcast()
        condition.unlock()
    }

    var isActive: Bool {
        condition.lock()
        defer { condition.unlock() }
        return !stopped && Self.now() < deadlineNanos
    }

    var submittedCommandBuffers: UInt64 {
        condition.lock()
        defer { condition.unlock() }
        return submitted
    }

    /// Wall time during which the hold had a deadline ahead of it.
    var activeSeconds: Double {
        condition.lock()
        defer { condition.unlock() }
        var nanos = activeNanos
        if let since = activeSinceNanos {
            nanos &+= Self.now() &- since
        }
        return Double(nanos) / 1e9
    }

    /// One turn of the hold thread: block while there is nothing to hold,
    /// otherwise queue one command buffer. Returns false once stopped.
    private func submitOneOrWait() -> Bool {
        condition.lock()
        while !stopped && Self.now() >= deadlineNanos {
            closeActiveWindowLocked(at: Self.now())
            condition.wait()
        }
        if stopped {
            condition.unlock()
            return false
        }
        condition.unlock()

        inFlight.wait()
        // `stop` may have landed while this turn waited for a slot; a buffer
        // committed after `stop` would make the count move after the owner
        // believed the hold was over.
        condition.lock()
        let stoppedWhileWaiting = stopped
        condition.unlock()
        guard !stoppedWhileWaiting,
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            inFlight.signal()
            return !stoppedWhileWaiting
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(sink, offset: 0, index: 0)
        var iterations = GPUClockHold.iterationsPerThread
        encoder.setBytes(&iterations, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        let inFlight = self.inFlight
        commandBuffer.addCompletedHandler { _ in inFlight.signal() }
        commandBuffer.commit()

        condition.lock()
        if !stopped { submitted &+= 1 }
        condition.unlock()
        return true
    }

    private func closeActiveWindowLocked(at now: UInt64) {
        if let since = activeSinceNanos {
            activeNanos &+= now &- since
            activeSinceNanos = nil
        }
    }

    private static func now() -> UInt64 {
        clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }
}
