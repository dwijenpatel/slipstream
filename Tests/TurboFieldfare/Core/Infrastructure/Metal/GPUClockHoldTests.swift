import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Measured 2026-09-05 on the M5 at 16 slots with every expert read from the
/// SSD: the chip lowers the GPU clock during each 1.5 ms read gap, so the same
/// decode kernels took 28 ms per token instead of 13. One threadgroup kept
/// looping on the GPU restored 13 ms and lifted decode 9.5 -> 12.6 tok/s with
/// byte-identical output. The hold is that threadgroup, scoped by a deadline
/// the decode step keeps extending.
@Suite("GPU clock hold")
struct GPUClockHoldTests {
    @Test func policyDecidesFromLowPowerMode() {
        #expect(RuntimeGPUClockHold.auto.holds(lowPowerMode: false))
        #expect(!RuntimeGPUClockHold.auto.holds(lowPowerMode: true))
        #expect(RuntimeGPUClockHold.on.holds(lowPowerMode: true))
        #expect(RuntimeGPUClockHold.on.holds(lowPowerMode: false))
        #expect(!RuntimeGPUClockHold.off.holds(lowPowerMode: false))
        #expect(!RuntimeGPUClockHold.off.holds(lowPowerMode: true))
    }

    @Test func touchedHoldSubmitsWorkAndStopsAfterItsDeadline() throws {
        let context = try MetalContext()
        let hold = try GPUClockHold(device: context.device, library: context.library)
        defer { hold.stop() }
        #expect(!hold.isActive)
        #expect(hold.submittedCommandBuffers == 0)

        hold.touch(seconds: 0.15)
        #expect(hold.isActive)
        Thread.sleep(forTimeInterval: 0.1)
        #expect(hold.submittedCommandBuffers > 0)

        // The deadline passed at 0.15 s; the loop must notice on its own.
        Thread.sleep(forTimeInterval: 0.3)
        #expect(!hold.isActive)
        let settled = hold.submittedCommandBuffers
        Thread.sleep(forTimeInterval: 0.1)
        #expect(hold.submittedCommandBuffers == settled)
        #expect(hold.activeSeconds > 0.1)
        #expect(hold.activeSeconds < 0.4)
    }

    @Test func touchExtendsTheDeadlineInsteadOfShorteningIt() throws {
        let context = try MetalContext()
        let hold = try GPUClockHold(device: context.device, library: context.library)
        defer { hold.stop() }
        hold.touch(seconds: 0.5)
        hold.touch(seconds: 0.05)
        Thread.sleep(forTimeInterval: 0.15)
        #expect(hold.isActive)
    }

    @Test func stopIsPromptWhileActive() throws {
        let context = try MetalContext()
        let hold = try GPUClockHold(device: context.device, library: context.library)
        hold.touch(seconds: 5)
        Thread.sleep(forTimeInterval: 0.05)
        hold.stop()
        #expect(!hold.isActive)
        let settled = hold.submittedCommandBuffers
        Thread.sleep(forTimeInterval: 0.1)
        #expect(hold.submittedCommandBuffers == settled)
    }
}
