import Foundation
import Testing
@testable import TurboFieldfare

/// The budget is the number macOS actually kills on: physical footprint
/// against the GPU's working-set limit. Admission is decided before each
/// prefill chunk from the footprint now plus a predicted delta, so a prompt
/// that would push the process over the limit fails with a named cause
/// instead of a jetsam kill. Structure after oMLX's prefill guard.
@Suite("Memory budget")
struct MemoryBudgetTests {
    let gb: UInt64 = 1 << 30

    @Test func capIsNinetyPercentOfTheDeviceLimit() {
        let budget = MemoryBudget(deviceLimitBytes: 20 * gb)
        #expect(budget.capBytes == 18 * gb)
    }

    @Test func admitsWhenCurrentPlusPredictedFitsUnderTheCap() {
        let budget = MemoryBudget(deviceLimitBytes: 20 * gb)
        #expect(budget.admits(currentBytes: 17 * gb, predictedDeltaBytes: gb))
        #expect(!budget.admits(currentBytes: 17 * gb, predictedDeltaBytes: gb + 1))
    }

    @Test func largestSlotCountWhoseCeilingFits() {
        // 5.4 GB cap on an 8 GB machine; 1.53 GB fixed; 70.8 MB per slot step.
        let fixed: UInt64 = 1_530_000_000
        let perSlot: UInt64 = 40 * 1_769_472
        let fits = MemoryBudget(deviceLimitBytes: 6 * gb)
            .largestSlotCount(fixedBytes: fixed, bytesPerSlot: perSlot,
                              allowed: [8, 16, 24, 32, 48, 64, 96, 128, 192, 256])
        #expect(fits == 48)
    }

    @Test func noSlotCountFitsWhenTheFixedPartAloneExceedsTheCap() {
        let fits = MemoryBudget(deviceLimitBytes: 2 * gb)
            .largestSlotCount(fixedBytes: 3 * gb, bytesPerSlot: 1, allowed: [8, 16])
        #expect(fits == nil)
    }
}

@Suite("Prefill transient estimator")
struct PrefillTransientEstimatorTests {
    @Test func withNoSamplesThePredictionIsTheAnalyticFloorWithMargin() {
        let e = PrefillTransientEstimator()
        // 1,000 bytes per token floor, 100 tokens, 1.3 margin.
        #expect(e.predictedDeltaBytes(tokens: 100, floorBytesPerToken: 1_000) == 130_000)
    }

    @Test func aRecordedSampleRaisesThePredictionAboveTheFloor() {
        var e = PrefillTransientEstimator()
        e.record(deltaBytes: 4_000_000, tokens: 1_000) // 4,000 bytes per token
        #expect(e.predictedDeltaBytes(tokens: 10, floorBytesPerToken: 1_000) == 52_000)
    }

    @Test func theFloorWinsWhenTheMeasuredRateIsBelowIt() {
        var e = PrefillTransientEstimator()
        e.record(deltaBytes: 500, tokens: 1_000)
        #expect(e.predictedDeltaBytes(tokens: 10, floorBytesPerToken: 1_000) == 13_000)
    }

    @Test func aSingleOutlierDoesNotPoisonTheAverage() {
        var e = PrefillTransientEstimator()
        e.record(deltaBytes: 1_000_000, tokens: 1_000)   // 1,000 per token
        e.record(deltaBytes: 100_000_000, tokens: 1_000) // 100,000 per token: 100x
        #expect(e.bytesPerToken < 2_000)
    }

    @Test func aShrinkingFootprintRecordsAsZeroGrowth() {
        var e = PrefillTransientEstimator()
        e.record(deltaBytes: -5_000, tokens: 100)
        #expect(e.bytesPerToken == 0)
    }
}

@Suite("Prefill memory guard")
struct PrefillMemoryGuardTests {
    let gb: UInt64 = 1 << 30

    /// A scripted footprint reader: each read returns the next value.
    final class Footprints: @unchecked Sendable {
        var values: [UInt64]
        init(_ values: [UInt64]) { self.values = values }
        func read() -> UInt64? { values.isEmpty ? nil : values.removeFirst() }
    }

    @Test func refusesTheChunkThatWouldCrossTheCap() throws {
        let footprints = Footprints([UInt64(17.9 * Double(gb))])
        let guard_ = PrefillMemoryGuard(budget: MemoryBudget(deviceLimitBytes: 20 * gb),
                                        floorBytesPerToken: 1 << 20, // 1 MiB per token
                                        footprint: footprints.read)
        // 17.9 GB now plus 4096 tokens at 1 MiB each with margin is 23 GB.
        #expect(throws: MemoryBudgetError.self) {
            try guard_.beforeChunk(tokens: 4096)
        }
    }

    @Test func admitsAChunkThatFitsAndLearnsFromItsGrowth() throws {
        // Before: 10 GB. After: 10.5 GB for 1,000 tokens, about 0.5 MB per token.
        let footprints = Footprints([10 * gb, UInt64(10.5 * Double(gb)), UInt64(10.5 * Double(gb))])
        let guard_ = PrefillMemoryGuard(budget: MemoryBudget(deviceLimitBytes: 20 * gb),
                                        floorBytesPerToken: 1_000,
                                        footprint: footprints.read)
        try guard_.beforeChunk(tokens: 1_000)
        guard_.afterChunk(tokens: 1_000)
        #expect(guard_.estimator.bytesPerToken > 400_000)
        // Now a 20,000-token chunk at the learned rate is about 13 GB: refused.
        #expect(throws: MemoryBudgetError.self) {
            try guard_.beforeChunk(tokens: 20_000)
        }
    }

    @Test func anUnreadableFootprintAdmitsRatherThanBlocks() throws {
        let guard_ = PrefillMemoryGuard(budget: MemoryBudget(deviceLimitBytes: 20 * gb),
                                        floorBytesPerToken: 1 << 30,
                                        footprint: { nil })
        try guard_.beforeChunk(tokens: 4096)
    }

    @Test func theErrorNamesTheNumbersAndTheFix() {
        let error = MemoryBudgetError.wouldExceed(currentBytes: 17 * gb,
                                                  predictedDeltaBytes: 3 * gb,
                                                  capBytes: 18 * gb,
                                                  chunkTokens: 4096)
        let text = String(describing: error)
        #expect(text.contains("17.0 GB"))
        #expect(text.contains("3.0 GB"))
        #expect(text.contains("18.0 GB"))
        #expect(text.contains("4096"))
        #expect(text.contains("--expert-cache-slots"))
    }
}
