import Testing
@testable import TurboFieldfare

@Suite struct RuntimeConfigurationTests {
    @Test func productionDefaultsAreStable() {
        let runtime = RuntimeConfiguration.production
        #expect(runtime.fp16RingEnabled)
        // 64, not the 16 this pinned for months: measured 27.76 tok/s against
        // 25.12 at 16 (6-point sweep, drift control 2.2%). Deliberately NOT
        // the best hit rate — 192 hits 97.7% and is the slowest arm, because
        // past ~64 this cache evicts the page cache serving its own misses.
        #expect(runtime.expertCacheSlots == 64)
        // lfu-aging since 2026-09-05: LFU with counts halved every 32 plans.
        // Replayed routing traces at 64 slots put plain LFU at 98.8 misses
        // per token on a ten-turn session and aging at 71.8; on the single
        // 3k prompt 68.6 against 62.4.
        #expect(runtime.expertCachePolicy == .lfuAging)
        #expect(runtime.rdadvisePolicy == .off)
        #expect(!runtime.rdadviseEnabled)
        #expect(runtime.prefillPolicy == .chunked)
        #expect(runtime.prefillChunkTokens == 128)
        #expect(runtime.prefillAttentionPath == .fullTensorOps2DPreferred)
        #expect(runtime.headPath == .fusedRows)
    }

    @Test func retainedControlsReachTypedRuntime() {
        let runtime = RuntimeConfiguration(
            expertCacheSlots: 32,
            expertCachePolicy: .lru,
            rdadvisePolicy: .adaptive,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            prefillAttentionPath: .causalTiled,
            forceLogitsHead: true)
        #expect(runtime.expertCacheSlots == 32)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.rdadviseEnabled)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.prefillAttentionPath == .causalTiled)
        #expect(runtime.headPath == .logits)
    }

    @Test(arguments: [32, 64, 128, 256, 512, 1024, 2048, 4096])
    func productionPrefillSupportsPublicChunkSizes(_ chunkTokens: Int) {
        let runtime = RuntimeConfiguration(prefillChunkTokens: chunkTokens)
        #expect(runtime.prefillConfig.mode == .chunked)
        #expect(runtime.prefillConfig.chunkTokens == chunkTokens)
    }
}
