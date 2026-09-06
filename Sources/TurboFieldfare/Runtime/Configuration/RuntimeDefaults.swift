/// The one place a runtime tuning default is written down.
///
/// These describe what the HARDWARE wants, so every surface — server, CLI,
/// Mac app, decode service — must agree on them. Product defaults
/// (temperature, max context, top-k/top-p) deliberately differ per surface
/// and are NOT here: the server answers coding agents at temperature 0 while
/// the app is a chat UI at 0.2, and collapsing those would be a bug, not a
/// fix.
///
/// WHY THIS EXISTS: `expertCacheSlots` was declared with a literal default in
/// six places. Raising it 16 -> 64 on measured evidence changed two of them,
/// and the other four kept shipping 16 — including the Mac app's persisted
/// settings, which meant the app silently ignored the change entirely. The
/// build was green throughout, because nothing compared the surfaces.
///
/// A second benefit, specific to this codebase: referencing a constant means
/// changing a VALUE no longer edits any parameter list, so it cannot change a
/// mangled symbol. Defaulted-parameter changes have broken cross-module
/// callers here twice.
///
/// `RuntimeDefaultsDriftTests` asserts every surface resolves to these.
public enum RuntimeDefaults {
    /// Routed-expert slots per layer.
    ///
    /// 64 measured optimal on the M5 24 GB host: 27.76 tok/s against 25.12 at
    /// 16, from a 6-point sweep at 3k with the drift control at 2.2%, and
    /// +10.7% confirmed on a real agent workload. Deliberately NOT the best
    /// hit rate — 192 slots hits 97.7% against 64's 78.7% and is the SLOWEST
    /// arm, because past ~64 this cache evicts the OS page cache that was
    /// absorbing its own misses. The two compete for the same RAM, so the
    /// optimum belongs to the host, not the model. Re-measure on new hardware
    /// rather than assuming this transfers.
    public static let expertCacheSlots = 64

    /// Prefill chunk size for surfaces that do not compute one per prompt.
    public static let prefillChunkTokens = 128

    public static let prefillEnabled = true

    /// LFU with its use counts halved every 32 plans. Monotonic LFU keeps
    /// counts for the life of the process, so on a long session experts that
    /// were hot early stay resident after the work moves on. Chosen
    /// 2026-09-05 from an offline replay of routing traces: on a ten-turn
    /// coding session at 64 slots, LFU 98.8 misses per token, LRU 77.2, this
    /// 71.8; on a single 3k prompt 68.6, 67.2, 62.4. Fewer misses is the
    /// whole gain where every miss is an SSD read; on a host whose page
    /// cache holds part of the pool the leftover misses cost more each, and
    /// LRU measured 9 percent slower there despite 22 percent fewer misses.
    public static let expertCachePolicy: RuntimeExpertCachePolicy = .lfuAging

    public static let rdadvisePolicy: RDAdvicePolicyMode = .off

    /// Keep the GPU clocked across decode's expert-read gaps. `auto` defers to
    /// macOS Low Power Mode. Measured 2026-09-05 on the M5 at 16 slots with
    /// every expert read from the SSD: decode kernels 28 -> 13 ms per token
    /// and 9.5 -> 12.6 tok/s, output byte-identical. Where the read gaps are
    /// short, a warm 64-slot machine, the clock is already up and the hold
    /// costs one idle threadgroup.
    public static let gpuClockHold: RuntimeGPUClockHold = .auto
}
