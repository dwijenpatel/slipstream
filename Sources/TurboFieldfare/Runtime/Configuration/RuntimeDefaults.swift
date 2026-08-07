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

    public static let expertCachePolicy: RuntimeExpertCachePolicy = .lfu

    public static let rdadvisePolicy: RDAdvicePolicyMode = .off
}
