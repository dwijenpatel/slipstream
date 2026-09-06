public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable, CaseIterable {
    case lfu
    case lru
    case lfuAging = "lfu-aging"
}

/// Whether decode keeps one threadgroup looping on the GPU so the chip does
/// not lower the GPU clock across expert-read gaps. `auto` defers to macOS
/// Low Power Mode, which is the user's explicit signal to save energy.
public enum RuntimeGPUClockHold: String, Codable, Sendable, CaseIterable {
    case auto
    case on
    case off

    public func holds(lowPowerMode: Bool) -> Bool {
        switch self {
        case .auto: return !lowPowerMode
        case .on: return true
        case .off: return false
        }
    }
}

public struct RuntimeConfiguration: Sendable, Equatable {
    // Uncapped beyond the upstream 32: slots x layers x expertStride is the
    // dominant memory block, and on machines with headroom more resident
    // experts directly cut decode expert-I/O await (the measured decode
    // bottleneck). 256 covers every expert of every layer on current models.
    public static let allowedExpertCacheSlots = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    public static let allowedPrefillChunkTokens = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath
    public let gpuClockHold: RuntimeGPUClockHold

    public init(expertCacheSlots: Int = RuntimeDefaults.expertCacheSlots,
                expertCachePolicy: RuntimeExpertCachePolicy = RuntimeDefaults.expertCachePolicy,
                rdadvisePolicy: RDAdvicePolicyMode = RuntimeDefaults.rdadvisePolicy,
                prefillEnabled: Bool = RuntimeDefaults.prefillEnabled,
                prefillChunkTokens: Int = RuntimeDefaults.prefillChunkTokens,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                forceLogitsHead: Bool = false,
                gpuClockHold: RuntimeGPUClockHold = RuntimeDefaults.gpuClockHold) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.gpuClockHold = gpuClockHold
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        switch expertCachePolicy {
        case .lfu: return .lfu
        case .lru: return .lru
        case .lfuAging: return .lfuAging
        }
    }
}
