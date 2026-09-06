import Foundation
import TurboFieldfare

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let port: Int
    /// Explicit --model-id value; nil defers to the loaded model's family
    /// default (gemma-4-26b-a4b-it or qwen3.6-35b-a3b).
    public let modelIDOverride: String?
    public var modelID: String { modelIDOverride ?? "gemma-4-26b-a4b-it" }
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    /// Routed-expert slots per layer. Exposed because it is the largest
    /// decode dial the server has and the optimum is machine-specific: it is
    /// set by RAM competition between this cache and the OS page cache, not
    /// by hit rate, so it has to be measured per host rather than assumed.
    public let expertCacheSlots: Int
    /// Replacement policy for those slots. Monotonic LFU keeps counts for the
    /// life of the process, so on a long session experts that were hot early
    /// stay resident after they stop being used; LRU adapts. Replayed on a
    /// recorded ten-turn coding session (2026-09-05): LFU 69.1 percent hit at
    /// 64 slots, LRU 75.9.
    public let expertCachePolicy: RuntimeExpertCachePolicy

    public static let usage = """
    usage: TurboFieldfareServer --model <completed .gturbo directory> [options]

      --model <dir>          Required model directory.
      --port <1...65535>     Loopback port (default 8080).
      --model-id <id>        API model identifier (default derived from the
                             installed model: gemma-4-26b-a4b-it or
                             qwen3.6-35b-a3b).
      --max-context <tokens> 4096, 8192, 16384, 32768, or 65536 (default 16384).
      --queue-limit <count>  Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix>
                             Prompt KV reuse mode (default single-prefix).
      --expert-cache-slots <n>
                             Routed-expert slots per layer (default 64).
                             More is not always faster: past the point where
                             this cache starts evicting the OS page cache,
                             throughput falls even as hit rate rises.
      --expert-cache-policy <lfu|lru>
                             Replacement policy for the slots (default lfu).
      --help                 Show this help.
    """

    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var port = 8080
        var modelIDOverride: String?
        var maxContext = 16_384
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .singlePrefix
        var expertCacheSlots = RuntimeConfiguration().expertCacheSlots
        var expertCachePolicy = RuntimeDefaults.expertCachePolicy
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelIDOverride = value
            case "--max-context":
                guard let parsed = Int(value),
                      [4_096, 8_192, 16_384, 32_768, 65_536].contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off or single-prefix")
                }
                promptCacheMode = parsed
            case "--expert-cache-slots":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--expert-cache-slots is not supported")
                }
                expertCacheSlots = parsed
            case "--expert-cache-policy":
                guard let parsed = RuntimeExpertCachePolicy(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--expert-cache-policy must be lfu or lru")
                }
                expertCachePolicy = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        return ServerArguments(model: model,
                               port: port,
                               modelIDOverride: modelIDOverride,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode,
                               expertCacheSlots: expertCacheSlots,
                               expertCachePolicy: expertCachePolicy)
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
