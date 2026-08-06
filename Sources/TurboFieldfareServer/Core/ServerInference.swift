import CryptoKit
import Foundation
import TurboFieldfare

public enum ServerInferenceEvent: Equatable, Sendable {
    case content(String)
    case toolCall(ParsedToolCall)
}

public struct ServerCompletion: Equatable, Sendable {
    public let content: String
    public let toolCalls: [ParsedToolCall]
    public let finishReason: String
    public let usage: OpenAIUsage

    public init(content: String,
                toolCalls: [ParsedToolCall],
                finishReason: String,
                usage: OpenAIUsage) {
        self.content = content
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
    }
}

public protocol ServerInferenceBackend: Sendable {
    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
}

public actor ServerCoordinator {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private let queueLimit: Int
    private var active = false
    private var waiters: [Waiter] = []
    private var shuttingDown = false

    public init(queueLimit: Int) {
        self.queueLimit = queueLimit
    }

    public func run<T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await acquire(onQueued: onQueued)
        defer { release() }
        return try await operation()
    }

    private func acquire(onQueued: @escaping @Sendable () -> Void) async throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        if !active {
            active = true
            return
        }
        guard waiters.count < queueLimit else { throw ServerRequestError.queueFull }
        onQueued()
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func release() {
        if waiters.isEmpty {
            active = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { active }
}

public actor ServerModelSession: ServerInferenceBackend {
    /// Chat dialect of the loaded tokenizer; drives request-validation rules.
    public nonisolated let chatDialect: ChatDialect
    /// Family-derived API model identifier used when --model-id is absent.
    public nonisolated var defaultModelID: String {
        switch modelFamily {
        case .gemma4: return "gemma-4-26b-a4b-it"
        case .qwen36: return "qwen3.6-35b-a3b"
        }
    }
    private nonisolated let modelFamily: ModelFamily

    private let context: MetalContext
    private let model: Model
    private let tokenizer: GFTokenizer
    private let runner: RealForwardRunner
    private let scratch: RawCompletionScratch
    private let prefillConfig: PrefillRuntimeConfig
    private let maxContext: Int
    private let promptCacheMode: ServerPromptCacheMode
    private let promptCacheDomain: ServerPromptCacheDomain
    private var promptCache = ServerPromptCache()

    public static func load(modelDirectory: URL,
                            maxContext: Int,
                            promptCacheMode: ServerPromptCacheMode = .singlePrefix) async throws -> ServerModelSession {
        let tokenizerFolder = GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw GFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            throw GFTokenizerError.missingToolTemplate
        }
        let tokenizer = try await GFTokenizer.load(from: tokenizerFolder)
        let context = try MetalContext()
        // Prefill in the largest supported chunk. The CLI computes this per
        // prompt ("auto"), which the server cannot do because the runtime is
        // built before any request arrives; but the allowed sizes top out at
        // 4096, so auto resolves to the maximum for every prompt larger than
        // that, and the maximum is within one step of auto for the rest.
        // Leaving this at the struct default of 128 cost 64 s of the 87.8 s a
        // 2,940-token prompt took through the server, against 36.7 s for the
        // same work through the CLI (measured 2026-08-06).
        // The fused head is now selected per request from the request's own
        // sampling config (see RawCompletion), so the server no longer has to
        // give it up wholesale to be able to serve temperature > 0.
        let runtime = RuntimeConfiguration(
            prefillChunkTokens: RuntimeConfiguration.allowedPrefillChunkTokens.last!,
            forceLogitsHead: false)
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let runner = try RealForwardRunner(model: model,
                                           context: context,
                                           maxContext: maxContext,
                                           runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let templateDigest = SHA256.hash(data: try Data(contentsOf: templateURL))
            .map { String(format: "%02x", $0) }
            .joined()
        let runtimeIdentity = [
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            runtime.prefillPolicy.rawValue,
            String(runtime.prefillChunkTokens),
            runtime.headPath.rawValue,
        ].joined(separator: ":")
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: PrefillKVStorageMode.fp16.rawValue,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        return ServerModelSession(context: context,
                                  model: model,
                                  tokenizer: tokenizer,
                                  runner: runner,
                                  scratch: scratch,
                                  prefillConfig: runtime.prefillConfig,
                                  maxContext: maxContext,
                                  promptCacheMode: promptCacheMode,
                                  promptCacheDomain: promptCacheDomain)
    }

    private init(context: MetalContext,
                 model: Model,
                 tokenizer: GFTokenizer,
                 runner: RealForwardRunner,
                 scratch: RawCompletionScratch,
                 prefillConfig: PrefillRuntimeConfig,
                 maxContext: Int,
                 promptCacheMode: ServerPromptCacheMode,
                 promptCacheDomain: ServerPromptCacheDomain) {
        self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.chatDialect = tokenizer.dialect
        self.modelFamily = model.config.family
        self.runner = runner
        self.scratch = scratch
        self.prefillConfig = prefillConfig
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.promptCacheDomain = promptCacheDomain
    }

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        var completed = false
        defer {
            if !completed {
                promptCache.invalidate()
                runner.reset()
            }
        }
        let needsToolTemplate = !request.tools.isEmpty
            || request.messages.contains {
                $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
            }
        let promptIDs: [Int32]
        if needsToolTemplate {
            promptIDs = try tokenizer.encodeToolChat(messages: request.messages, tools: request.tools)
        } else {
            let rendered = try tokenizer.applyChatTemplate(request.messages)
            promptIDs = tokenizer.encode(rendered, addBOS: false)
        }
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        let effectivePromptIDs: [Int32]
        let completionStart: RawCompletionStart
        if promptCacheMode == .singlePrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                request: request,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer) {
            case .miss:
                promptCache.invalidate()
                effectivePromptIDs = promptIDs
                completionStart = .reset
            case .hit(let effective, let cached):
                effectivePromptIDs = effective
                completionStart = .resume(cachedPromptTokens: cached)
            }
        } else {
            promptCache.invalidate()
            effectivePromptIDs = promptIDs
            completionStart = .reset
        }
        guard effectivePromptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        var config = request.generationConfig
        config.maxNewTokens = min(
            request.maximumCompletionTokens,
            maxContext - effectivePromptIDs.count)
        config.stopStrings = []

        let decoder = needsToolTemplate
            ? StructuredAssistantDecoder(
                tokenizer: tokenizer,
                allowedTools: Set(request.tools.map(\.name)))
            : nil
        var stopMatcher = StreamingStopMatcher(stops: request.generationConfig.stopStrings)
        var content = ""
        var calls: [ParsedToolCall] = []
        var decodingError: Error?
        var shouldStop = false

        let result = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: effectivePromptIDs,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: completionStart,
            shouldStop: { shouldStop }) { progress in
                guard decodingError == nil else { return }
                do {
                    switch progress {
                    case .prefill:
                        break
                    case .token(_, let tokenID, let delta):
                        let events = if let decoder {
                            try decoder.consume(tokenID: tokenID, delta: delta)
                        } else {
                            delta.isEmpty ? [] : [StructuredAssistantEvent.content(delta)]
                        }
                        for event in events {
                            switch event {
                            case .content(let text):
                                let visible = stopMatcher.push(text)
                                if !visible.isEmpty {
                                    content += visible
                                    onEvent(.content(visible))
                                }
                                if stopMatcher.isStopped { shouldStop = true }
                            case .toolCall(let call):
                                calls.append(call)
                                onEvent(.toolCall(call))
                            }
                        }
                    case .tail(let text):
                        let visible = stopMatcher.push(text)
                        if !visible.isEmpty {
                            content += visible
                            onEvent(.content(visible))
                        }
                    }
                } catch {
                    decodingError = error
                    shouldStop = true
                }
        }
        if let decodingError { throw decodingError }
        try decoder?.finish()
        if needsToolTemplate, result.reason == .toolCalls, calls.isEmpty {
            throw GemmaToolCallParserError.malformed
        }
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
        let reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if result.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        // One line per request, so a long agent run measures itself instead of
        // being reconstructed afterwards from trajectory timestamps. Written
        // to stderr, which is where the server's other diagnostics go.
        let ttft = result.prefillSeconds
        let decodeRate = result.decodeSeconds > 0
            ? Double(result.newTokens) / result.decodeSeconds
            : 0
        let prefillRate = ttft > 0 ? Double(result.computedPrefillTokens) / ttft : 0
        FileHandle.standardError.write(Data((
            "request prompt=\(result.prefillTokens) cached=\(result.cachedPromptTokens) "
            + "new_prompt=\(result.computedPrefillTokens) completion=\(result.newTokens) "
            + String(format: "ttft=%.2fs prefill=%.1ftok/s decode=%.1ftok/s total=%.2fs",
                     ttft, prefillRate, decodeRate, ttft + result.decodeSeconds)
            + " stop=\(reason)\n").utf8))

        if promptCacheMode == .singlePrefix {
            promptCache.publish(
                domain: promptCacheDomain,
                request: request,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopMatcher.isStopped)
        }
        completed = true
        return ServerCompletion(
            content: content,
            toolCalls: calls,
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: result.prefillTokens,
                               completionTokens: result.newTokens,
                               totalTokens: result.prefillTokens + result.newTokens,
                               cachedTokens: result.cachedPromptTokens))
    }
}
