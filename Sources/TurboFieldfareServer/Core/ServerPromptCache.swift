import Foundation
import TurboFieldfare

public enum ServerPromptCacheMode: String, Sendable, Equatable {
    case off
    case singlePrefix = "single-prefix"
}

struct ServerPromptCacheDomain: Sendable, Equatable {
    let modelID: String
    let sourceSnapshotHash: String?
    let runtimeProfileHash: String
    let maximumContext: Int
    let kvStorage: String
    let fp16RingEnabled: Bool
    let templateSHA256: String
}

struct CachedAssistantTurn: Sendable, Equatable {
    let message: GFTokenizer.Message
    let rawStopReason: StopReason
}

struct ServerPromptCacheEntry: Sendable, Equatable {
    let domain: ServerPromptCacheDomain
    let inputMessages: [GFTokenizer.Message]
    let tools: [GFTokenizer.FunctionDefinition]
    let assistantTurn: CachedAssistantTurn
    let kvBackedTokenIDs: [Int32]
    let uncommittedBoundaryTokenIDs: [Int32]
    let kvPosition: Int
}

/// Why a lookup missed. A miss is only legitimate when the KV genuinely cannot
/// serve the request; anything else is a bug that shows up as a full re-prefill
/// and is otherwise indistinguishable from correct behavior. Logging the reason
/// is what makes the difference visible — `historyDiverged` and
/// `assistantDiverged` on a steady agent loop mean something is wrong.
enum ServerPromptCacheMiss: String, Sendable, Equatable {
    /// Nothing cached yet, or the last turn was not publishable.
    case noEntry
    /// Model, runtime, or template changed under us.
    case domainChanged
    /// The client altered its tool definitions.
    case toolsChanged
    /// The entry's own KV invariants do not hold.
    case entryUnusable
    /// The replayed history is not an extension of what we served.
    case historyDiverged
    /// The client sent back a different assistant turn than we generated.
    case assistantDiverged
    /// Continuation is not the tool-result or single-user shape we can bridge.
    case continuationShape
    /// The template could not produce a bridge from the cached boundary.
    case bridgeUnavailable
}

enum ServerPromptCacheMatch: Sendable, Equatable {
    case miss(ServerPromptCacheMiss)
    case hit(effectivePromptIDs: [Int32], cachedPromptTokens: Int)
}

struct ServerPromptCache: Sendable {
    private(set) var entry: ServerPromptCacheEntry?

    mutating func invalidate() {
        entry = nil
    }

    mutating func publish(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        stopStringFiltered: Bool = false
    ) {
        guard result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              !stopStringFiltered,
              result.reason == .endOfTurn
                || result.reason == .toolCalls
                || result.reason == .maxTokens else {
            entry = nil
            return
        }
        let historicalCalls = calls.map {
            GFTokenizer.HistoricalToolCall(
                id: $0.id,
                name: $0.name,
                arguments: $0.arguments)
        }
        // Keep the content even alongside tool calls. Discarding it made
        // assistantMatches unable to compare it, so it demanded both sides be
        // empty and every turn carrying prose AND a tool call fell through to
        // a full re-prefill.
        let assistant = GFTokenizer.Message(
            role: .assistant,
            content: content.isEmpty ? nil : content,
            toolCalls: historicalCalls)
        entry = ServerPromptCacheEntry(
            domain: domain,
            inputMessages: request.messages,
            tools: request.tools,
            assistantTurn: CachedAssistantTurn(
                message: assistant,
                rawStopReason: result.reason),
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            kvPosition: result.kvPosition)
    }

    func match(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        guard let entry else { return .miss(.noEntry) }
        guard entry.domain == domain else { return .miss(.domainChanged) }
        guard entry.tools == request.tools else { return .miss(.toolsChanged) }
        guard entry.kvPosition == entry.kvBackedTokenIDs.count,
              entry.kvPosition > 0,
              entry.uncommittedBoundaryTokenIDs.count == 1 else {
            return .miss(.entryUnusable)
        }

        if renderedPromptIDs.count > entry.kvPosition,
           renderedPromptIDs.prefix(entry.kvPosition)
            .elementsEqual(entry.kvBackedTokenIDs) {
            return .hit(
                effectivePromptIDs: renderedPromptIDs,
                cachedPromptTokens: entry.kvPosition)
        }

        let inputCount = entry.inputMessages.count
        guard request.messages.count > inputCount + 1,
              request.messages.prefix(inputCount)
                .elementsEqual(entry.inputMessages) else {
            return .miss(.historyDiverged)
        }
        guard assistantMatches(
                request.messages[inputCount],
                entry.assistantTurn.message) else {
            return .miss(.assistantDiverged)
        }
        let continuation = Array(request.messages.dropFirst(inputCount + 1))

        if entry.assistantTurn.message.toolCalls.isEmpty {
            return matchTextContinuation(
                entry: entry,
                continuation: continuation,
                tokenizer: tokenizer)
        }
        return matchToolContinuation(
            entry: entry,
            request: request,
            continuation: continuation,
            tokenizer: tokenizer)
    }

    private func assistantMatches(
        _ incoming: GFTokenizer.Message,
        _ cached: GFTokenizer.Message
    ) -> Bool {
        guard incoming.role == .assistant,
              cached.role == .assistant,
              incoming.toolCalls == cached.toolCalls,
              incoming.toolCallID == cached.toolCallID,
              incoming.name == cached.name else {
            return false
        }
        // Compare content in both cases. A turn may legitimately carry prose
        // and a tool call together, and with thinking enabled that is the
        // common shape rather than the exception.
        return (incoming.content ?? "") == (cached.content ?? "")
    }

    private func matchTextContinuation(
        entry: ServerPromptCacheEntry,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        guard continuation.count == 1,
              continuation[0].role == .user,
              let content = continuation[0].content,
              continuation[0].toolCalls.isEmpty,
              continuation[0].toolCallID == nil,
              entry.assistantTurn.rawStopReason == .endOfTurn
                || entry.assistantTurn.rawStopReason == .maxTokens else {
            return .miss(.continuationShape)
        }
        var bridge = tokenizer.encodeTextContinuation(userContent: content)
        if entry.assistantTurn.rawStopReason == .maxTokens {
            bridge = entry.uncommittedBoundaryTokenIDs + bridge
        } else if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
            return .miss(.bridgeUnavailable)
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }

    private func matchToolContinuation(
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        let calls = entry.assistantTurn.message.toolCalls
        // `.toolCalls` is Gemma reporting its `<tool_response>` stop token.
        // ChatML has no such token in its stop set: a tool-calling turn there
        // ends on `<|im_end|>` and reports `.endOfTurn`, so requiring
        // `.toolCalls` alone made this path unreachable for Qwen. Both mean the
        // same thing here — a turn that ended cleanly holding tool calls.
        guard entry.assistantTurn.rawStopReason == .toolCalls
                || entry.assistantTurn.rawStopReason == .endOfTurn,
              continuation.count == calls.count,
              zip(continuation, calls).allSatisfy({ message, call in
                  message.role == .tool
                    && message.toolCallID == call.id
                    && (message.name == nil || message.name == call.name)
                    && message.content != nil
                    && message.toolCalls.isEmpty
              }) else {
            return .miss(.continuationShape)
        }
        guard let bridge = try? tokenizer.encodeToolResultContinuation(
            cachedMessages: entry.inputMessages,
            assistant: entry.assistantTurn.message,
            incomingMessages: request.messages,
            tools: request.tools),
              bridge.first == entry.uncommittedBoundaryTokenIDs.first else {
            return .miss(.bridgeUnavailable)
        }
        return .hit(
            effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
            cachedPromptTokens: entry.kvPosition)
    }
}
