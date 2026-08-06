import Foundation
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// ChatML (Qwen) tool-result continuation. The suite above covers Gemma, whose
/// `<tool_response>` IS a stop token; ChatML's is not, so a tool-calling turn
/// there stops on `<|im_end|>` and reports `.endOfTurn`. That difference kept
/// the structural path unreachable for Qwen, which went unnoticed because the
/// exact-prefix fast path absorbed every request while thinking was disabled.
@Suite("Server prompt cache (ChatML)")
struct ServerPromptCacheChatMLTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "qwen3.6-35b-a3b",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 65_536,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    private typealias Message = GFTokenizer.Message

    /// The ChatML tokenizer fixture lives in the TurboFieldfare test target;
    /// reach it by path rather than duplicating a resource into this bundle.
    private static func chatMLFixtureFolder() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(
                "TurboFieldfare/Core/Tokenization/Fixtures/ChatMLTokenizer",
                isDirectory: true)
    }

    private let tools = [GFTokenizer.FunctionDefinition(
        name: "read",
        description: "read a file",
        parameters: .object(["type": .string("object")]))]

    private let assistant = Message(
        role: .assistant,
        content: nil,
        toolCalls: [GFTokenizer.HistoricalToolCall(
            id: "call_1",
            name: "read",
            arguments: .object(["path": .string("foo.txt")]))])

    @Test("A thinking tool-call turn continues from the KV instead of re-prefilling")
    func thinkingToolCallContinues() async throws {
        let tokenizer = try await GFTokenizer.load(from: Self.chatMLFixtureFolder())
        let cached = [Message(role: .user, content: "what is in foo.txt?")]
        let initial = request(messages: cached, tools: tools)
        let initialPrompt = try tokenizer.encodeToolChat(
            messages: cached, tools: tools)

        // What the model actually emits: the prompt has already opened
        // `<think>`, so reasoning arrives bare, then `</think>`, then the call.
        let prefix = try tokenizer.encodeToolChat(
            messages: cached + [assistant], tools: tools,
            addGenerationPrompt: false)
        let callStart = try #require(prefix.lastIndex(of: tokenizer.toolCallStartID))
        let callEnd = try #require(prefix.lastIndex(of: tokenizer.toolCallEndID))
        let reasoning = tokenizer.encode("I should read that file.", addBOS: false)
            + [try #require(tokenizer.thinkEndID)]
            + tokenizer.encode("\n\n", addBOS: false)
        let kvBacked = initialPrompt + reasoning + Array(prefix[callStart...callEnd])

        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "",
            calls: [ParsedToolCall(
                id: "call_1",
                name: "read",
                arguments: .object(["path": .string("foo.txt")]),
                argumentsJSON: try JSONValue
                    .object(["path": .string("foo.txt")]).encoded())],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                // ChatML stops a tool-calling turn on `<|im_end|>`.
                reason: .endOfTurn))
        #expect(cache.entry != nil, "publish rejected a well-formed ChatML turn")

        let continuation = request(
            messages: cached + [assistant, Message(
                role: .tool, content: "hello world", toolCallID: "call_1")],
            tools: tools)
        let rendered = try tokenizer.encodeToolChat(
            messages: continuation.messages, tools: continuation.tools)

        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cachedCount) = match else {
            Issue.record("expected a ChatML tool-result hit, got a miss")
            return
        }
        #expect(cachedCount == kvBacked.count)
        // The reasoning stays in the KV verbatim; only the bridge is prefilled.
        #expect(effective.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cachedCount] == tokenizer.endOfTurnID)
        #expect(effective.count > kvBacked.count)
        // The re-render cannot reproduce the KV, which is the whole point.
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
    }

    private func request(
        messages: [Message],
        tools: [GFTokenizer.FunctionDefinition]
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    private func rawResult(
        prompt: [Int32],
        kvBacked: [Int32],
        boundary: Int32,
        reason: StopReason
    ) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: prompt.count,
            cachedPromptTokens: 0,
            computedPrefillTokens: prompt.count,
            prefillSeconds: 0,
            prepareSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            reason: reason,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [boundary])
    }
}
