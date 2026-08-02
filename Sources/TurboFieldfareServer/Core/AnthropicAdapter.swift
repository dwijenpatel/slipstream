import Foundation
import TurboFieldfare

/// Anthropic Messages API (/v1/messages) adapter: translates requests into
/// the server's OpenAI-shaped internal path and completions back out, so
/// Anthropic-speaking clients (Claude Code) drive the same validated
/// pipeline — template, Qwen tool parsing, prompt cache — as everyone else.
/// Modeled on oMLX's adapter (Apache 2.0): same normalization rules,
/// including merging inline system-role messages (Claude Code 2.1.154+
/// sends system content in messages[] instead of the system field).
/// v1 scope: text, tool_use, tool_result, thinking (dropped), and
/// string/block system content. Images and documents are rejected.

// MARK: - Request models

public struct AnthropicMessagesRequest: Decodable, Sendable {
    public let model: String
    public let maxTokens: Int
    public let system: AnthropicSystem?
    public let messages: [AnthropicMessage]
    public let tools: [AnthropicTool]?
    public let stream: Bool?
    public let temperature: Float?
    public let topP: Float?
    public let topK: Int?
    public let stopSequences: [String]?

    enum CodingKeys: String, CodingKey {
        case model, system, messages, tools, stream, temperature
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case stopSequences = "stop_sequences"
    }
}

public enum AnthropicSystem: Decodable, Sendable {
    case text(String)
    case blocks([AnthropicContentBlock])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) { self = .text(s); return }
        self = .blocks(try c.decode([AnthropicContentBlock].self))
    }

    var flattened: String {
        switch self {
        case .text(let s): return s
        case .blocks(let blocks):
            return blocks.compactMap { block in
                if case .text(let t) = block { return t }
                return nil
            }.joined(separator: "\n")
        }
    }
}

public struct AnthropicMessage: Decodable, Sendable {
    public let role: String
    public let content: AnthropicMessageContent
}

public enum AnthropicMessageContent: Decodable, Sendable {
    case text(String)
    case blocks([AnthropicContentBlock])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) { self = .text(s); return }
        self = .blocks(try c.decode([AnthropicContentBlock].self))
    }
}

public enum AnthropicContentBlock: Decodable, Sendable {
    case text(String)
    case toolUse(id: String, name: String, input: JSONValue)
    case toolResult(toolUseID: String, content: String, isError: Bool)
    case thinking(String)
    case unsupported(type: String)

    enum CodingKeys: String, CodingKey {
        case type, text, id, name, input, thinking, content
        case toolUseID = "tool_use_id"
        case isError = "is_error"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let type = try c.decode(String.self, forKey: .type)
        switch type {
        case "text":
            self = .text(try c.decode(String.self, forKey: .text))
        case "tool_use":
            self = .toolUse(id: try c.decode(String.self, forKey: .id),
                            name: try c.decode(String.self, forKey: .name),
                            input: try c.decode(JSONValue.self, forKey: .input))
        case "tool_result":
            // content may be a string, a block list, or absent.
            var text = ""
            if let s = try? c.decode(String.self, forKey: .content) {
                text = s
            } else if let blocks = try? c.decode([AnthropicContentBlock].self,
                                                 forKey: .content) {
                text = blocks.compactMap { block in
                    if case .text(let t) = block { return t }
                    return nil
                }.joined(separator: "\n")
            }
            self = .toolResult(
                toolUseID: try c.decode(String.self, forKey: .toolUseID),
                content: text,
                isError: (try? c.decode(Bool.self, forKey: .isError)) ?? false)
        case "thinking":
            self = .thinking((try? c.decode(String.self, forKey: .thinking)) ?? "")
        default:
            self = .unsupported(type: type)
        }
    }
}

public struct AnthropicTool: Decodable, Sendable {
    public let name: String
    public let description: String?
    public let inputSchema: JSONValue

    enum CodingKeys: String, CodingKey {
        case name, description
        case inputSchema = "input_schema"
    }
}

// MARK: - Translation

public enum AnthropicAdapterError: Error {
    case unsupported(String)
}

public enum AnthropicAdapter {

    /// Anthropic request -> the server's OpenAI-shaped request. Throws on
    /// content the pipeline cannot represent (images, documents).
    public static func toOpenAI(_ request: AnthropicMessagesRequest,
                                modelID: String) throws -> OpenAIChatRequest {
        var messages: [OpenAIChatMessage] = []

        // System: canonical field first, then any inline system-role
        // messages appended in order (Claude Code 2.1.154+ behavior).
        var systemParts: [String] = []
        if let system = request.system {
            let text = system.flattened
            if !text.isEmpty { systemParts.append(text) }
        }
        var body: [AnthropicMessage] = []
        for message in request.messages {
            if message.role == "system" {
                systemParts.append(flattenText(message.content))
            } else {
                body.append(message)
            }
        }
        if !systemParts.isEmpty {
            messages.append(OpenAIChatMessage(role: "system",
                                              content: .text(systemParts.joined(separator: "\n\n")),
                                              toolCalls: nil, toolCallID: nil, name: nil))
        }

        for message in body {
            switch message.content {
            case .text(let text):
                messages.append(OpenAIChatMessage(role: message.role,
                                                  content: .text(text),
                                                  toolCalls: nil, toolCallID: nil, name: nil))
            case .blocks(let blocks):
                try appendBlocks(blocks, role: message.role, into: &messages)
            }
        }

        let tools: [OpenAITool]? = request.tools.map { list in
            list.map { tool in
                OpenAITool(type: "function",
                           function: OpenAIFunctionDefinition(
                               name: tool.name,
                               description: tool.description,
                               parameters: tool.inputSchema))
            }
        }

        return OpenAIChatRequest(model: modelID,
                                 messages: messages,
                                 stream: request.stream,
                                 streamOptions: nil,
                                 temperature: request.temperature,
                                 topP: request.topP,
                                 maxTokens: request.maxTokens,
                                 maxCompletionTokens: nil,
                                 stop: request.stopSequences.map { .many($0) },
                                 seed: nil,
                                 tools: tools,
                                 toolChoice: nil,
                                 parallelToolCalls: nil,
                                 topK: request.topK,
                                 repetitionPenalty: nil,
                                 n: nil,
                                 logprobs: nil,
                                 presencePenalty: nil,
                                 frequencyPenalty: nil)
    }

    private static func flattenText(_ content: AnthropicMessageContent) -> String {
        switch content {
        case .text(let s): return s
        case .blocks(let blocks):
            return blocks.compactMap { block in
                if case .text(let t) = block { return t }
                return nil
            }.joined(separator: "\n")
        }
    }

    private static func appendBlocks(_ blocks: [AnthropicContentBlock],
                                     role: String,
                                     into messages: inout [OpenAIChatMessage]) throws {
        var textParts: [String] = []
        var toolCalls: [OpenAIToolCall] = []
        var toolResults: [(id: String, content: String, isError: Bool)] = []

        for block in blocks {
            switch block {
            case .text(let t):
                textParts.append(t)
            case .toolUse(let id, let name, let input):
                let argsData = (try? JSONEncoder().encode(input)) ?? Data("{}".utf8)
                toolCalls.append(OpenAIToolCall(
                    id: id,
                    type: "function",
                    function: .init(name: name,
                                    arguments: String(decoding: argsData, as: UTF8.self))))
            case .toolResult(let id, let content, let isError):
                toolResults.append((id, content, isError))
            case .thinking:
                // Replayed reasoning from earlier turns; the template
                // regenerates its own. Dropped by design (oMLX fallback
                // inlines <think> tags; Qwen re-derives either way).
                continue
            case .unsupported(let type):
                throw AnthropicAdapterError.unsupported(
                    "content block type \(type) is not supported by this server")
            }
        }

        // Tool results become role:"tool" messages (one per result), before
        // any accompanying user text.
        for result in toolResults {
            let content = result.isError ? "ERROR: \(result.content)" : result.content
            messages.append(OpenAIChatMessage(role: "tool",
                                              content: .text(content),
                                              toolCalls: nil,
                                              toolCallID: result.id,
                                              name: nil))
        }
        if !textParts.isEmpty || !toolCalls.isEmpty {
            messages.append(OpenAIChatMessage(
                role: role,
                content: textParts.isEmpty ? nil : .text(textParts.joined(separator: "\n")),
                toolCalls: toolCalls.isEmpty ? nil : toolCalls,
                toolCallID: nil,
                name: nil))
        }
    }

    // MARK: Response translation

    public static func stopReason(fromFinishReason reason: String) -> String {
        switch reason {
        case "tool_calls": return "tool_use"
        case "length": return "max_tokens"
        default: return "end_turn"
        }
    }

    /// Non-streaming response body.
    public static func responseObject(id: String,
                                      model: String,
                                      completion: ServerCompletion) -> [String: Any] {
        var content: [[String: Any]] = []
        if !completion.content.isEmpty {
            content.append(["type": "text", "text": completion.content])
        }
        for call in completion.toolCalls {
            content.append(["type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": parsedArguments(call.argumentsJSON)])
        }
        return [
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stopReason(fromFinishReason: completion.finishReason),
            "stop_sequence": NSNull(),
            "usage": ["input_tokens": completion.usage.promptTokens,
                      "output_tokens": completion.usage.completionTokens],
        ]
    }

    static func parsedArguments(_ arguments: String) -> Any {
        guard let data = arguments.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) else {
            return [:] as [String: Any]
        }
        return obj
    }

    // MARK: Streaming events

    public static func sse(_ event: String, _ payload: [String: Any]) -> String {
        var object = payload
        object["type"] = event
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            return ""
        }
        return "event: \(event)\ndata: \(String(decoding: data, as: UTF8.self))\n\n"
    }

    public static func messageStart(id: String, model: String, inputTokens: Int) -> String {
        sse("message_start", [
            "message": ["id": id, "type": "message", "role": "assistant",
                        "model": model, "content": [] as [Any],
                        "stop_reason": NSNull(), "stop_sequence": NSNull(),
                        "usage": ["input_tokens": inputTokens, "output_tokens": 0]],
        ])
    }

    public static func textBlockStart(index: Int) -> String {
        sse("content_block_start",
            ["index": index, "content_block": ["type": "text", "text": ""]])
    }

    public static func textDelta(index: Int, text: String) -> String {
        sse("content_block_delta",
            ["index": index, "delta": ["type": "text_delta", "text": text]])
    }

    public static func toolUseBlock(index: Int, id: String, name: String,
                                    argumentsJSON: String) -> String {
        sse("content_block_start",
            ["index": index,
             "content_block": ["type": "tool_use", "id": id, "name": name,
                               "input": [:] as [String: Any]]])
            + sse("content_block_delta",
                  ["index": index,
                   "delta": ["type": "input_json_delta",
                             "partial_json": argumentsJSON]])
            + blockStop(index: index)
    }

    public static func blockStop(index: Int) -> String {
        sse("content_block_stop", ["index": index])
    }

    public static func messageDelta(stopReason: String, outputTokens: Int) -> String {
        sse("message_delta",
            ["delta": ["stop_reason": stopReason, "stop_sequence": NSNull()],
             "usage": ["output_tokens": outputTokens]])
    }

    public static func messageStop() -> String {
        sse("message_stop", [:])
    }

    public static func errorBody(_ message: String,
                                 type: String = "invalid_request_error") -> [String: Any] {
        ["type": "error", "error": ["type": type, "message": message]]
    }
}

/// Thread-safe content-block index tracking for the Anthropic stream: one
/// text block that opens lazily on the first content delta, then tool_use
/// blocks appended after it.
final class AnthropicBlockState: @unchecked Sendable {
    private let lock = NSLock()
    private var textOpen = false
    private var index = -1

    var currentIndex: Int { lock.withLock { max(index, 0) } }

    /// Returns true when this call opened the text block.
    func openTextBlockIfNeeded() -> Bool {
        lock.withLock {
            guard !textOpen else { return false }
            textOpen = true
            index += 1
            return true
        }
    }

    /// Returns the closed block's index, or nil when no text block is open.
    func closeTextBlock() -> Int? {
        lock.withLock {
            guard textOpen else { return nil }
            textOpen = false
            return index
        }
    }

    func nextBlockIndex() -> Int {
        lock.withLock {
            index += 1
            return index
        }
    }
}
