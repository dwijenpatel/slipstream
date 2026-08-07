import Foundation
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// The completion cap when a client sends none.
///
/// The captured-fixture tests do NOT pin this: those requests carry an
/// explicit max_tokens, so they passed unchanged while the default moved
/// 4096 -> 8192 -> 4096. A default nothing asserts is a default that drifts,
/// which is exactly how the expert-cache slot count ended up declared six
/// different ways.
@Suite("Default completion budget")
struct CompletionBudgetDefaultTests {
    private func validated(_ json: String) throws -> ValidatedChatRequest {
        let request = try JSONDecoder().decode(
            OpenAIChatRequest.self, from: Data(json.utf8))
        return try OpenAIRequestValidator.validate(
            request, modelID: "qwen3.6-35b-a3b")
    }

    /// 4096 rather than 8192. Measured 2026-08-06: normal agent turns peak
    /// near 1k tokens, and one turn that reasoned all the way to an 8192 cap
    /// produced no tool call at all, was discarded by the client, and cost 39%
    /// of that run's wall clock. Raising the cap does not rescue such a turn,
    /// it only makes it more expensive.
    @Test("A request with no limit gets 4096")
    func defaultBudget() throws {
        let v = try validated("""
        {"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}]}
        """)
        #expect(v.maximumCompletionTokens == 4096)
    }

    @Test("An explicit limit still wins")
    func explicitBudgetWins() throws {
        let v = try validated("""
        {"model":"qwen3.6-35b-a3b","max_tokens":128,
         "messages":[{"role":"user","content":"hi"}]}
        """)
        #expect(v.maximumCompletionTokens == 128)
    }
}
