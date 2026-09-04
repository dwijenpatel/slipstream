import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// A misspelled option used to be dropped by the synthesized decoder, so the
/// request ran under settings the caller did not ask for. Adapted from
/// upstream TurboFieldfare pull request 171.
@Suite("OpenAI request decoding")
struct OpenAIRequestDecodingTests {
    private typealias Rejection = (message: String, param: String?, code: String)

    private func decode(_ body: String) throws -> OpenAIChatRequest {
        try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(body.utf8))
    }

    private func request(_ fields: String) -> String {
        """
        {"model":"m","messages":[{"role":"user","content":"x"}],\(fields)}
        """
    }

    private func decodeRejection(_ body: String) -> Rejection? {
        do {
            _ = try decode(body)
            Issue.record("request decoded instead of failing")
            return nil
        } catch ServerRequestError.invalid(let message, let param, let code) {
            return (message, param, code)
        } catch {
            Issue.record("decoding threw \(error) rather than a ServerRequestError")
            return nil
        }
    }

    private func validationRejection(_ body: String) -> Rejection? {
        do {
            _ = try OpenAIRequestValidator.validate(try decode(body), modelID: "m")
            Issue.record("request validated instead of failing")
            return nil
        } catch ServerRequestError.invalid(let message, let param, let code) {
            return (message, param, code)
        } catch {
            Issue.record("validation threw \(error) rather than a ServerRequestError")
            return nil
        }
    }

    @Test func unknownKeyIsRefusedByName() throws {
        let refusal = try #require(decodeRejection(request(#""max_token":4"#)))
        #expect(refusal.message.contains("max_token"))
        #expect(refusal.param == "max_token")
        #expect(refusal.code == "unknown_parameter")
    }

    // The key sweep runs before the typed decode, so the answer names the key
    // the caller misspelled rather than whatever DecodingError another field
    // raises first.
    @Test func unknownKeyIsNamedEvenBesideAMistypedDeclaredField() throws {
        let refusal = try #require(
            decodeRejection(request(#""max_token":4,"temperature":"hot""#)))
        #expect(refusal.param == "max_token")
        #expect(refusal.code == "unknown_parameter")
    }

    @Test func severalUnknownKeysAreListedSorted() throws {
        let refusal = try #require(
            decodeRejection(request(#""zeta":1,"alpha":2"#)))
        #expect(refusal.message.contains("\"alpha\", \"zeta\""))
        #expect(refusal.param == "alpha")
    }

    @Test func unsupportedRealParameterIsRefusedAsUnsupportedNotUnknown() throws {
        let refusal = try #require(
            decodeRejection(request(#""logit_bias":{"1":2},"temperature":"hot""#)))
        #expect(refusal.message == "logit_bias is not supported")
        #expect(refusal.param == "logit_bias")
        #expect(refusal.code == "unsupported_value")
    }

    @Test func legacyFunctionsPointAtTools() throws {
        let refusal = try #require(decodeRejection(request(#""functions":[]"#)))
        #expect(refusal.message.contains("use tools"))
        #expect(refusal.code == "unsupported_value")
    }

    // openai-python serializes an option the caller never set as an explicit
    // `null`, which asks for nothing and must not be refused on presence.
    @Test(arguments: [
        #""logit_bias":null"#,
        #""web_search_options":null"#,
        #""foo":null"#,
        #""n":null"#,
        #""response_format":null"#,
    ])
    func aKeySetToNullIsTreatedAsAbsent(_ field: String) throws {
        let decoded = try decode(request(field))
        #expect(decoded.messages.count == 1)
    }

    @Test(arguments: ["user", "store", "metadata", "service_tier",
                      "prompt_cache_key", "safety_identifier"])
    func callerBookkeepingKeysAreAcceptedAndIgnored(_ key: String) throws {
        let decoded = try decode(request(#""\#(key)":"anything""#))
        #expect(decoded.model == "m")
    }

    @Test func unknownKeyNamesAreBoundedInTheRefusal() throws {
        let long = String(repeating: "k", count: 500)
        let refusal = try #require(decodeRejection(request(#""\#(long)":1"#)))
        #expect(refusal.message.count < 200)
        #expect((refusal.param ?? "").count <= 67)
    }

    @Test func responseFormatTextIsAccepted() throws {
        let validated = try OpenAIRequestValidator.validate(
            try decode(request(#""response_format":{"type":"text"}"#)), modelID: "m")
        #expect(validated.messages.count == 1)
    }

    @Test func responseFormatJSONObjectIsUnsupported() throws {
        let refusal = try #require(
            validationRejection(request(#""response_format":{"type":"json_object"}"#)))
        #expect(refusal.message.contains("structured output"))
        #expect(refusal.param == "response_format")
        #expect(refusal.code == "unsupported_value")
    }

    @Test(arguments: [#""json_object""#, "[]", "7", "true"])
    func nonObjectResponseFormatNamesTheField(_ value: String) throws {
        let refusal = try #require(
            validationRejection(request(#""response_format":\#(value)"#)))
        #expect(refusal.message.contains("must be an object"))
        #expect(refusal.param == "response_format")
        #expect(refusal.code == "invalid_value")
    }

    @Test(arguments: ["{}", #"{"type":null}"#])
    func responseFormatWithoutATypeIsInvalid(_ value: String) throws {
        let refusal = try #require(
            validationRejection(request(#""response_format":\#(value)"#)))
        #expect(refusal.message.contains("response_format.type is required"))
        #expect(refusal.code == "invalid_value")
    }

    @Test func declaredFieldsStillDecode() throws {
        let decoded = try decode(request(
            #""temperature":0.5,"top_p":0.9,"max_tokens":12,"stream":true,"seed":7"#))
        #expect(decoded.temperature == 0.5)
        #expect(decoded.topP == 0.9)
        #expect(decoded.maxTokens == 12)
        #expect(decoded.stream == true)
        #expect(decoded.seed == 7)
    }
}
