import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Server arguments")
struct ServerArgumentsTests {
    @Test func expertCachePolicyDefaultsToTheRuntimeDefault() throws {
        let arguments = try ServerArguments.parse(["--model", "m.gturbo"])
        #expect(arguments.expertCachePolicy == RuntimeDefaults.expertCachePolicy)
    }

    @Test func expertCachePolicyParsesBothPolicies() throws {
        #expect(try ServerArguments.parse(["--model", "m.gturbo", "--expert-cache-policy", "lru"])
            .expertCachePolicy == .lru)
        #expect(try ServerArguments.parse(["--model", "m.gturbo", "--expert-cache-policy", "lfu"])
            .expertCachePolicy == .lfu)
    }

    @Test func expertCachePolicyRejectsAnUnknownValue() {
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model", "m.gturbo", "--expert-cache-policy", "warm"])
        }
    }
}
