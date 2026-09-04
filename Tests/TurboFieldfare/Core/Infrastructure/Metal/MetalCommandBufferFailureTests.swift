import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// A failed command buffer used to be printed and ignored, so a Metal
/// watchdog kill during a long prefill produced a wrong answer instead of an
/// error. The detail formatter is the part that can be tested without a GPU.
@Suite("Metal command buffer failure detail")
struct MetalCommandBufferFailureTests {
    @Test func completedBufferWithNoErrorHasNoDetail() {
        #expect(metalCommandBufferFailureDetail(label: "decode",
                                                status: .completed,
                                                error: nil) == nil)
    }

    @Test func errorStatusWithoutAnErrorObjectIsStillAFailure() throws {
        let detail = try #require(metalCommandBufferFailureDetail(label: "prefill",
                                                                  status: .error,
                                                                  error: nil))
        #expect(detail.contains("label=prefill"))
        #expect(detail.contains("status=error"))
        #expect(detail.contains("error=<none>"))
    }

    @Test func errorObjectContributesDomainCodeAndDescription() throws {
        let error = NSError(domain: "MTLCommandBufferErrorDomain",
                            code: 1,
                            userInfo: [NSLocalizedDescriptionKey: "Impacting Interactivity"])
        let detail = try #require(metalCommandBufferFailureDetail(label: nil,
                                                                  status: .error,
                                                                  error: error))
        #expect(detail.contains("label=<none>"))
        #expect(detail.contains("domain=MTLCommandBufferErrorDomain"))
        #expect(detail.contains("code=1"))
        #expect(detail.contains("Impacting Interactivity"))
    }

    @Test func completedBufferThatStillCarriesAnErrorIsReported() throws {
        let error = NSError(domain: "MTLCommandBufferErrorDomain", code: 7, userInfo: [:])
        let detail = try #require(metalCommandBufferFailureDetail(label: "cb1",
                                                                  status: .completed,
                                                                  error: error))
        #expect(detail.contains("status=completed"))
        #expect(detail.contains("code=7"))
    }
}
