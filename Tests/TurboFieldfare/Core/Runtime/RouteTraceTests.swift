import Foundation
import Testing
@testable import TurboFieldfare

/// A routing trace records, per token and layer, which experts the router
/// chose. That sequence is all any cache policy depends on, so one trace
/// replays every policy at every slot count offline (`playbook/route_replay.py`).
@Suite("Routing trace")
struct RouteTraceTests {
    private func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("route-trace-\(UUID().uuidString).bin")
    }

    @Test func recordsRoundTripThroughTheFile() throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let trace = try RouteTrace(url: url, numLayers: 40, topK: 8, numExperts: 256)
        trace.record(phase: .prefill, position: 0, layer: 0, experts: [1, 2, 3, 4, 5, 6, 7, 8])
        trace.record(phase: .prefill, position: 1, layer: 39, experts: [255, 0, 9, 9, 9, 9, 9, 9])
        trace.record(phase: .decode, position: 2, layer: 7, experts: [10, 20, 30, 40, 50, 60, 70, 80])
        trace.close()

        let read = try RouteTrace.read(url: url)
        #expect(read.header.numLayers == 40)
        #expect(read.header.topK == 8)
        #expect(read.header.numExperts == 256)
        #expect(read.records.count == 3)
        #expect(read.records[0] == RouteTrace.Record(phase: .prefill, position: 0, layer: 0,
                                                     experts: [1, 2, 3, 4, 5, 6, 7, 8]))
        #expect(read.records[1].layer == 39)
        #expect(read.records[1].experts.first == 255)
        #expect(read.records[2].phase == .decode)
        #expect(read.records[2].position == 2)
    }

    @Test func recordsSurviveWithoutAnExplicitClose() throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        do {
            let trace = try RouteTrace(url: url, numLayers: 2, topK: 2, numExperts: 4)
            for position in 0..<1000 {
                trace.record(phase: .decode, position: position, layer: 1, experts: [3, 0])
            }
        }
        let read = try RouteTrace.read(url: url)
        #expect(read.records.count == 1000)
        #expect(read.records[999].position == 999)
    }

    @Test func rejectsTheWrongNumberOfExperts() throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let trace = try RouteTrace(url: url, numLayers: 1, topK: 8, numExperts: 256)
        trace.record(phase: .decode, position: 0, layer: 0, experts: [1, 2, 3])
        trace.close()
        #expect(try RouteTrace.read(url: url).records.isEmpty)
    }

    @Test func environmentVariableSelectsTheFile() throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let trace = try RouteTrace.fromEnvironment(["TURBO_FIELDFARE_ROUTE_TRACE": url.path],
                                                   numLayers: 3, topK: 8, numExperts: 256)
        #expect(trace != nil)
        #expect(try RouteTrace.fromEnvironment([:], numLayers: 3, topK: 8, numExperts: 256) == nil)
    }
}
