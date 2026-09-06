import Testing
@testable import TurboFieldfare

/// The prediction and prefetch diagnostics used to be switched on by the CLI
/// alone, so a server started with TURBO_FIELDFARE_PREFETCH=1 ran without it
/// and the overnight session arm measured nothing (smoke run, 2026-09-06).
/// Both surfaces now read the same switches from the environment.
@Suite("Runtime diagnostic switches")
struct RuntimeDiagnosticSwitchesTests {
    @Test func nothingSetMeansNothingOn() {
        let s = RuntimeDiagnosticSwitches(environment: [:])
        #expect(!s.predictRouting)
        #expect(!s.prefetchExperts)
        #expect(s.prefetchDistance == 1)
        #expect(s.predictRoutingDistances == [1])
    }

    @Test func prefetchImpliesPredictionAtItsDistance() {
        let s = RuntimeDiagnosticSwitches(environment: [
            "TURBO_FIELDFARE_PREFETCH": "1",
            "TURBO_FIELDFARE_PREFETCH_DISTANCE": "2",
        ])
        #expect(s.predictRouting)
        #expect(s.prefetchExperts)
        #expect(s.prefetchDistance == 2)
        #expect(s.predictRoutingDistances.contains(2))
    }

    @Test func distancesParseAndClampToTheSupportedRange() {
        let s = RuntimeDiagnosticSwitches(environment: [
            "TURBO_FIELDFARE_PRED_ROUTE_DISTANCES": "3, 1, 9, x",
            "TURBO_FIELDFARE_PREFETCH": "1",
            "TURBO_FIELDFARE_PREFETCH_DISTANCE": "7",
        ])
        #expect(s.predictRouting)
        #expect(s.prefetchDistance == RealForwardRunner.maxPredictionDistances)
        #expect(s.predictRoutingDistances.contains(1))
        #expect(s.predictRoutingDistances.contains(3))
        #expect(s.predictRoutingDistances.contains(RealForwardRunner.maxPredictionDistances))
        #expect(!s.predictRoutingDistances.contains(9))
    }

    @Test func predRouteAloneEnablesPredictionAtDistanceOne() {
        let s = RuntimeDiagnosticSwitches(environment: ["TURBO_FIELDFARE_PRED_ROUTE": "1"])
        #expect(s.predictRouting)
        #expect(!s.prefetchExperts)
        #expect(s.predictRoutingDistances == [1])
    }
}
