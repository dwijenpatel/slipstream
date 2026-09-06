import Foundation

/// The prediction and prefetch diagnostics, read from the environment the
/// same way by every surface. They used to be read by the CLI alone, so a
/// server started with `TURBO_FIELDFARE_PREFETCH=1` ran without prefetch and
/// an overnight session arm measured nothing (smoke run, 2026-09-06).
///
/// - `TURBO_FIELDFARE_PRED_ROUTE=1`: score layer L+1's routing guessed from
///   layer L's state; diagnostic only.
/// - `TURBO_FIELDFARE_PRED_ROUTE_DISTANCES=1,2,3`: the same at several
///   lookahead distances, up to `RealForwardRunner.maxPredictionDistances`.
/// - `TURBO_FIELDFARE_PREFETCH=1`: use the guess to warm the slots ahead.
/// - `TURBO_FIELDFARE_PREFETCH_DISTANCE=<n>`: how many layers ahead, default 1.
public struct RuntimeDiagnosticSwitches: Equatable, Sendable {
    public let predictRouting: Bool
    public let predictRoutingDistances: [Int]
    public let prefetchExperts: Bool
    public let prefetchDistance: Int

    public init(environment: [String: String]) {
        let limit = RealForwardRunner.maxPredictionDistances
        var distances = Set<Int>()
        if let raw = environment["TURBO_FIELDFARE_PRED_ROUTE_DISTANCES"] {
            for piece in raw.split(separator: ",") {
                if let d = Int(piece.trimmingCharacters(in: .whitespaces)), (1...limit).contains(d) {
                    distances.insert(d)
                }
            }
        }
        let prefetch = environment["TURBO_FIELDFARE_PREFETCH"] == "1"
        let requested = Int(environment["TURBO_FIELDFARE_PREFETCH_DISTANCE"] ?? "") ?? 1
        let prefetchDistance = max(1, min(limit, requested))
        if prefetch { distances.insert(prefetchDistance) }
        let predict = prefetch || !distances.isEmpty || environment["TURBO_FIELDFARE_PRED_ROUTE"] == "1"
        if distances.isEmpty { distances.insert(1) }
        self.predictRouting = predict
        self.predictRoutingDistances = distances.sorted()
        self.prefetchExperts = prefetch
        self.prefetchDistance = prefetchDistance
    }

    public func apply(to runner: RealForwardRunner) {
        guard predictRouting else { return }
        runner.predictRouting = true
        runner.predictRoutingDistances = predictRoutingDistances
        runner.prefetchExperts = prefetchExperts
        runner.prefetchDistance = prefetchDistance
    }
}
