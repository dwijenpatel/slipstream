import Testing
@testable import TurboFieldfare

/// The predicted-routing diagnostic applies layer L+d's router to layer L's
/// state and scores the guess against layer L+d's real routing when that
/// layer arrives. The ledger holds the outstanding guesses per distance and
/// keeps recall per distance, so one run measures several lookaheads.
@Suite("Route prediction ledger")
struct RoutePredictionLedgerTests {
    @Test func scoresEachDistanceAgainstTheLayerItPredicted() {
        var ledger = RoutePredictionLedger(distances: [1, 2])
        ledger.record(fromLayer: 0, experts: [1, 2, 3, 4], distance: 1)   // predicts layer 1
        ledger.record(fromLayer: 0, experts: [5, 6, 7, 8], distance: 2)   // predicts layer 2
        ledger.score(layer: 1, actual: [1, 2, 9, 9])                       // 2 of 4 right at distance 1
        ledger.record(fromLayer: 1, experts: [5, 6, 7, 8], distance: 1)   // predicts layer 2
        ledger.score(layer: 2, actual: [5, 8, 9, 9])                       // 2 of 4 at both distances
        #expect(ledger.hits[1] == 4)
        #expect(ledger.total[1] == 8)
        #expect(ledger.hits[2] == 2)
        #expect(ledger.total[2] == 4)
        #expect(ledger.recall(distance: 1) == 0.5)
        #expect(ledger.recall(distance: 2) == 0.5)
    }

    @Test func aLayerWithNoGuessIsNotCounted() {
        var ledger = RoutePredictionLedger(distances: [1])
        ledger.score(layer: 0, actual: [1, 2])
        #expect(ledger.total[1] == 0)
        #expect(ledger.recall(distance: 1) == nil)
    }

    @Test func resetDropsOutstandingGuessesButKeepsTheTallies() {
        var ledger = RoutePredictionLedger(distances: [1])
        ledger.record(fromLayer: 0, experts: [1], distance: 1)
        ledger.score(layer: 1, actual: [1])
        ledger.record(fromLayer: 5, experts: [1], distance: 1)
        ledger.reset()
        ledger.score(layer: 6, actual: [1])
        #expect(ledger.hits[1] == 1)
        #expect(ledger.total[1] == 1)
    }

    @Test func aGuessIsConsumedOnce() {
        var ledger = RoutePredictionLedger(distances: [1])
        ledger.record(fromLayer: 0, experts: [1, 2], distance: 1)
        ledger.score(layer: 1, actual: [1, 2])
        ledger.score(layer: 1, actual: [1, 2])
        #expect(ledger.total[1] == 2)
    }
}
