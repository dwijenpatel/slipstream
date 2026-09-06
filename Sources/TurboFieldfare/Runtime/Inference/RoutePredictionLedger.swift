/// Outstanding routing guesses and their recall, per lookahead distance.
///
/// The predicted-routing diagnostic applies layer L+d's router to layer L's
/// post-attention state, the residual-stream approximation from the MoE
/// prefetch literature. A guess made at layer L for layer L+d is scored when
/// layer L+d's real routing is read back. Several distances run in one pass,
/// so a single run gives the recall curve that decides how far ahead a
/// prefetch can be issued.
public struct RoutePredictionLedger: Sendable {
    public let distances: [Int]
    public private(set) var hits: [Int: UInt64]
    public private(set) var total: [Int: UInt64]
    /// target layer -> (distance -> guessed experts)
    private var outstanding: [Int: [Int: [Int]]] = [:]

    public init(distances: [Int]) {
        self.distances = distances
        self.hits = Dictionary(uniqueKeysWithValues: distances.map { ($0, 0) })
        self.total = Dictionary(uniqueKeysWithValues: distances.map { ($0, 0) })
    }

    /// A guess made at `fromLayer` about layer `fromLayer + distance`.
    public mutating func record(fromLayer: Int, experts: [Int], distance: Int) {
        outstanding[fromLayer + distance, default: [:]][distance] = experts
    }

    /// Scores every outstanding guess about `layer` and consumes them.
    public mutating func score(layer: Int, actual: [Int]) {
        guard let guesses = outstanding.removeValue(forKey: layer) else { return }
        let actualSet = Set(actual)
        for (distance, guessed) in guesses {
            let guessedSet = Set(guessed)
            total[distance, default: 0] &+= UInt64(actual.count)
            hits[distance, default: 0] &+= UInt64(actualSet.filter { guessedSet.contains($0) }.count)
        }
    }

    /// Drops outstanding guesses, for a token boundary; the tallies stay.
    public mutating func reset() {
        outstanding.removeAll()
    }

    public func recall(distance: Int) -> Double? {
        guard let t = total[distance], t > 0, let h = hits[distance] else { return nil }
        return Double(h) / Double(t)
    }
}
