import Foundation
import Metal

/// The process footprint macOS kills on. Resident set size under-reports
/// GPU allocations on unified memory; `phys_footprint` is what jetsam
/// compares against the limit, so it is the number every guard here reads.
public enum ProcessMemory {
    public static func physicalFootprintBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return UInt64(info.phys_footprint)
    }
}

/// How much of the machine a request may take. The cap is a fraction of the
/// device's working-set limit, because the limit is where allocations start
/// failing and the process starts paging, not where it is still fast.
public struct MemoryBudget: Sendable, Equatable {
    public static let admissionFraction = 0.90

    public let deviceLimitBytes: UInt64
    public let capBytes: UInt64

    public init(deviceLimitBytes: UInt64) {
        self.deviceLimitBytes = deviceLimitBytes
        self.capBytes = UInt64(Double(deviceLimitBytes) * Self.admissionFraction)
    }

    /// The GPU's recommended working set, lowered to the wired limit when an
    /// operator has set one (`sysctl iogpu.wired_limit_mb`).
    public static func forDevice(_ device: MTLDevice) -> MemoryBudget {
        var limit = device.recommendedMaxWorkingSetSize
        var wiredMB: Int64 = 0
        var size = MemoryLayout<Int64>.size
        if sysctlbyname("iogpu.wired_limit_mb", &wiredMB, &size, nil, 0) == 0, wiredMB > 0 {
            limit = min(limit, UInt64(wiredMB) << 20)
        }
        return MemoryBudget(deviceLimitBytes: limit)
    }

    public func admits(currentBytes: UInt64, predictedDeltaBytes: UInt64) -> Bool {
        currentBytes.addingReportingOverflow(predictedDeltaBytes).partialValue <= capBytes
            && !currentBytes.addingReportingOverflow(predictedDeltaBytes).overflow
    }

    /// The largest allowed slot count whose static ceiling fits under the
    /// cap, or nil when the fixed part alone does not fit. `bytesPerSlot` is
    /// one slot across every layer.
    public func largestSlotCount(fixedBytes: UInt64,
                                 bytesPerSlot: UInt64,
                                 allowed: [Int]) -> Int? {
        guard fixedBytes <= capBytes else { return nil }
        let room = capBytes - fixedBytes
        return allowed.filter { UInt64($0) * bytesPerSlot <= room }.max()
    }
}

/// Bytes of footprint growth per prefilled token, learned from the chunks
/// already run. An exponential average with an outlier gate: a single tail
/// chunk once measured 13.6 times the running rate in oMLX's serving and
/// poisoned admission for the process lifetime.
public struct PrefillTransientEstimator: Sendable, Equatable {
    public static let alpha = 0.3
    public static let outlierRatio = 8.0
    public static let margin = 1.3

    public private(set) var bytesPerToken: Double = 0
    public private(set) var samples = 0
    /// The first chunk's growth, kept for diagnostics and never learned. It
    /// is one-time: the expert slots become resident as the chunk streams
    /// every expert through them, and the prefill scratch is allocated. At
    /// 96 slots on the 12k prompt it was 7.6 GB, and extrapolating it with
    /// the margin refused the second chunk of a process whose real peak was
    /// 8.4 GB (sweep of 2026-09-06). A 4,096-token chunk cannot grow by
    /// gigabytes on its own, so admitting the second chunk on the analytic
    /// floor lets nothing through that the guard exists to stop.
    public private(set) var firstChunkBytesPerToken: Double?

    public init() {}

    public mutating func record(deltaBytes: Int64, tokens: Int) {
        guard tokens > 0 else { return }
        let sample = max(0, Double(deltaBytes)) / Double(tokens)
        if firstChunkBytesPerToken == nil {
            firstChunkBytesPerToken = sample
            return
        }
        if samples > 0, bytesPerToken > 0, sample > bytesPerToken * Self.outlierRatio {
            return
        }
        bytesPerToken = samples == 0
            ? sample
            : bytesPerToken * (1 - Self.alpha) + sample * Self.alpha
        samples += 1
    }

    /// The larger of the analytic floor and the learned rate, times the
    /// chunk, with the margin.
    public func predictedDeltaBytes(tokens: Int, floorBytesPerToken: UInt64) -> UInt64 {
        let rate = max(Double(floorBytesPerToken), bytesPerToken)
        return UInt64((rate * Double(tokens) * Self.margin).rounded(.up))
    }
}

public enum MemoryBudgetError: Error, CustomStringConvertible, Equatable {
    case wouldExceed(currentBytes: UInt64, predictedDeltaBytes: UInt64,
                     capBytes: UInt64, chunkTokens: Int)

    public var description: String {
        switch self {
        case .wouldExceed(let current, let predicted, let cap, let tokens):
            func gb(_ bytes: UInt64) -> String {
                String(format: "%.1f GB", Double(bytes) / Double(1 << 30))
            }
            return "memory budget: \(gb(current)) in use plus a predicted \(gb(predicted)) "
                + "for the next \(tokens)-token prefill chunk exceeds the \(gb(cap)) "
                + "admission cap; reduce --expert-cache-slots or --max-context, "
                + "or send a shorter prompt"
        }
    }
}

/// Decides, before each prefill chunk, whether the process can afford it,
/// and learns the per-token cost from the chunks it admits. A refusal is an
/// error on the request; the alternative is a jetsam kill of the server.
public final class PrefillMemoryGuard: @unchecked Sendable {
    public let budget: MemoryBudget
    public let floorBytesPerToken: UInt64
    public private(set) var estimator = PrefillTransientEstimator()
    private let footprint: () -> UInt64?
    private var footprintBeforeChunk: UInt64?
    private let lock = NSLock()

    public init(budget: MemoryBudget,
                floorBytesPerToken: UInt64,
                footprint: @escaping () -> UInt64? = ProcessMemory.physicalFootprintBytes) {
        self.budget = budget
        self.floorBytesPerToken = floorBytesPerToken
        self.footprint = footprint
    }

    /// Throws when the footprint now plus the predicted growth for `tokens`
    /// would cross the cap. An unreadable footprint admits: the guard must
    /// never turn a diagnostics failure into a refused request.
    public func beforeChunk(tokens: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let current = footprint() else {
            footprintBeforeChunk = nil
            return
        }
        footprintBeforeChunk = current
        let predicted = estimator.predictedDeltaBytes(tokens: tokens,
                                                      floorBytesPerToken: floorBytesPerToken)
        guard budget.admits(currentBytes: current, predictedDeltaBytes: predicted) else {
            throw MemoryBudgetError.wouldExceed(currentBytes: current,
                                                predictedDeltaBytes: predicted,
                                                capBytes: budget.capBytes,
                                                chunkTokens: tokens)
        }
    }

    public func afterChunk(tokens: Int) {
        lock.lock()
        defer { lock.unlock() }
        guard let before = footprintBeforeChunk, let after = footprint() else { return }
        footprintBeforeChunk = nil
        estimator.record(deltaBytes: Int64(after) - Int64(before), tokens: tokens)
    }
}
