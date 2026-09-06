import Foundation

/// Records which experts the router chose, per token and layer.
///
/// A cache policy's behavior depends only on the sequence of expert IDs each
/// layer requests, so one trace replays every policy at every slot count
/// offline: `playbook/route_replay.py`. Prefill routing is recorded too,
/// because prefill streams experts through the same slot cache and sets its
/// state at the first decode token. Enabled by `TURBO_FIELDFARE_ROUTE_TRACE=path`.
///
/// Format, little-endian: header `RTRC`, version 1, numLayers (u8), topK (u8),
/// numExperts (u16); then records of phase (u8), position (u32), layer (u8),
/// and topK expert IDs (u8 each). Expert IDs above 255 are not representable,
/// so models with more than 256 experts per layer are refused.
public final class RouteTrace: @unchecked Sendable {
    public enum Phase: UInt8, Sendable {
        case prefill = 0
        case decode = 1
    }

    public struct Header: Equatable, Sendable {
        public let numLayers: Int
        public let topK: Int
        public let numExperts: Int
    }

    public struct Record: Equatable, Sendable {
        public let phase: Phase
        public let position: Int
        public let layer: Int
        public let experts: [Int]

        public init(phase: Phase, position: Int, layer: Int, experts: [Int]) {
            self.phase = phase
            self.position = position
            self.layer = layer
            self.experts = experts
        }
    }

    public struct Contents: Sendable {
        public let header: Header
        public let records: [Record]
    }

    public enum TraceError: Error, Equatable {
        case unsupportedShape(String)
        case malformed(String)
    }

    static let magic: [UInt8] = Array("RTRC".utf8)
    static let version: UInt8 = 1
    static let headerBytes = 9
    static let flushThresholdBytes = 64 * 1024

    private let handle: FileHandle
    private let topK: Int
    private let lock = NSLock()
    private var buffer = Data()
    private var closed = false

    public init(url: URL, numLayers: Int, topK: Int, numExperts: Int) throws {
        guard numExperts >= 1, numExperts <= 256, topK >= 1, topK <= 255,
              numLayers >= 1, numLayers <= 255 else {
            throw TraceError.unsupportedShape(
                "route trace supports up to 256 experts, 255 layers, top-255; got \(numExperts)/\(numLayers)/\(topK)")
        }
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw TraceError.malformed("cannot create \(url.path)")
        }
        self.handle = try FileHandle(forWritingTo: url)
        self.topK = topK
        var header = Data(Self.magic)
        header.append(Self.version)
        header.append(UInt8(numLayers))
        header.append(UInt8(topK))
        header.append(UInt8(numExperts & 0xFF))
        header.append(UInt8(numExperts >> 8))
        handle.write(header)
    }

    deinit {
        close()
    }

    /// The trace file named by `TURBO_FIELDFARE_ROUTE_TRACE`, or nil when the
    /// variable is unset or empty.
    public static func fromEnvironment(_ environment: [String: String],
                                       numLayers: Int, topK: Int, numExperts: Int) throws -> RouteTrace? {
        guard let path = environment["TURBO_FIELDFARE_ROUTE_TRACE"], !path.isEmpty else {
            return nil
        }
        return try RouteTrace(url: URL(fileURLWithPath: path),
                              numLayers: numLayers, topK: topK, numExperts: numExperts)
    }

    /// Records one layer's routing for one token. A wrong expert count is
    /// dropped rather than corrupting the fixed-width record stream.
    public func record(phase: Phase, position: Int, layer: Int, experts: [Int]) {
        guard experts.count == topK, position >= 0, position <= Int(UInt32.max) else { return }
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        buffer.append(phase.rawValue)
        let p = UInt32(position)
        buffer.append(UInt8(p & 0xFF))
        buffer.append(UInt8((p >> 8) & 0xFF))
        buffer.append(UInt8((p >> 16) & 0xFF))
        buffer.append(UInt8(p >> 24))
        buffer.append(UInt8(clamping: layer))
        for expert in experts {
            buffer.append(UInt8(clamping: expert))
        }
        if buffer.count >= Self.flushThresholdBytes {
            flushLocked()
        }
    }

    /// Writes buffered records to the file. The runner calls this after each
    /// token so a killed process loses at most one token.
    public func flush() {
        lock.lock()
        flushLocked()
        lock.unlock()
    }

    public func close() {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        flushLocked()
        closed = true
        try? handle.close()
    }

    private func flushLocked() {
        guard !buffer.isEmpty else { return }
        handle.write(buffer)
        buffer.removeAll(keepingCapacity: true)
    }

    public static func read(url: URL) throws -> Contents {
        let data = try Data(contentsOf: url)
        guard data.count >= headerBytes,
              Array(data[0..<4]) == magic,
              data[4] == version else {
            throw TraceError.malformed("not a version-\(version) route trace: \(url.path)")
        }
        let header = Header(numLayers: Int(data[5]),
                            topK: Int(data[6]),
                            numExperts: Int(data[7]) | (Int(data[8]) << 8))
        let recordBytes = 6 + header.topK
        var records: [Record] = []
        records.reserveCapacity((data.count - headerBytes) / recordBytes)
        var offset = headerBytes
        while offset + recordBytes <= data.count {
            guard let phase = Phase(rawValue: data[offset]) else {
                throw TraceError.malformed("bad phase byte at offset \(offset)")
            }
            let position = Int(data[offset + 1])
                | (Int(data[offset + 2]) << 8)
                | (Int(data[offset + 3]) << 16)
                | (Int(data[offset + 4]) << 24)
            let layer = Int(data[offset + 5])
            let experts = (0..<header.topK).map { Int(data[offset + 6 + $0]) }
            records.append(Record(phase: phase, position: position, layer: layer, experts: experts))
            offset += recordBytes
        }
        return Contents(header: header, records: records)
    }
}
