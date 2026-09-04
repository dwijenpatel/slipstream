import Foundation
import Testing
@testable import TurboFieldfare

/// The snapshot file is the one artifact that outlives the process, so its
/// contract is pinned here: exact round trip, refusal on a different prompt,
/// atomic replacement, and no stray files in the directory afterwards.
@Suite("KV snapshot file")
struct KVSnapshotFileTests {
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kvsnapshot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func write(_ sections: [[UInt8]], promptIDs: [Int32], to url: URL,
                       seedKind: UInt32 = 1, seedToken: UInt32 = 42) throws {
        var buffers = sections
        try buffers.withUnsafeMutableBufferPointer { outer in
            var raw: [(ptr: UnsafeRawPointer, bytes: Int)] = []
            for i in outer.indices {
                let bytes = outer[i].count
                raw.append((UnsafeRawPointer(outer[i].withUnsafeBufferPointer { $0.baseAddress! }),
                            bytes))
            }
            try KVSnapshotFile.write(url: url, position: 7, promptIDs: promptIDs,
                                     seedKind: seedKind, seedToken: seedToken, sections: raw)
        }
    }

    private func read(sectionSizes: [Int], promptIDs: [Int32], from url: URL)
        throws -> (position: Int, seedKind: UInt32, seedToken: UInt32, sections: [[UInt8]]) {
        var out = sectionSizes.map { [UInt8](repeating: 0, count: $0) }
        let r = try out.withUnsafeMutableBufferPointer { outer -> (Int, UInt32, UInt32) in
            var dest: [(ptr: UnsafeMutableRawPointer, bytes: Int)] = []
            for i in outer.indices {
                dest.append((UnsafeMutableRawPointer(outer[i].withUnsafeMutableBufferPointer { $0.baseAddress! }),
                             outer[i].count))
            }
            let r = try KVSnapshotFile.read(url: url, promptIDs: promptIDs, sections: dest)
            return (r.position, r.seedKind, r.seedToken)
        }
        return (r.0, r.1, r.2, out)
    }

    @Test func roundTripsHeaderAndSectionsByteForByte() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("a.kv")
        let sections: [[UInt8]] = [[1, 2, 3, 4], [9, 8, 7], []]
        try write(sections, promptIDs: [5, 6, 7], to: url)

        let r = try read(sectionSizes: [4, 3, 0], promptIDs: [5, 6, 7], from: url)
        #expect(r.position == 7)
        #expect(r.seedKind == 1)
        #expect(r.seedToken == 42)
        #expect(r.sections == sections)
    }

    @Test func refusesADifferentPrompt() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("a.kv")
        try write([[1, 2]], promptIDs: [5, 6, 7], to: url)

        #expect(throws: KVSnapshotError.self) {
            _ = try read(sectionSizes: [2], promptIDs: [5, 6, 8], from: url)
        }
    }

    @Test func overwriteReplacesTheWholeFile() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("a.kv")
        try write([[1, 2, 3, 4, 5, 6, 7, 8]], promptIDs: [1], to: url)
        try write([[9]], promptIDs: [2], to: url)

        let r = try read(sectionSizes: [1], promptIDs: [2], from: url)
        #expect(r.sections == [[9]])
    }

    @Test func leavesNoOtherFilesInTheDirectory() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("a.kv")
        try write([[1, 2, 3]], promptIDs: [1], to: url)
        try write([[4, 5, 6]], promptIDs: [1], to: url)

        let names = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        #expect(names == ["a.kv"])
    }

    @Test func aMissingParentDirectoryIsAnIOError() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("missing/a.kv")
        #expect(throws: KVSnapshotError.self) {
            try write([[1]], promptIDs: [1], to: url)
        }
    }
}
