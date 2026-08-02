import Foundation
import Metal

/// Whole-conversation KV/state snapshot: exact-prefix resume for a single
/// request. Deliberately NOT a block cache — snapshotting the entire state
/// at a known position needs no block slicing, which is what makes the
/// non-sliceable parts (GDN recurrent state, conv tails) trivial: they are
/// just bytes at that position. Partial-prefix reuse is a later, separate
/// problem.
///
/// File layout (little-endian):
///   magic "TFKV" | version u32 | position u64 | promptCount u64 |
///   promptHash u64 | seedKind u32 (0 = logits section present,
///   1 = greedy token) | seedToken u32 | sectionCount u64 |
///   sectionLengths [u64] | raw section bytes...
/// Sections are emitted in deterministic model order (per layer: K then V
/// for attention layers, state then conv tail for linear layers), then the
/// seed logits section when seedKind == 0. Restore recomputes the expected
/// lengths from the live configuration and refuses on any mismatch.
public enum KVSnapshotError: Error {
    case ioFailed(String)
    case incompatible(String)
}

public struct KVSnapshotFile {
    static let magic: UInt32 = 0x54464B56 // "TFKV"
    static let version: UInt32 = 1

    public static func promptHash(_ ids: [Int32]) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for t in ids {
            h ^= UInt64(UInt32(bitPattern: t))
            h = h &* 0x100000001b3
        }
        return h
    }

    public static func write(url: URL,
                             position: Int,
                             promptIDs: [Int32],
                             seedKind: UInt32,
                             seedToken: UInt32,
                             sections: [(ptr: UnsafeRawPointer, bytes: Int)]) throws {
        var data = Data()
        func put<T>(_ v: T) { withUnsafeBytes(of: v) { data.append(contentsOf: $0) } }
        put(magic); put(version)
        put(UInt64(position)); put(UInt64(promptIDs.count)); put(promptHash(promptIDs))
        put(seedKind); put(seedToken)
        put(UInt64(sections.count))
        for s in sections { put(UInt64(s.bytes)) }
        for s in sections {
            data.append(Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: s.ptr),
                             count: s.bytes, deallocator: .none))
        }
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw KVSnapshotError.ioFailed("write \(url.path): \(error)")
        }
    }

    /// Validates the header against the caller's expectations and copies
    /// section bytes into the provided destinations. Returns (position,
    /// seedKind, seedToken).
    public static func read(url: URL,
                            promptIDs: [Int32],
                            sections: [(ptr: UnsafeMutableRawPointer, bytes: Int)])
        throws -> (position: Int, seedKind: UInt32, seedToken: UInt32) {
        guard let data = try? Data(contentsOf: url) else {
            throw KVSnapshotError.ioFailed("read \(url.path)")
        }
        var off = 0
        func take<T>(_ type: T.Type) throws -> T {
            let n = MemoryLayout<T>.size
            guard off + n <= data.count else {
                throw KVSnapshotError.incompatible("truncated header")
            }
            defer { off += n }
            return data.subdata(in: off..<off + n).withUnsafeBytes { $0.loadUnaligned(as: T.self) }
        }
        guard try take(UInt32.self) == magic, try take(UInt32.self) == version else {
            throw KVSnapshotError.incompatible("bad magic/version")
        }
        let position = Int(try take(UInt64.self))
        let promptCount = Int(try take(UInt64.self))
        let hash = try take(UInt64.self)
        guard promptCount == promptIDs.count, hash == promptHash(promptIDs) else {
            throw KVSnapshotError.incompatible("prompt mismatch")
        }
        let seedKind = try take(UInt32.self)
        let seedToken = try take(UInt32.self)
        let sectionCount = Int(try take(UInt64.self))
        guard sectionCount == sections.count else {
            throw KVSnapshotError.incompatible(
                "section count \(sectionCount) != expected \(sections.count)")
        }
        var lengths: [Int] = []
        for _ in 0..<sectionCount { lengths.append(Int(try take(UInt64.self))) }
        for (i, s) in sections.enumerated() {
            guard lengths[i] == s.bytes else {
                throw KVSnapshotError.incompatible(
                    "section \(i) length \(lengths[i]) != expected \(s.bytes)")
            }
        }
        for s in sections {
            guard off + s.bytes <= data.count else {
                throw KVSnapshotError.incompatible("truncated payload")
            }
            data.subdata(in: off..<off + s.bytes).withUnsafeBytes {
                s.ptr.copyMemory(from: $0.baseAddress!, byteCount: s.bytes)
            }
            off += s.bytes
        }
        return (position, seedKind, seedToken)
    }
}
