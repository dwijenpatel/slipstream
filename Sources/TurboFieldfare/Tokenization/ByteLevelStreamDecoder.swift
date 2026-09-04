import Foundation

/// Streaming decoder for byte-level BPE token strings (Qwen and the GPT-2
/// lineage). Each token string is a sequence of symbols from a fixed
/// 256-symbol alphabet, one symbol per byte: printable Latin-1 bytes map to
/// themselves and the rest to U+0100 onward, in byte order. Decoding is
/// therefore a byte buffer plus a UTF-8 boundary: every complete codepoint is
/// emitted as soon as its last byte arrives, and an incomplete tail is held.
/// Cost is proportional to the token, never to the history.
struct ByteLevelStreamDecoder {
    private var pending: [UInt8] = []

    /// GPT-2 `bytes_to_unicode`, inverted. The forward table maps each of
    /// the 188 printable Latin-1 bytes to itself and assigns the remaining
    /// 68 bytes to U+0100... in increasing byte order.
    static let byteForScalar: [UInt32: UInt8] = {
        var table: [UInt32: UInt8] = [:]
        var next: UInt32 = 256
        for byte in 0...255 {
            let printable = (0x21...0x7E).contains(byte)
                || (0xA1...0xAC).contains(byte)
                || (0xAE...0xFF).contains(byte)
            if printable {
                table[UInt32(byte)] = UInt8(byte)
            } else {
                table[next] = UInt8(byte)
                next += 1
            }
        }
        return table
    }()

    static let scalarForByte: [UInt32] = {
        var forward = [UInt32](repeating: 0, count: 256)
        for (scalar, byte) in byteForScalar { forward[Int(byte)] = scalar }
        return forward
    }()

    /// The bytes a token string denotes, or nil if any symbol is outside the
    /// alphabet, which means the token is not byte-level (a metaspace token,
    /// for one).
    static func bytes(tokenString: String) -> [UInt8]? {
        var out: [UInt8] = []
        out.reserveCapacity(tokenString.unicodeScalars.count)
        for scalar in tokenString.unicodeScalars {
            guard let byte = byteForScalar[scalar.value] else { return nil }
            out.append(byte)
        }
        return out
    }

    /// The token string for a byte sequence. Test support and the inverse of
    /// `bytes(tokenString:)`.
    static func tokenString(bytes: [UInt8]) -> String {
        var scalars = String.UnicodeScalarView()
        for byte in bytes {
            scalars.append(Unicode.Scalar(scalarForByte[Int(byte)])!)
        }
        return String(scalars)
    }

    /// Appends the token's bytes and returns the text of every codepoint that
    /// is now complete. Returns nil, and holds nothing, if the token is not a
    /// byte-level token.
    mutating func push(tokenString: String) -> String? {
        guard let bytes = Self.bytes(tokenString: tokenString) else { return nil }
        pending.append(contentsOf: bytes)
        let complete = Self.completePrefixLength(pending)
        guard complete > 0 else { return "" }
        let text = String(decoding: pending[..<complete], as: UTF8.self)
        pending.removeFirst(complete)
        return text
    }

    /// Emits whatever is held. An incomplete tail cannot become valid, so it
    /// decodes lossily, one replacement character per byte.
    mutating func flush() -> String {
        guard !pending.isEmpty else { return "" }
        let text = String(decoding: pending, as: UTF8.self)
        pending.removeAll()
        return text
    }

    /// Length of the prefix that ends on a codepoint boundary. Only a trailing
    /// lead byte with too few continuation bytes behind it is held back; any
    /// other malformed run is left to the lossy decode, which the reference
    /// decoder also does.
    static func completePrefixLength(_ bytes: [UInt8]) -> Int {
        let n = bytes.count
        guard n > 0 else { return 0 }
        // Walk back at most three bytes to find the last lead byte.
        var i = n - 1
        var back = 0
        while i >= 0, back < 3, bytes[i] & 0xC0 == 0x80 {
            i -= 1
            back += 1
        }
        guard i >= 0 else { return n }
        let lead = bytes[i]
        let need: Int
        if lead & 0x80 == 0 {
            need = 1
        } else if lead & 0xE0 == 0xC0 {
            need = 2
        } else if lead & 0xF0 == 0xE0 {
            need = 3
        } else if lead & 0xF8 == 0xF0 {
            need = 4
        } else {
            return n
        }
        let have = n - i
        return have < need ? i : n
    }
}
