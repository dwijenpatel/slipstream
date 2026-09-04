import Foundation
import Tokenizers

/// Streaming detokenizer for generation loops.
///
/// Two challenges drive the design:
///
/// 1. BPE byte-fallback splits multi-byte codepoints (e.g. emoji) across several
///    tokens. Naively decoding each token in isolation yields broken UTF-8.
/// 2. swift-transformers' decoder silently drops byte-fallback tokens that sit
///    at the **end** of the decoded sequence (the bytes are committed only once
///    a non-byte-fallback token follows). For us this matters at `flush()`.
///
/// Strategy:
///   - During `push(_:)` we decode the longest prefix of accumulated IDs that
///     does NOT end with byte-fallback tokens, then emit the delta vs. previously
///     emitted text. Any trailing byte-fallback IDs are held back.
///   - During `flush()` we decode the stable prefix as above AND manually
///     assemble the trailing byte-fallback bytes into a UTF-8 string. This
///     recovers text the library would otherwise drop on a sequence-ending
///     codepoint.
struct GFDetokenizer {
    @usableFromInline let tokenizer: any Tokenizer
    @usableFromInline var stableIDs: [Int] = []
    @usableFromInline var trailingByteIDs: [Int] = []
    @usableFromInline var emitted: String = ""
    /// Set when the vocabulary is byte-level (Qwen). That path is a byte
    /// buffer plus a UTF-8 boundary, so it costs one token per push instead
    /// of one full re-decode, and it never emits a partial codepoint.
    private var byteLevel: ByteLevelStreamDecoder?
    /// Whether an id is a special (added) token, decided once per id by the
    /// library's own skip-special filter, which is the reference behavior.
    private var specialByID: [Int: Bool] = [:]

    init(tokenizer: GFTokenizer) {
        self.tokenizer = tokenizer.tokenizer
        self.byteLevel = Self.usesByteLevelTokens(tokenizer.tokenizer)
            ? ByteLevelStreamDecoder()
            : nil
    }

    /// Byte-level BPE writes a leading space as U+0120 in its token strings;
    /// a metaspace tokenizer writes U+2581. One short tokenize tells them
    /// apart without reading the tokenizer's configuration.
    static func usesByteLevelTokens(_ tokenizer: any Tokenizer) -> Bool {
        tokenizer.tokenize(text: "hello world").contains { $0.contains("Ġ") }
    }

    mutating func push(_ id: Int32) -> String {
        let tokenID = Int(id)
        if byteLevel != nil {
            return pushByteLevel(tokenID)
        }
        let token = tokenizer.convertIdToToken(tokenID) ?? ""
        if Self.isByteFallback(token) {
            trailingByteIDs.append(tokenID)
            return ""
        }

        if !trailingByteIDs.isEmpty {
            stableIDs.append(contentsOf: trailingByteIDs)
            trailingByteIDs.removeAll(keepingCapacity: true)
        }
        stableIDs.append(tokenID)

        let current = tokenizer.decode(tokens: stableIDs, skipSpecialTokens: true)
        return commitDelta(current)
    }

    private mutating func pushByteLevel(_ tokenID: Int) -> String {
        guard let token = tokenizer.convertIdToToken(tokenID) else { return "" }
        if isSpecial(tokenID, token) { return "" }
        if let text = byteLevel!.push(tokenString: token) { return text }
        // Not a byte-level string after all: let the library decode this one
        // token on its own rather than drop it.
        return tokenizer.decode(tokens: [tokenID], skipSpecialTokens: true)
    }

    private mutating func isSpecial(_ tokenID: Int, _ token: String) -> Bool {
        if let known = specialByID[tokenID] { return known }
        let special = !token.isEmpty
            && tokenizer.decode(tokens: [tokenID], skipSpecialTokens: true).isEmpty
        specialByID[tokenID] = special
        return special
    }

    mutating func flush() -> String {
        if byteLevel != nil {
            return byteLevel!.flush()
        }
        let stableText = stableIDs.isEmpty
            ? ""
            : tokenizer.decode(tokens: stableIDs, skipSpecialTokens: true)

        let trailingText = assembleByteFallback(trailingByteIDs)
        let fullText = stableText + trailingText
        return commitDelta(fullText)
    }

    @usableFromInline
    mutating func commitDelta(_ current: String) -> String {
        guard current.hasPrefix(emitted) else {
            // Decoder altered the prefix — extremely rare in append-only streams.
            // Resync rather than emit garbage; the user-visible loss is bounded
            // to whatever was retokenized.
            emitted = current
            return ""
        }
        let delta = String(current.dropFirst(emitted.count))
        emitted = current
        return delta
    }

    @usableFromInline
    func assembleByteFallback(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count)
        for id in ids {
            guard let tok = tokenizer.convertIdToToken(id) else { continue }
            guard Self.isByteFallback(tok),
                  let byte = UInt8(tok.dropFirst(3).dropLast(), radix: 16)
            else { continue }
            bytes.append(byte)
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    @usableFromInline
    static func isByteFallback(_ token: String) -> Bool {
        token.count == 6
            && token.hasPrefix("<0x")
            && token.hasSuffix(">")
            && token.dropFirst(3).dropLast().allSatisfy { $0.isHexDigit }
    }
}
