import Foundation
import Testing
@testable import TurboFieldfare

/// Byte-level BPE (Qwen, GPT-2 lineage) writes each token as a string over a
/// 256-symbol alphabet, one symbol per byte. Streaming decode is then a byte
/// buffer plus a UTF-8 boundary, with no re-decode of the history.
@Suite("Byte-level stream decoder")
struct ByteLevelStreamDecoderTests {
    @Test func asciiTokensPassThrough() {
        var d = ByteLevelStreamDecoder()
        #expect(d.push(tokenString: "Hello") == "Hello")
        #expect(d.push(tokenString: "Ġworld") == " world")
        #expect(d.flush() == "")
    }

    @Test func alphabetMapsSpaceNewlineAndNul() {
        var d = ByteLevelStreamDecoder()
        #expect(d.push(tokenString: "Ġ") == " ")
        #expect(d.push(tokenString: "Ċ") == "\n")
        #expect(d.push(tokenString: "Ā") == "\u{0}")
    }

    @Test func holdsAnIncompleteCodepointUntilItsLastByteArrives() {
        // U+1F99D raccoon is F0 9F A6 9D. Two bytes arrive, then two more.
        var d = ByteLevelStreamDecoder()
        let first = ByteLevelStreamDecoder.tokenString(bytes: [0xF0, 0x9F])
        let second = ByteLevelStreamDecoder.tokenString(bytes: [0xA6, 0x9D])
        #expect(d.push(tokenString: first) == "")
        #expect(d.push(tokenString: second) == "🦝")
        #expect(d.flush() == "")
    }

    @Test func emitsTheCompletePrefixWhenATokenEndsMidCodepoint() {
        var d = ByteLevelStreamDecoder()
        let token = ByteLevelStreamDecoder.tokenString(bytes: Array("ab".utf8) + [0xE6, 0xBC])
        #expect(d.push(tokenString: token) == "ab")
        #expect(d.push(tokenString: ByteLevelStreamDecoder.tokenString(bytes: [0xA2])) == "漢")
    }

    @Test func flushReplacesADanglingPartialCodepoint() {
        var d = ByteLevelStreamDecoder()
        _ = d.push(tokenString: ByteLevelStreamDecoder.tokenString(bytes: [0xF0, 0x9F]))
        let tail = d.flush()
        #expect(tail.unicodeScalars.allSatisfy { $0 == "\u{FFFD}" })
        #expect(!tail.isEmpty)
    }

    @Test func aTokenOutsideTheAlphabetIsNotByteLevel() {
        var d = ByteLevelStreamDecoder()
        #expect(d.push(tokenString: "▁world") == nil)
    }

    @Test func tokenStringRoundTripsEveryByte() {
        let all = (0...255).map { UInt8($0) }
        let s = ByteLevelStreamDecoder.tokenString(bytes: all)
        #expect(ByteLevelStreamDecoder.bytes(tokenString: s) == all)
    }
}

/// The ChatML fixture is a real byte-level BPE with a bytes-only vocabulary,
/// so every non-ASCII character arrives one byte per token. That is the worst
/// case for a streaming decoder and the case the old one got wrong: a
/// partially decoded codepoint changed the emitted prefix, and the delta was
/// dropped on resync.
@Suite("ChatML streaming detokenizer")
struct ChatMLStreamingDetokenizerTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    }

    private func streamed(_ text: String) -> (deltas: [String], tail: String, ids: [Int32]) {
        let ids = tok.encode(text, addBOS: false)
        var detok = GFDetokenizer(tokenizer: tok)
        var deltas: [String] = []
        for id in ids { deltas.append(detok.push(id)) }
        return (deltas, detok.flush(), ids)
    }

    @Test(arguments: [
        "mixed 漢 and 🦝 text",
        "🦝🦝🦝",
        "Здравствуй мир",
        "ends with emoji 🦝",
        "plain ascii, line\nbreak",
    ])
    func streamingEqualsBatchDecodeAndNeverEmitsReplacementCharacters(_ text: String) {
        let r = streamed(text)
        let joined = r.deltas.joined() + r.tail
        #expect(joined == tok.decode(r.ids))
        #expect(joined == text)
        for delta in r.deltas {
            #expect(!delta.unicodeScalars.contains("\u{FFFD}"), "delta \(delta.debugDescription)")
        }
    }

    @Test func aSpecialTokenMidStreamEmitsNothingAndDoesNotCorruptTheRest() {
        var detok = GFDetokenizer(tokenizer: tok)
        var out = ""
        for id in tok.encode("ab", addBOS: false) { out += detok.push(id) }
        #expect(detok.push(tok.endOfTurnID) == "")
        for id in tok.encode(" 漢", addBOS: false) { out += detok.push(id) }
        out += detok.flush()
        #expect(out == "ab 漢")
    }
}
