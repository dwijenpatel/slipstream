import Foundation
import Metal

/// Streaming callbacks from `runRawCompletion`. `.prefill` reports monotonic
/// producer-defined prompt progress; scalar replay reports per token, while a
/// prefill-capable producer may report per internal chunk. `.token` fires per
/// decoded non-stop token; `.tail` carries the detokenizer flush remainder at a
/// stop boundary.
public enum RawDecodeProgress: Sendable {
    case prefill(done: Int, total: Int)
    case token(index: Int, id: Int32, delta: String)
    case tail(String)
}

public enum RawCompletionStart: Sendable, Equatable {
    case reset
    case resume(cachedPromptTokens: Int)
}

public struct RawDecodeResult: Sendable {
    public let prefillTokens: Int
    public let cachedPromptTokens: Int
    public let computedPrefillTokens: Int
    public let prefillSeconds: Double
    public let newTokens: Int
    public let decodeSeconds: Double
    public let reason: StopReason
    public let kvPosition: Int
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
}

/// Preallocated per-generation buffers (two 512 KiB vocab buffers plus a token
/// slot) and sampler. A warm session reuses them for every token, avoiding
/// per-token Metal buffer allocation.
///
/// `@unchecked Sendable`: the buffers and sampler are exclusively owned by one
/// generation at a time — the single-in-flight guard upstream is the contract.
public struct RawCompletionScratch: @unchecked Sendable {
    let logits: MTLBuffer
    let probs: MTLBuffer
    let outToken: MTLBuffer
    let sampler: Sampler

    public init(context: MetalContext, vocab: Int, logitSoftcap: Float = 30.0) throws {
        guard let logits = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                     options: .storageModeShared),
              let probs = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                    options: .storageModeShared),
              let outToken = context.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                       options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.logits = logits
        self.probs = probs
        self.outToken = outToken
        self.sampler = try Sampler(context: context, vocab: vocab,
                                   logitSoftcap: logitSoftcap)
    }
}

extension GenerationConfig {
    /// A pure-greedy config can use the fused head's GPU argmax
    /// (`RealForwardRunner.lastGreedyToken`) instead of sampling from the
    /// logits buffer. Anything else needs real logits.
    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1
    }

}

/// Producers that can whole-state snapshot/restore at a position
/// (exact-prefix resume). Optional capability; RealForwardRunner adopts it.
public protocol KVSnapshotting: AnyObject {
    func saveKVSnapshot(to url: URL, promptIDs: [Int32], position: Int,
                        seed: PrefillSeed, logits: MTLBuffer) throws
    func restoreKVSnapshot(from url: URL, promptIDs: [Int32],
                           logits: MTLBuffer) throws -> (position: Int, seed: PrefillSeed)?
}

/// Raw-completion prefill + decode loop shared by the CLI and the Mac app.
/// Consumes pre-encoded `promptIds` (BOS + verbatim encode upstream — no chat
/// template). Stop handling, detokenizer flush ordering, and history append
/// ordering are shared by both front ends.
///
/// When the producer runs the fused lm_head (`RealForwardRunner` default) the
/// logits buffer is never written; the loop then requires a pure-greedy config
/// and reads `lastGreedyToken`. Callers with sampling configs must construct
/// the runner with `forceLogitsHead: true`.
public func runRawCompletion(producer: any LogitProducer,
                             tokenizer: GFTokenizer,
                             promptIds: [Int32],
                             config: GenerationConfig,
                             context: MetalContext,
                             scratch: RawCompletionScratch,
                             prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                             kvSnapshotPath: String? = nil,
                             start: RawCompletionStart = .reset,
                             shouldStop: () -> Bool = { false },
                             onProgress: (RawDecodeProgress) -> Void) async throws -> RawDecodeResult {
    try config.validate()
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }
    let fusedRunner = producer as? RealForwardRunner
    let fusedGreedy = fusedRunner?.usesFusedGreedyHead == true
    guard !fusedGreedy || config.isPureGreedy else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }

    let cachedPromptTokens: Int
    switch start {
    case .reset:
        cachedPromptTokens = 0
    case .resume(let count):
        guard count > 0, count < promptIds.count else {
            throw GeneratorError.invalidContinuation(
                "cached prompt token count must be greater than zero and less than the effective prompt")
        }
        guard producer is any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        cachedPromptTokens = count
    }
    let computedPrefillTokens = promptIds.count - cachedPromptTokens

    var detok = GFDetokenizer(tokenizer: tokenizer)
    var history = Array(promptIds.prefix(cachedPromptTokens))
    history.reserveCapacity(promptIds.count + config.maxNewTokens)

    if let context = producer as? any ContextWindowReporting,
       promptIds.count + config.maxNewTokens > context.maxContext {
        throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                             maxNew: config.maxNewTokens,
                                             maxContext: context.maxContext)
    }
    switch start {
    case .reset:
        producer.reset()
    case .resume:
        let continuable = producer as! any ContinuableLogitProducer
        try continuable.prepareForContinuation(expectedPosition: cachedPromptTokens)
    }
    let prefillStart = Date()
    var position = cachedPromptTokens
    var prefillSeed: PrefillSeed?
    let prefillTokens = promptIds[cachedPromptTokens...]

    // Exact-prefix resume: a compatible snapshot replaces the whole prefill
    // with a state read. Restore failures other than absence are surfaced;
    // a wrong-prompt file must not silently fall through to prefill with a
    // half-written state.
    var restoredFromSnapshot = false
    var snapshotter: (any KVSnapshotting)?
    var snapshotURL: URL?
    if let kvSnapshotPath, start == .reset, cachedPromptTokens == 0,
       let snap = producer as? any KVSnapshotting {
        snapshotter = snap
        snapshotURL = URL(fileURLWithPath: kvSnapshotPath)
        if let r = try snap.restoreKVSnapshot(from: snapshotURL!,
                                              promptIDs: promptIds,
                                              logits: scratch.logits) {
            guard r.position == promptIds.count else {
                throw KVSnapshotError.incompatible(
                    "snapshot position \(r.position) != prompt length \(promptIds.count)")
            }
            if case .greedyToken = r.seed, !config.isPureGreedy {
                throw KVSnapshotError.incompatible(
                    "greedy-seed snapshot cannot serve a sampling configuration")
            }
            position = r.position
            prefillSeed = r.seed
            history.append(contentsOf: prefillTokens)
            restoredFromSnapshot = true
            onProgress(.prefill(done: promptIds.count, total: promptIds.count))
        }
    }

    switch restoredFromSnapshot ? PrefillRuntimeConfig.Mode.off : prefillConfig.mode {
    case .chunked where producer is any ChunkedPrefillRunner:
        let chunked = producer as! any ChunkedPrefillRunner
        let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
        let result = try await chunked.prefillChunked(tokens: prefillTokens,
                                                      startPosition: position,
                                                      outputMode: mode,
                                                      config: prefillConfig,
                                                      into: scratch.logits) { done in
            onProgress(.prefill(done: cachedPromptTokens + done, total: promptIds.count))
        }
        if mode == .logits, result.seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill requested logits but producer returned \(result.seed)")
        }
        if case .greedyToken = result.seed, !config.isPureGreedy {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill returned a greedy token for a sampling config")
        }
        position = result.newPosition
        prefillSeed = result.seed
        history.append(contentsOf: prefillTokens)
    case .chunked:
        throw PrefillError.chunkedUnsupported(
            PrefillError.chunkedRequiresChunkedRunnerReason)
    case .off:
        guard !restoredFromSnapshot else { break }
        for t in prefillTokens {
            try Task.checkCancellation()
            try await producer.produce(token: t, position: position, into: scratch.logits)
            position += 1
            history.append(t)
            onProgress(.prefill(done: position, total: promptIds.count))
        }
    }

    if !restoredFromSnapshot, let snap = snapshotter, let url = snapshotURL,
       let seed = prefillSeed {
        try snap.saveKVSnapshot(to: url,
                                promptIDs: promptIds,
                                position: position,
                                seed: seed,
                                logits: scratch.logits)
    }

    let decodeStart = Date()
    let prefillSeconds = decodeStart.timeIntervalSince(prefillStart)
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []

    // Speculative decoding (M1, greedy only): a round verifies
    // prompt-lookup drafts in one multi-token forward and enqueues the
    // accepted tokens; iterations drain the queue. specRound* tracks the
    // last round so a mid-queue stop can repair the state to match what
    // was actually emitted.
    let specEnabled = fusedGreedy
        && (fusedRunner?.specDecodeEnabled ?? false)
        && prefillConfig.mode == .chunked
    var specQueue: [Int32] = []
    var specRoundTokens: [Int32] = []   // [anchor, drafts...] of last round
    var specRoundStart = 0              // position of the round's anchor
    var specRoundQueueSize = 0          // emissions the round enqueued

    while true {
        try Task.checkCancellation()

        let tokenID: Int32
        if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                tokenID = Int32(bitPattern: token)
            case .logitsWritten:
                tokenID = sampleOnce(scratch: scratch, context: context,
                                     history: history, config: config, position: generated)
            }
        } else if fusedGreedy {
            tokenID = specQueue.isEmpty
                ? Int32(bitPattern: fusedRunner!.lastGreedyToken)
                : specQueue.removeFirst()
        } else {
            tokenID = sampleOnce(scratch: scratch, context: context,
                                 history: history, config: config, position: generated)
        }
        generated += 1
        uncommittedBoundaryTokenIDs = [tokenID]

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            if tokenID == tokenizer.endOfTurnID {
                reason = .endOfTurn
            } else if tokenID == tokenizer.toolResponseID {
                reason = .toolCalls
            } else {
                reason = .eos
            }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            break
        }

        let delta = detok.push(tokenID)
        let visible = stopMatcher.push(delta)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))

        let hitStopString = stopMatcher.isStopped || shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || hitMax {
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            reason = hitStopString ? .stopString : .maxTokens
            break
        }

        history.append(tokenID)
        if specEnabled, !specQueue.isEmpty {
            // This token was processed by the round already; nothing to do.
            uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
            continue
        }
        if specEnabled,
           let runner = fusedRunner,
           let draft = promptLookupDraft(history: history,
                                         maxDraft: min(8, RealForwardRunner.maxSpecTokens - 1)),
           position + draft.count + 1 <= (producer as? any ContextWindowReporting)
               .map({ $0.maxContext }) ?? Int.max {
            runner.specCheckpoint()
            specRoundStart = position
            specRoundTokens = [tokenID] + draft
            let tVerify = DispatchTime.now().uptimeNanoseconds
            let g = try await runner.specVerifyGreedy(
                tokens: specRoundTokens[...],
                startPosition: position,
                config: prefillConfig,
                into: scratch.logits)
            let verifyNanos = DispatchTime.now().uptimeNanoseconds - tVerify
            var accepted = 0
            while accepted < draft.count,
                  g[accepted] == UInt32(bitPattern: draft[accepted]) {
                accepted += 1
            }
            if accepted < draft.count {
                // Rewind to the round start and replay only what stands:
                // anchor + accepted drafts. The correction token g[accepted]
                // is emitted from the queue and processed on a later
                // iteration (or by the next round).
                runner.specRestoreCheckpoint()
                runner.specRewind(to: specRoundStart)
                let standing = Array(specRoundTokens.prefix(accepted + 1))
                try await runner.specReplay(
                    tokens: standing[...],
                    startPosition: specRoundStart,
                    config: prefillConfig,
                    into: scratch.logits)
                position = specRoundStart + accepted + 1
                runner.specNoteRound(drafted: draft.count, accepted: accepted,
                                     replayed: accepted + 1,
                                     verifyNanos: verifyNanos,
                                     replayNanos: DispatchTime.now().uptimeNanoseconds
                                         - tVerify - verifyNanos)
            } else {
                position = specRoundStart + draft.count + 1
                runner.specNoteRound(drafted: draft.count, accepted: accepted,
                                     replayed: 0, verifyNanos: verifyNanos)
            }
            specQueue = (0...accepted).map { Int32(bitPattern: g[$0]) }
            specRoundQueueSize = specQueue.count
            uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
            continue
        }
        try await producer.produce(token: tokenID, position: position, into: scratch.logits)
        position += 1
        uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
    }

    // A stop that fired while draining a round leaves KV/GDN state ahead of
    // what was emitted; repair so continuation (server prompt cache) sees a
    // state exactly matching `history`.
    if specEnabled, !specQueue.isEmpty, let runner = fusedRunner {
        let delivered = specRoundQueueSize - specQueue.count
        let standing = Array(specRoundTokens.prefix(max(0, delivered - 1)))
        runner.specRestoreCheckpoint()
        runner.specRewind(to: specRoundStart)
        if !standing.isEmpty {
            try await runner.specReplay(
                tokens: standing[...],
                startPosition: specRoundStart,
                config: prefillConfig,
                into: scratch.logits)
        }
        position = specRoundStart + standing.count
    }

    return RawDecodeResult(prefillTokens: promptIds.count,
                           cachedPromptTokens: cachedPromptTokens,
                           computedPrefillTokens: computedPrefillTokens,
                           prefillSeconds: prefillSeconds,
                           newTokens: generated,
                           decodeSeconds: Date().timeIntervalSince(decodeStart),
                           reason: reason,
                           kvPosition: position,
                           kvBackedTokenIDs: history,
                           uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs)
}

/// Prompt-lookup drafting: find the most recent PRIOR occurrence of the
/// history's trailing n-gram (n = 4, then 3, then 2) and propose the
/// tokens that followed it. Zero-cost draft source; strong on code and
/// structured text, harmless elsewhere (rejected drafts still yield the
/// normal token from the same forward).
private func promptLookupDraft(history: [Int32], maxDraft: Int) -> [Int32]? {
    guard maxDraft > 0, history.count >= 8 else { return nil }
    for n in stride(from: 4, through: 2, by: -1) {
        let suffix = Array(history.suffix(n))
        let searchEnd = history.count - n   // exclude the suffix itself
        guard searchEnd > n else { continue }
        var i = searchEnd - 1
        while i >= n - 1 {
            var match = true
            for j in 0..<n where history[i - n + 1 + j] != suffix[j] {
                match = false; break
            }
            if match {
                let start = i + 1
                let count = min(maxDraft, history.count - n - start)
                if count >= 1 {
                    return Array(history[start..<start + count])
                }
                break
            }
            i -= 1
        }
    }
    return nil
}

private func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                        history: [Int32], config: GenerationConfig, position: Int) -> Int32 {
    let cb = context.queue.makeCommandBuffer()!
    scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
                           history: history, config: config, position: position,
                           outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    return Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
}
