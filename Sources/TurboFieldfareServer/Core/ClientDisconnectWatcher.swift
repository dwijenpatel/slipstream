import Foundation
import NIOCore

/// Cancels in-flight generation when the client goes away.
///
/// WHY THIS IS A SEPARATE HANDLER AT THE FRONT OF THE PIPELINE.
/// `configureHTTPServerPipeline(withPipeliningAssistance: true)` deliberately
/// suppresses `read()` while a response is outstanding, so it does not buffer
/// unbounded pipelined requests. The side effect is that the socket is not
/// polled at all during a long generation, so a client that disconnects is
/// invisible until the work finishes. Measured 2026-08-05: an abandoned
/// request kept the GPU at 70 percent for 47 minutes with zero connections
/// open, and because the server serves one request at a time, every retry
/// queued behind it. That is what killed the kb-build agent run.
///
/// Turning the assistance off does deliver the disconnect, but it also breaks
/// response ordering for pipelined requests, which an existing test covers.
/// So instead this handler sits AHEAD of the HTTP handlers, where two things
/// are true: inbound socket events reach it first, and an outbound `read()`
/// issued from here travels straight to the head without passing through the
/// pipelining handler that would suppress it.
///
/// One `read()` at the start of a generation is enough. It arms read interest,
/// and the EOF that follows a disconnect is then delivered as
/// `ChannelEvent.inputClosed` (the channel is configured with
/// `allowRemoteHalfClosure`) or as `channelInactive` on a full close.
final class ClientDisconnectWatcher: ChannelInboundHandler {
    typealias InboundIn = ByteBuffer
    typealias InboundOut = ByteBuffer

    private let cancellation: GenerationCancellation
    private var context: ChannelHandlerContext?

    init(cancellation: GenerationCancellation) {
        self.cancellation = cancellation
    }

    func handlerAdded(context: ChannelHandlerContext) {
        self.context = context
        cancellation.armRead = { [weak self] in
            guard let self, let context = self.context else { return }
            // Hop to the event loop: the generation starts from a Task, not
            // from the loop that owns this channel.
            context.eventLoop.execute { context.read() }
        }
    }

    func handlerRemoved(context: ChannelHandlerContext) {
        self.context = nil
        cancellation.armRead = nil
    }

    func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
        if case ChannelEvent.inputClosed = event {
            cancellation.cancel(reason: "client closed the connection")
        }
        context.fireUserInboundEventTriggered(event)
    }

    func channelInactive(context: ChannelHandlerContext) {
        cancellation.cancel(reason: "connection went inactive")
        context.fireChannelInactive()
    }
}

/// Shared between the watcher at the front of the pipeline and the HTTP
/// handler at the back, which is the one that owns the generation task. Both
/// run on the same event loop, so no locking is needed.
final class GenerationCancellation: @unchecked Sendable {
    private var task: Task<Void, Never>?
    var armRead: (() -> Void)?

    /// Called by the HTTP handler when it starts generating. Arms read
    /// interest so the socket is polled while the GPU works.
    func begin(_ task: Task<Void, Never>) {
        self.task = task
        armRead?()
    }

    func finish() {
        task = nil
    }

    func cancel(reason: String) {
        guard let task else { return }
        task.cancel()
        self.task = nil
        FileHandle.standardError.write(
            Data("cancelled in-flight generation: \(reason)\n".utf8))
    }
}
