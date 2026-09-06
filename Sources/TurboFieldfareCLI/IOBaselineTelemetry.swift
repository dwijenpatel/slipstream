import Darwin
import Foundation
import TurboFieldfare

/// Opt-in benchmark windows. OS disk bytes are process-wide, not an expert-only
/// counter; compare against logical expert bytes only after prefill has finished.
final class IOBaselineTelemetry {
    private let output: FileHandle
    private let stride: UInt64
    private var previous: (index: Int, time: UInt64, disk: UInt64, misses: UInt64, io: UInt64)?
    private var lastIndex = -1

    init(output: FileHandle, stride: UInt64, cacheEnabled: Bool) {
        self.output = output
        self.stride = stride
        output.write(Data("[io-mode expert_file_cache=\(cacheEnabled ? "default" : "nocache") integrity=full-sha256]\n".utf8))
    }

    func token(index: Int, runner: RealForwardRunner, force: Bool = false) {
        lastIndex = index
        guard previous == nil || force || index - previous!.index >= 128 else { return }
        var usage = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &usage) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        guard result == 0 else {
            output.write(Data("[io-window error=proc_pid_rusage errno=\(errno)]\n".utf8))
            return
        }
        let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        if let p = previous, index > p.index {
            let elapsed = Double(now - p.time) / 1e9
            let logical = (runner.expertMisses - p.misses) * stride
            let disk = usage.ri_diskio_bytesread - p.disk
            let line = String(format: "[io-window first=%d last=%d tokens=%d seconds=%.6f tok_s=%.3f expert_bytes=%llu disk_read_bytes=%llu io_await_ms=%.3f footprint_bytes=%llu]\n",
                              p.index + 1, index, index - p.index, elapsed,
                              Double(index - p.index) / elapsed, logical, disk,
                              Double(runner.totalIoNanos - p.io) / 1e6, usage.ri_phys_footprint)
            output.write(Data(line.utf8))
        }
        previous = (index, now, usage.ri_diskio_bytesread, runner.expertMisses, runner.totalIoNanos)
    }

    func finish(runner: RealForwardRunner) {
        guard lastIndex >= 0 else { return }
        token(index: lastIndex, runner: runner, force: true)
    }
}
