// Read-only cache-control check; never evicts pages or changes file contents.
// clang playbook/expert_io_probe.c -o /tmp/expert-io-probe
// /tmp/expert-io-probe PATH_TO_LAYER 0|1
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <libproc.h>
#include <sys/resource.h>
#include <time.h>

static unsigned long long disk_bytes(void) {
    struct rusage_info_v4 usage = {0};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&usage)) {
        perror("proc_pid_rusage"); exit(1);
    }
    return usage.ri_diskio_bytesread;
}

int main(int argc, char **argv) {
    if (argc != 3 || (argv[2][0] != '0' && argv[2][0] != '1') || argv[2][1]) {
        fprintf(stderr, "usage: %s layer-file 0|1\n", argv[0]); return 2;
    }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    int nocache = argv[2][0] == '1';
    if (nocache && fcntl(fd, F_NOCACHE, 1)) { perror("F_NOCACHE"); close(fd); return 1; }
    void *buffer = NULL;
    // Qwen3.6 packed expert geometry, matching the runtime's alignment.
    size_t stride = 1769472;
    int error = posix_memalign(&buffer, 2097152, stride);
    if (error) { fprintf(stderr,"allocation error %d\n",error); close(fd); return 1; }
    for (int pass = 0; pass < 3; pass++) {
        unsigned long long before = disk_bytes();
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int expert = 0; expert < 16; expert++) {
            if (pread(fd, buffer, stride, expert * stride) != (ssize_t)stride) {
                fprintf(stderr,"short or failed read\n"); free(buffer); close(fd); return 1;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("nocache=%d pass=%d logical=%zu physical=%llu seconds=%.6f\n",
               nocache, pass, stride * 16, disk_bytes() - before,
               end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9);
    }
    free(buffer); close(fd); return 0;
}
