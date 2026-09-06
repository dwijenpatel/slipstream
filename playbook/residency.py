#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Print how much of a model's expert pool is resident in the page cache,
without touching the data: mincore over every file in the directory.

    residency.py ~/models/qwen36.gturbo/packed_experts
"""
import ctypes, mmap, os, sys

PROT_READ, MAP_SHARED = 1, 1


def resident_bytes(directory: str):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    page = mmap.PAGESIZE
    total = resident = 0
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        size = os.path.getsize(path)
        if size == 0:
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), "mmap")
            pages = (size + page - 1) // page
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(addr, size, vec):
                raise OSError(ctypes.get_errno(), "mincore")
            resident += sum(1 for b in vec if b & 1)
            total += pages
            libc.munmap(addr, size)
        finally:
            os.close(fd)
    return resident * page, total * page


if __name__ == "__main__":
    r, t = resident_bytes(sys.argv[1])
    print(f"resident {r / 1e9:.2f} GB of {t / 1e9:.2f} GB ({100 * r / max(1, t):.1f}%)")
