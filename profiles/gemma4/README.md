# Gemma 4 26B-A4B profile

Partially measured (M5, 24 GB, 2026-08-01). 30 layers with sliding-window
(1024) plus full-attention layers; 128 experts/layer, topk 8, stride
3.36 MB; packed experts 12 GB.

Measured so far: community protocol with `--prefill-chunk auto` cuts
long-synthesis wall 86.1 to 60.1 s and prefill reads 205 to 50 GB,
byte-identical outputs, flat RSS (KV ring sized from configured chunk).
Slots curve, decode phase split, and per-tier settings not yet run.
