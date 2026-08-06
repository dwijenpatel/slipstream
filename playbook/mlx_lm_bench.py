"""Resident-weight baseline for fill_table.sh.

Reports time to first token and sustained decode on the same prompt files
the streaming arms use, so the numbers land in the same table honestly.

The first run against a fresh process also materializes the lazily mapped
weights, roughly 19 GB, which is charged to time-to-first-token and is not
comparable to a warm streaming row. fill_table.sh notes this; do not quote
that TTFT as prefill.
"""

import json
import sys
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

model_path, prompt_path, max_new = sys.argv[1], sys.argv[2], int(sys.argv[3])

messages = json.load(open(prompt_path))
model, tokenizer = load(model_path)
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(f"prompt tokens: {len(prompt)}", flush=True)

sampler = make_sampler(temp=0.0)
start = time.perf_counter()
ttft = None
count = 0
for _ in stream_generate(model, tokenizer, prompt, max_tokens=max_new, sampler=sampler):
    if ttft is None:
        ttft = time.perf_counter() - start
    count += 1
decode_seconds = time.perf_counter() - start - ttft

print(f"TTFT {ttft:.2f} s")
print(f"{count} tokens in {decode_seconds:.2f} s = {count / decode_seconds:.2f} tok/s")
print(f"mlx peak {mx.get_peak_memory() / 1e9:.2f} GB")
