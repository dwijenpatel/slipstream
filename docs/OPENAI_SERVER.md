# Local server

`slipstream-server` exposes one model on the loopback interface through the
OpenAI Chat Completions API and the Anthropic Messages API, and serves a chat
page at its root. It binds to `127.0.0.1` without authentication or TLS. Do
not expose it through a proxy or tunnel.

## Start the server

Install the model with `slipstream-repack` first. Then check that no other
model process is running:

```bash
pgrep -fl 'slipstream|TurboFieldfare|llama-server|mlx_lm'
```

If the command prints a match, do not start the server.

```bash
swift build -c release --product slipstream-server
.build/release/slipstream-server --model ~/models/qwen36.gturbo --port 8091 --max-context 65536
```

The server loads the model before opening the port. Wait for the ready line,
then keep the process running while clients use it. Press Control-C, or send
SIGTERM, to stop it; a generation in progress is cancelled first.

Options:

| Flag | Default | Effect |
| --- | --- | --- |
| `--model <dir>` | required | A completed `.gturbo` directory. |
| `--port <n>` | 8080 | Loopback port. |
| `--model-id <id>` | `qwen3.6-35b-a3b` for Qwen, `gemma-4-26b-a4b-it` for Gemma | The identifier reported by `/v1/models` and accepted in requests. |
| `--max-context <tokens>` | 16384 | 4096, 8192, 16384, 32768, or 65536. |
| `--queue-limit <n>` | 4 | Requests queued behind the one running. |
| `--prompt-cache-mode <off\|single-prefix>` | `single-prefix` | Whether to keep one conversation's KV prefix for reuse. |
| `--expert-cache-slots <n>` | 64 | Routed-expert slots per layer. More is not always faster; see `RUNTIME_CONTROLS.md`. |
| `--expert-cache-policy <lfu\|lru>` | `lfu` | Replacement policy for those slots; see `RUNTIME_CONTROLS.md`. |

The server prefills in 4,096-token chunks, defaults to temperature 0, caps a
completion at 4,096 tokens unless the request asks for fewer, and selects the
greedy fused head or the sampling head per request. It logs one line per
request to stderr, with these fields in this order: an ISO 8601 timestamp,
`prompt` and `cached` token counts, `new_prompt` (the tokens prefilled),
`completion`, `ttft`, `prefill` and `decode` rates, `total`, the stop reason,
a `cache_miss` reason when the prefix cache missed, the `render`, `match`, and
`prepare` times spent outside prefill, the request's expert-cache hit rate and
miss count, the expert I/O await, and the number of reasoning tokens.

`ttft` counts prefill only, so the three times after the stop reason are wait
the client saw that `ttft` did not. The model's reasoning is withheld from the
client and written to stderr as a delimited block after the request line.

## Check it

```bash
curl --silent --show-error http://127.0.0.1:8091/health
curl --silent --show-error http://127.0.0.1:8091/v1/models
curl --silent --show-error http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "max_completion_tokens": 16
  }'
```

Open `http://127.0.0.1:8091/` in a browser for the built-in chat page.

## Connect a client

The OpenAI base URL is `http://127.0.0.1:8091/v1`. Client libraries that
require an API key can send any string; the server ignores it.

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8091/v1", api_key="local")
response = client.chat.completions.create(
    model="qwen3.6-35b-a3b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

Claude Code: `Scripts/claude-local.sh` sets `ANTHROPIC_BASE_URL` to the server
and trims the opening request from about 33k tokens of tool schemas and
memory to a few thousand, which matters at local prefill speeds. Start the
server on port 8091 first, then run the script with `-p "task"` for a one-shot
or with no arguments for a session. One client at a time: the server has one
slot, and an abandoned request is cancelled only when its client disconnects.

OpenCode:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "slipstream": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "slipstream",
      "options": {
        "baseURL": "http://127.0.0.1:8091/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "Qwen3.6 35B-A3B"
        }
      }
    }
  }
}
```

## Prompt reuse

Single-prefix KV reuse is on by default. Send the complete message history with
every request. When a request continues the retained conversation exactly, the
server reuses the verified KV prefix and reports the number of reused tokens
in `usage.prompt_tokens_details.cached_tokens`, and in the request log line as
`cached`.

The server retains one prefix. A different or incompatible history replaces
it, and a miss costs a full prefill. The reasons a lookup can miss are named
in the source (`ServerPromptCacheMiss`): no entry yet, the model or template
changed, the tool set changed, the history is not an extension of what was
served, the assistant turn came back altered, or the continuation is not a
shape the server can bridge. Use `--prompt-cache-mode off` to disable reuse.

## Tool calls

The server returns OpenAI-style function calls and Anthropic-style tool-use
blocks, but it cannot authorize or execute them. The client runs the tool
loop:

1. Send function schemas in `tools`.
2. When `finish_reason` is `"tool_calls"`, inspect each function name and JSON
   argument object. Apply the client's normal permission checks before running
   the function.
3. Append the assistant message, including its unchanged `tool_calls`.
4. Append each result as a `role: "tool"` message. Its `tool_call_id` must
   match the call it resolves.
5. Send the complete history and tool schemas again.

The server accepts only function tools. Omit `tool_choice` or set it to `auto`
to allow calls. Set it to `none` to disable them. The server does not support
`required`, named tool selection, or `parallel_tool_calls: false`.

## Supported API

Endpoints:

- `GET /` and `GET /index.html`: the chat page.
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`: OpenAI Chat Completions, JSON or Server-Sent
  Events with `"stream": true`; `"stream_options": {"include_usage": true}`
  adds a final usage chunk.
- `POST /v1/messages`: Anthropic Messages, JSON or streamed, translated onto
  the same generation path. System prompts, tool definitions, and tool-use
  and tool-result blocks are carried through; thinking blocks in a request
  are dropped, and the model's own reasoning goes to the server log, not to
  the client.

Requests may contain system, developer, user, assistant, and tool messages.
Supported options include `temperature`, `top_p`, `top_k`,
`repetition_penalty`, `seed`, `stop`, `max_tokens`,
`max_completion_tokens`, `response_format` of type `text`, and function-tool
fields.

A top-level field the server does not declare is refused with a 400 whose
`code` is `unknown_parameter` and whose `param` names the field, so a
misspelled option such as `max_token` cannot run the request under other
settings. Real OpenAI parameters the server cannot honor (`logit_bias`,
`top_logprobs`, `reasoning_effort`, `verbosity`, `modalities`, `audio`,
`prediction`, `web_search_options`, and the legacy `functions` and
`function_call`) are refused with `unsupported_value`. Caller-side
bookkeeping fields (`user`, `store`, `metadata`, `service_tier`,
`prompt_cache_key`, `safety_identifier`) are accepted and ignored. A field
set to `null` is treated as absent, which is what openai-python sends for an
unset option.

A request refused by the memory guard returns status 503 with code
`memory_budget_exceeded` and a message naming the footprint, the predicted
growth, and the cap. See the memory guard section of `RUNTIME_CONTROLS.md`.

The server supports one model and one choice. It does not support the
Responses API, legacy Completions, embeddings, image input, structured output,
batching, log probabilities, or model switching.

Context length can be 4K, 8K, 16K, 32K, or 64K. Larger contexts use more KV
memory: about 84 MB per 4K on Qwen3.6, whose linear-attention layers keep a
fixed-size state instead of a cache. Run one model process at a time.
