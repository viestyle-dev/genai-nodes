# genai-nodes

A nonoha-engine plugin that calls generative AI models: LLMs describe images
or interpret signals and features. Requests are event-driven (only on
`trigger`), paced, and capped per day, so a graph never calls an API in a
loop.

[genai-nodes on nonoha Hub](https://hub.nonoha.dev/plugins/official/nonoha-official/genai-nodes/): the published versions, and the engine versions and platforms they support.

## Overview

The nodes call models over HTTPS (cloud APIs or servers you run), so the
plugin stays light and runs wherever the engine runs. Running models inside
the engine process (ONNX, PyTorch, llama.cpp bindings) is out of scope.

## Nodes

- **LLMDescribeImage**: sends the latest encoded image (JPEG/PNG bytes, e.g.
  from cv-nodes' `Imencode`) with a prompt and outputs the reply. With a JSON
  Schema in `schema`, it also outputs the parsed JSON.
- **LLMInterpret**: sends named feature values (plus optional context text
  and image) and outputs the interpretation. Every top-level
  number/string/boolean field of `schema` becomes an output port `out_<field>`.

Both have `text`, `json`, `busy`, `error` and `usage` outputs. Without a key,
or when the provider is unreachable, they only report on `error`; the rest of
the graph keeps running.

Every property also has an input port (`@<name>`, for example `@send_images`), so another node's output can set it.

## Providers and API keys

| `provider` | Key name (default) | Get a key | Cost | Your data |
| --- | --- | --- | --- | --- |
| `gemini` (default) | `GEMINI_API_KEY` | <https://aistudio.google.com/apikey> | Free tier with small, changing limits | Google may use free-tier inputs to improve its products |
| `openrouter` | `OPENROUTER_API_KEY` | <https://openrouter.ai/settings/keys> | Free `:free` models (50 requests a day) | Some providers train on inputs; opt out in OpenRouter's settings |
| `anthropic` | `ANTHROPIC_API_KEY` | <https://platform.claude.com/settings/keys> | Paid (prepaid API credits, separate from Claude plans) | API inputs are not used for training by default |

The key is an **environment value**: the graph stores only its *name*
(`api_key_env`, empty = the default above), never the key.

Set the value in one of these places:

- **While the engine runs**: in the editor, open *Environment settings* from
  the top-right menu. The next request uses the new key, with no restart.
- **While the engine is stopped**: in the nonoha Hub's workspace page
  (*Environment values*), or with `nonoha ws env set <workspace> GEMINI_API_KEY`
  (it asks for the value; never put it on the command line).
- **By hand**: in the workspace's `.env` file (`GEMINI_API_KEY="..."`). The
  file is never published, exported or copied into a build. Keep it out of
  git (`.env` in `.gitignore`).
- **OS environment variable**: it takes precedence over `.env` (useful for
  systemd or CI), but it applies to every workspace of that user.

To use different keys within one workspace, give each node its own name in
`api_key_env` (e.g. `GEMINI_API_KEY_SUB`).

## Costs and limits

- `min_interval_s` spaces requests; `max_requests_per_day` caps them per UTC
  day. One counted request is one HTTP call: the SDKs' own retries (and
  Gemini's automatic continuation) are turned off, and a rate-limited
  request is re-sent once, after the pause the provider asks for.
- With a paid provider the worst case per day is `max_requests_per_day` ×
  the cost of one request (`max_tokens` caps its output); `usage` reports
  each request's token counts. Also set a spending limit on the provider's
  console.
- The daily count is kept in the graph's data folder
  (`<graph>.vng/data/genai-nodes-usage.json`). It is per machine: exclude it
  from publishing in the workspace's `pyproject.toml`:

  ```toml
  [tool.nonoha-hub.publish]
  exclude = ["flows/default.vnf/graphs/app.vng/data/genai-nodes-usage.json"]
  ```

## Privacy

- LLMDescribeImage sends the image it received (turn `send_images` off to
  block it); LLMInterpret includes an image only with `send_images` on. Do
  not send live camera images of people to a free tier.
- Gemini requests are sent with `store` off, so the provider keeps no
  interaction for later retrieval.
- Keys never appear in outputs, errors or logs: error texts are masked.

## Mobile and embedded hosts

The vendor SDKs are not installed on Android/iOS. An app should not embed a
developer's key; call your own relay server instead. A key the user enters is
stored by the app (Keychain/Keystore) and handed to the runtime as an
environment value.

## Installation

### From the Repository

```sh
pip install git+https://github.com/viestyle-dev/genai-nodes.git
```

Requires Python 3.11+ and a nonoha engine with environment values (1.1 or
later). `nonoha-core` is provided by the host runtime (the nonoha engine), so
it is not a runtime dependency of this package.

### As a Dependency in nonoha-engine Workspace

1. Add `genai-nodes` to your workspace's `pyproject.toml`:

   ```toml
   [project]
   dependencies = [
       "genai-nodes",
       # other dependencies...
   ]

   [tool.uv.sources]
   genai-nodes = { path = "./plugins/genai-nodes", editable = false }
   ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

## Requirements

- Python >= 3.11
- nonoha-core >= 1.1 (nonoha-engine core library, provided by the host)
- httpx2 (OpenRouter)
- google-genai (Gemini) and anthropic (Claude), except on Android/iOS

## Development

```bash
uv sync        # also installs the dev dependency group (pytest, nonoha-core)
uv run pytest  # no network: providers and SDK modules are faked
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Pull requests require signing the [nonoha CLA](CLA.md) (a one-time
comment on your first PR, shared across the nonoha repositories).

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
