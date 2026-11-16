# Security Policy

## Reporting a Vulnerability

Please use GitHub's **Private Vulnerability Reporting**
(Security tab → "Report a vulnerability") on this repository.
Do not open public issues for security reports.

## Scope

genai-nodes sends prompts, feature values, context text and (when
`send_images` is on) images to the provider chosen in `provider`, over
HTTPS. What a provider keeps or trains on is set by that provider and
your account; see the README.

- **API keys** are environment values: the graph stores only the value's
  name (`api_key_env`). The key is read when a request is made, and is
  masked in error outputs and logs.
- **Replies are untrusted data.** Text, JSON and schema fields are
  dispatched into the graph; the plugin never executes them. A prompt,
  context or image can steer a reply (prompt injection), so a graph or app
  that acts on replies should validate them.
- **Cost guard, not a security boundary.** Requests start only from a
  trigger and are spaced (`min_interval_s`) and capped per UTC day
  (`max_requests_per_day`). The daily count is a file in the graph's data
  folder; anyone who can edit the workspace can reset it. Also set a
  spending limit with the provider.
- The plugin opens no listening sockets and does not execute code from
  files or graph definitions.

The host runtime it plugs into (the nonoha engine) may include script
nodes with their own threat model — see the security policy of the host
project for loading untrusted graphs.

## Supported versions

Security fixes target the latest minor release.
