# Contributing to genai-nodes

Thank you for considering a contribution! This document covers how to
make changes to this repository (pull requests and issues). For
development setup and usage, see the [README](README.md).

## Issues

- Bug reports and feature requests: the issue templates will guide you.
- Security issues: do **not** open a public issue — see
  [SECURITY.md](SECURITY.md).
- Never paste an API key into an issue or a pull request.

## Contributor License Agreement (CLA)

Your first pull request triggers a bot comment. Reply with:

> I have read the CLA Document and I hereby sign the CLA

— that's it. The signature covers all your future contributions and is
shared across the nonoha repositories.

Why a CLA? Parts of nonoha are dual-licensed to fund development. You
keep your copyright; you grant VIE a license that includes relicensing.
See [CLA.md](CLA.md) for the full text.

*Corporate contributors: if your employer requires a corporate
agreement, contact VIE before submitting.*

## Pull request flow

1. Fork the repository and create a branch.
2. Make your change and add tests.
3. Ensure `uv run pytest` passes (the dev dependency group provides
   `nonoha-core`, which the nodes are registered against).
4. Open a pull request (the template will guide you).

## Coding guidelines for changes

- Comments and docstrings are written in English.
- Tests never call a provider: fake the provider or the SDK module.
- Requests start only from a trigger. Pacing, the daily cap and retries
  belong to the runner; providers turn their SDK's own retries and
  continuations off, so one counted request is one call.
- API keys are read when a request is made and never stored, logged or
  dispatched. Error texts pass through the runner, which masks the key.
- Outputs follow nonoha-core's origin rule: values reported inside the
  trigger's dispatch are not flushed; the timer and the request task flush
  once at their end.
- Dispatched values are valid only during the dispatch: copy what a node
  keeps.
- Keep the registered node names and port names stable; graphs reference
  them by name.
