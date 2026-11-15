# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/almalikgroup-cmd/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/almalikgroup-cmd/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free-model aggregator: one long-running Node program (standard library only, fully deterministic) that serves every client request with the best free model it can get. It reads the trial brief (provider directory, quality table, workload manifest), probes all ten providers' catalogs in their own wires, registers one catalog with the env, then answers OpenAI- and Anthropic-shaped client requests — JSON or SSE — by routing each one to the highest-rated free model that meets the request's requirements (tools, vision, JSON mode, minimum context).

All three upstream wires are spoken: OpenAI, Anthropic and Gemini. Requests are translated field-for-field (system text, last user text, tool names in order, max_tokens, stop sequences, image bytes, JSON mode) and answers are rendered back in the client's own wire and shape. When a provider refuses (402/429/5xx) or answers with an unusable body, the request falls through to the next-best candidate, so exactly one successful provider call answers each request; calls spread across providers on rating ties, and a provider that fails twice in a row is demoted so a spent budget or an outage cannot take later requests down with it. An OpenAI request that asks for JSON mode is never routed to an Anthropic-wire provider, whose answers cannot report JSON mode; a Gemini upstream reports it through `responseMimeType` instead.

Local practice scores (kit's `grade-local`, process driver, practice trials 0–2): **99 / 99 / 99** (quality 99, served 100, wire 100, failover 100, catalog 100 on each), against the untouched starter baseline's 38.

## Run it

```sh
node main.ts
# The trial harness sets MAKEFUN_ENV_URL (the env to register with) and
# MAKEFUN_PROMPT_FILE (the brief). `node main.ts --list` prints the catalog
# the entry registers with and exits.
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=fp1aw4h5&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
