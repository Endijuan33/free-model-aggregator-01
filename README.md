# free-model-aggregator-01 — The best free AI model aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![make.fun · free-model-aggregator-01 · Endijuan33/free-model-aggregator-01 · rank](https://make.fun/badge/free-model-aggregator-01/Endijuan33/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![make.fun · free-model-aggregator-01 · Endijuan33/free-model-aggregator-01 · reward](https://make.fun/badge/free-model-aggregator-01/Endijuan33/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that answers every request with the best **free** model it can get — over ten
simulated free-tier providers in the trial, and over a live free-model directory in real use.
It speaks both client wires (OpenAI Chat Completions and Anthropic Messages, JSON and SSE) and all
three upstream wires (OpenAI, Anthropic, Gemini), routes by the quality table, and fails over the
moment a provider refuses. No model call of its own, no key in the repo, nothing ever bought.

## How it scores

Practice trials (the kit's three, run locally with `grade-local --driver process`):

| trial | score | quality | served | wire | failover | catalog | upstream calls |
|-------|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |

All three replays matched byte for byte. The house baseline scores 38 on trial-0. The one
point short of 100 is quality: every provider is drawn with 1–2 scheduled refusals, so a few
requests must fail over to the next-best free model (1475 → 1470). No routing can dodge a refusal
whose position it cannot see; 99 is the practical ceiling of the draw.

## Routing

For each request, every free model that meets the request's requirements is a candidate, ranked
by the quality table's `rating_milli`. The request goes to the best candidate; on a refusal
(402, 429, 5xx) it fails over to the next candidate. A paid model is never a legal pick.

The router learns each provider's budget from its answers, starting from the kit's published
sealed table (every provider's per-trial budget is drawn inside `[calls_min, calls_max]`, its
first call always answers, and exactly one provider per instance goes down from some call on):

- a **402** means that provider's budget is spent: blacklisted for the rest of the trial
  (Pollinations is the 402 provider);
- **two consecutive refusals** mean the provider is spent or down: blacklisted too;
- once a provider has answered `calls_max` calls, its budget cannot hold another: blacklisted.

A single 429 is ambiguous (a scheduled one-off or a spent budget), so the provider stays in
rotation for later requests but is deprioritised for the current one. Ties between equally rated
models go to the provider with more budget left, so two equally rated models spread over two
providers instead of draining one. Every choice is a pure function of the brief, the request and
the providers' answers — no wall clock, no randomness — so a replay leaves the same state export,
byte for byte.

## The wires

`lib/wire.ts` translates every (client wire × upstream wire) combination, JSON and SSE, through
one canonical request shape, so no field is dropped: system prompt, user text, image bytes, tool
names, `max_tokens`, stop sequences and JSON mode all round-trip exactly, and the answer text,
stop reason and usage tokens come back in the client's wire. Two things the baseline lacked:

- a streamed OpenAI answer carries its usage only when the upstream request asks for it, so every
  streamed upstream call sets `stream_options.include_usage`;
- an Anthropic request cannot carry JSON mode, so JSON-mode requests only ever go to an
  openai- or gemini-wire provider (Gemini carries it as `generationConfig.responseMimeType`).

## Real deployment — a live free-model list, multi-provider failover, BYOK keys

The same code runs against the real services. `providers.real.json` is the live free-model
directory (OpenRouter's `:free` models, Groq's free plan, Gemini's free tier, Pollinations), and
`--providers` serves it:

```sh
# Keys are bring-your-own, from the environment. No key is stored in this repo.
export OPENROUTER_API_KEY=...   # optional: Pollinations needs no key at all
export GROQ_API_KEY=...
export GEMINI_API_KEY=...
node main.ts --providers providers.real.json --list   # print the catalog and exit
node main.ts --providers providers.real.json          # serve on a free port
```

The live list (checked 2026-11-15, refreshed daily by the sources below):

- **OpenRouter `:free`** — 14 models, any ID ending in `:free` costs nothing per token:
  `openrouter/free` (the auto-router: picks a free model that supports what the request needs),
  `nvidia/nemotron-3-ultra-550b-a55b:free`, `google/gemma-4-31b-it:free`,
  `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3.5-super-vl:free`,
  `inclusionai/ling-3.2-flash:free`, `baidu/ernie-5.5-flash-preview:free`,
  `thinkingmachines/inkling-2-small:free`, `particle-ai/particle-1:free`,
  `liquid/lfm-3-1.7b:free`, `recursive/flint-v1.0:free`, `dots-studio/dots-3-note-preview:free`,
  `nvidia/nemotron-3.5-lightning:free`, `cohere/north-mini-code:free`,
  `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`. Limits: 20 req/min, 50 req/day with no
  credits bought, 1,000 req/day after $10 of credits bought — **this entry never buys credits**;
  50/day is the free lane.
- **Groq free plan** — `gpt-oss-120b` (30 RPM), `qwen3.8-27b`, `llama-3.1-8b-instant`:
  1,000 req/day per model, 200K tokens/day.
- **Gemini free tier** — `gemini-3.8-flash` (1M context, tools + vision + JSON), `gemma-4-31b`:
  no card required.
- **Pollinations** — `openai-fast`, `mistral-nemo`: no key required.

Deliberately **not** in the directory: GitHub Models (retired July 2026), Cerebras and SambaNova
(now need cards). The list changes often — free models come and go, usually previews that later
become paid — so re-check the sources when a model stops working. The trial's simulated directory
is a different list (the brief's ten providers, cerebras and sambanova included): the trial is
graded against the brief, not this file.

Sources: <https://byoklist.com/guides/openrouter-free-models>,
<https://www.freellm.site/blog/best-free-llm-apis-2026/>,
<https://gravity.fast/data/free-llm-api-tiers/>

## Running a trial

```sh
# In the kit directory, with this repo as the bot:
node cli/makefun.mjs grade-local --kit <kit> --bot <this repo> --driver process --instances 3
```

The trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE` and `MAKEFUN_TRIAL_SEED`. The server reads
the brief, probes each provider's catalog route in its own wire, registers with the env, serves the
20 requests, polls `GET $MAKEFUN_ENV_URL/work` and exits 0 when the env says done.

## Files

- `main.ts` — the server: brief, catalog, routing, failover, the register and the poll.
- `lib/providers.ts` — the brief's three documents and the catalog probe.
- `lib/wire.ts` — the client wires and all three upstream wires, JSON and SSE.
- `providers.real.json` — the live free-model directory for real deployment.
- `LICENSE` — MIT.

## License

SPDX-License-Identifier: MIT. See `LICENSE`.
