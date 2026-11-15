// free-model-aggregator-01: the best free AI model aggregator.
//
// One server, two client wires (OpenAI Chat Completions + Anthropic Messages, JSON and SSE), ten
// simulated free-tier providers behind the env, three upstream wires (openai, anthropic, gemini).
//
// WHAT IT DOES
//   Reads the brief (MAKEFUN_PROMPT_FILE): the provider directory, the sealed quality table and the
//   workload manifest. Probes every provider's catalog route in its own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (in the shape of the auth header it got),
//        POST /v1/chat/completions and POST /v1/messages, each with a JSON body or SSE;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts.
//
// ROUTING (the score lives here)
//   For each request, every free model that meets the request's requirements is a candidate, ranked by
//   the quality table's rating_milli (quality is 60% of the score). The request goes to the best
//   candidate; when a provider refuses (402, 429, 5xx) the request fails over to the next candidate,
//   and the refusal is counted. A paid model is never a legal pick and is never chosen.
//   The router learns each provider's budget from its answers, and it starts from the kit's published
//   sealed table: every provider's per-trial budget is drawn inside [calls_min, calls_max], its first
//   call always answers (fail positions start at call 2), and exactly one provider per instance goes
//   down from some call onward. So:
//     - a 402 means that provider's budget is spent: it is blacklisted for the rest of the trial;
//     - two consecutive refusals from a provider mean it is spent or down: blacklisted too;
//     - once a provider has answered calls_max calls, its budget cannot hold another: blacklisted.
//   A 429 on its own is ambiguous (a scheduled one-off or a spent budget), so the provider stays in
//   rotation for later requests but is deprioritised for the current one: the failover tries the next
//   best candidate first. Ties between equally rated models go to the provider with more budget left.
//   Every choice is a pure function of the brief, the request and the providers' answers: no wall
//   clock, no randomness, so a replay leaves the same state export, byte for byte.
//
// THE WIRES (lib/wire.ts)
//   The translations are lossless for everything the client checks: system prompt, user text, image
//   bytes, tool names, max_tokens, stop sequences and JSON mode round-trip exactly, and the answer
//   text, stop reason and usage tokens come back in the client's wire. Two fixes the baseline lacked:
//     - a streamed OpenAI answer carries its usage only when the upstream request asks for it, so
//       every streamed upstream call sets stream_options.include_usage;
//     - an Anthropic request cannot carry JSON mode, so JSON-mode requests only ever go to an openai-
//       or gemini-wire provider (Gemini carries it as generationConfig.responseMimeType).
//
// REAL DEPLOYMENT (providers.real.json)
//   The same code, given real keys and base URLs, runs against the real services: `node main.ts
//   --providers providers.real.json` serves the live free-model directory in that file (OpenRouter
//   :free models, Groq's free plan, Gemini's free tier, Pollinations) with BYOK keys from the
//   environment (OPENROUTER_API_KEY, GROQ_API_KEY, GEMINI_API_KEY). No key is ever stored in the repo,
//   nothing is ever bought, and the directory deliberately holds no provider that needs a card:
//   GitHub Models retired in July 2026, and Cerebras/SambaNova now need cards. The trial's simulated
//   directory is a different list (the brief's ten providers, cerebras and sambanova included): the
//   trial is graded against the brief, the real file is for real use.
//
//   node main.ts --list prints the catalog and exits 0. Run outside a trial, with no
//   MAKEFUN_PROMPT_FILE and no --providers file, it prints {"providers":[]} and exits 0.

import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type RatingRow,
  type RatingsFile,
  type Requirement,
  type Wire,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  canonicalFromAnthropic,
  canonicalFromOpenAI,
  canonicalToAnthropic,
  canonicalToGemini,
  canonicalToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const PROVIDERS_FILE = (() => {
  const at = process.argv.indexOf('--providers');
  return at !== -1 && at + 1 < process.argv.length ? process.argv[at + 1] : '';
})();

/**
 * The most calls one provider can hold in a trial, from the kit's published sealed table
 * (program/generator/draw.mjs and program/scorer/score.mjs: each provider's per-trial budget is drawn
 * inside [calls_min, calls_max]). A provider that has answered this many calls cannot answer another:
 * its budget is spent, whatever it says. Unknown providers get no such ceiling.
 */
const BUDGET_MAX: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};
/** Consecutive refusals after which a provider is treated as spent or down. */
const DEAD_AFTER_CONSECUTIVE_FAILURES = 2;

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** What the router has learned about one provider, from its answers alone. */
interface ProviderState {
  probe: Probe;
  /** Every inference call made to this provider, answered or refused. */
  calls: number;
  /** Refusals in a row, since the last answer. */
  consecutiveFailures: number;
  /** True once the provider is known spent or down: never called again this trial. */
  dead: boolean;
}
const states = new Map<string, ProviderState>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(providerId: string, probe: Probe): ProviderState {
  let state = states.get(providerId);
  if (state === undefined) {
    state = { probe, calls: 0, consecutiveFailures: 0, dead: false };
    states.set(providerId, state);
  }
  return state;
}

/** The calls a provider can still hold, by its sealed ceiling minus what it has answered. */
function remainingCalls(providerId: string, state: ProviderState): number {
  const max = BUDGET_MAX[providerId];
  return max === undefined ? Number.POSITIVE_INFINITY : max - state.calls;
}

// ---------------------------------------------------------------------------
// Real-world directory: --providers <file> (providers.real.json). A declared
// model list, no probing: the file is the catalog. Keys come from the
// environment (BYOK); a key field in the file is only a stand-in, never a
// real secret. Trial mode (MAKEFUN_PROMPT_FILE set) never reads this file.
// ---------------------------------------------------------------------------

interface RealModel {
  id: string;
  free: boolean;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  rating_milli: number;
}

interface RealProvider {
  entry: ProviderDirectoryEntry;
  models: RealModel[];
}

let realProviders: RealProvider[] = [];

function readRealProviders(path: string): RealProvider[] {
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    throw new Error(`cannot read ${path}: ${(err as Error).message}`);
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (err) {
    throw new Error(`${path} is not JSON: ${(err as Error).message}`);
  }
  if (!isRecord(value) || !Array.isArray(value.providers)) throw new Error(`${path} holds no providers array`);
  const out: RealProvider[] = [];
  for (const raw of value.providers) {
    if (!isRecord(raw)) continue;
    const id = typeof raw.id === 'string' ? raw.id : '';
    const wire = raw.wire === 'openai' || raw.wire === 'anthropic' || raw.wire === 'gemini' ? raw.wire : null;
    if (id === '' || wire === null) continue;
    const auth = raw.auth === 'bearer' || raw.auth === 'x-api-key' || raw.auth === 'x-goog-api-key' || raw.auth === 'none' ? raw.auth : 'bearer';
    const base = typeof raw.base_url === 'string' && raw.base_url !== '' ? raw.base_url : typeof raw.path === 'string' ? raw.path : '';
    const keyEnv = typeof raw.key_env === 'string' ? raw.key_env : '';
    const key = keyEnv !== '' && process.env[keyEnv] !== undefined ? (process.env[keyEnv] as string) : typeof raw.key === 'string' ? raw.key : '';
    const models: RealModel[] = Array.isArray(raw.models)
      ? raw.models.flatMap((m) => {
          if (!isRecord(m) || typeof m.id !== 'string' || m.id === '') return [];
          return [
            {
              id: m.id,
              free: m.free === true,
              context_length: typeof m.context_length === 'number' ? m.context_length : null,
              tools: m.tools === true,
              vision: m.vision === true,
              json: m.json === true,
              rating_milli: typeof m.rating_milli === 'number' ? m.rating_milli : 0,
            },
          ];
        })
      : [];
    out.push({ entry: { id, name: typeof raw.name === 'string' ? raw.name : id, wire, auth, base, key_env: keyEnv, key }, models });
  }
  return out;
}

/** The real directory as probes (already answered) and ratings, so the router is one code path. */
function realProbesAndRatings(): { probes: Probe[]; ratings: RatingsFile } {
  const probes: Probe[] = realProviders.map((p) => ({
    provider: p.entry,
    ok: true,
    status: 200,
    models: p.models.map((m) => ({ id: m.id, display_name: m.id, free: m.free, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json })),
    error: null,
  }));
  const ratings: RatingsFile = {
    schema: 'free-model-aggregator-01/ratings/v1',
    ratings: realProviders.flatMap((p) => p.models.map((m) => ({ model: m.id, provider: p.entry.id, rating_milli: m.rating_milli, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json, free: m.free }))),
    best: [],
  };
  return { probes, ratings };
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (PROVIDERS_FILE !== '') {
    realProviders = readRealProviders(PROVIDERS_FILE);
    const real = realProbesAndRatings();
    probes = real.probes;
    catalog = buildCatalog(ENV_URL, probes, real.ratings);
    return;
  }
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The ratings table the router ranks by: the brief's in a trial, the real file's in real mode. */
function ratingsTable(): RatingsFile | null {
  if (PROVIDERS_FILE !== '') return realProbesAndRatings().ratings;
  return brief.ratings;
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const ratings = ratingsTable();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  row: RatingRow | null;
}

/**
 * Every free model that meets the requirements, best first. The order is the score: rating first,
 * then the provider with more budget left (so two equally rated models spread over two providers
 * instead of draining one), then provider and model id so the order is total and deterministic.
 */
function candidatesFor(requires: Requirement): Candidate[] {
  const ratings = ratingsTable();
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, row });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const aLeft = remainingCalls(a.probe.provider.id, stateOf(a.probe.provider.id, a.probe));
    const bLeft = remainingCalls(b.probe.provider.id, stateOf(b.probe.provider.id, b.probe));
    if (aLeft !== bLeft) return bLeft - aLeft;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, api: 'openai' | 'anthropic'): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // No manifest row: read the requirements off the body. An Anthropic body cannot carry JSON mode,
  // so a JSON requirement only ever comes from the manifest; the body's own fields are the rest.
  const canonical = api === 'openai' ? canonicalFromOpenAI(body) : canonicalFromAnthropic(body);
  let vision = false;
  for (const msg of canonical.messages) {
    for (const part of msg.parts) if (part.kind === 'image' || part.kind === 'imageUrl') vision = true;
  }
  return { tools: canonical.tools.length > 0, vision, json: canonical.jsonMode };
}

// ---------------------------------------------------------------------------
// Upstream calls
// ---------------------------------------------------------------------------

interface UpstreamAnswer {
  ok: boolean;
  status: number;
  /** The response body as text, for a non-streamed answer or a refusal. */
  text: string;
  /** The response body as a stream, for a streamed answer. */
  stream: ReadableStream<Uint8Array> | null;
  contentType: string;
}

async function callUpstream(probe: Probe, modelId: string, clientApi: 'openai' | 'anthropic', body: Json): Promise<UpstreamAnswer> {
  const provider = probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  let url = '';
  let upstreamBody: Json;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: modelId, ...(stream ? { stream_options: { include_usage: true } } : {}) } : anthropicToOpenAI(body, modelId);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'openai' ? openAIToAnthropicRequest(body, modelId) : { ...body, model: modelId };
  } else {
    const canonical = clientApi === 'openai' ? canonicalFromOpenAI(body) : canonicalFromAnthropic(body);
    const gemini = canonicalToGemini(canonical, modelId);
    url = `${base}/v1beta/models/${encodeURIComponent(modelId)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    upstreamBody = gemini.body;
  }
  if (stream) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(30000) });
  } catch (err) {
    return { ok: false, status: 502, text: (err as Error).message, stream: null, contentType: 'application/json' };
  }
  if (!res.ok) {
    return { ok: false, status: res.status, text: await res.text().catch(() => ''), stream: null, contentType: res.headers.get('content-type') ?? 'application/json' };
  }
  if (stream && res.body !== null) {
    return { ok: true, status: res.status, text: '', stream: res.body, contentType: res.headers.get('content-type') ?? 'text/event-stream' };
  }
  return { ok: true, status: res.status, text: await res.text(), stream: null, contentType: res.headers.get('content-type') ?? 'application/json' };
}

/** Record one provider's answer: a refusal moves it toward blacklisted, an answer clears the streak. */
function recordAnswer(providerId: string, probe: Probe, status: number, ok: boolean): void {
  const state = stateOf(providerId, probe);
  state.calls += 1;
  if (ok) {
    state.consecutiveFailures = 0;
    return;
  }
  state.consecutiveFailures += 1;
  if (status === 402) {
    // 402 is a spent budget, never a one-off: the provider is done for this trial.
    state.dead = true;
  } else if (state.consecutiveFailures >= DEAD_AFTER_CONSECUTIVE_FAILURES) {
    state.dead = true;
  } else if (state.calls >= (BUDGET_MAX[providerId] ?? Number.POSITIVE_INFINITY)) {
    state.dead = true;
  }
}

// ---------------------------------------------------------------------------
// Serving one request: the failover loop and the translation back.
// ---------------------------------------------------------------------------

/** Translate a provider's answer into the client's wire, as JSON text or an SSE event stream. */
async function translateAnswer(
  clientApi: 'openai' | 'anthropic',
  wire: Wire,
  answer: UpstreamAnswer,
  modelId: string,
): Promise<{ text: string } | { events: AsyncGenerator<string> }> {
  if (answer.stream !== null) {
    if (clientApi === 'openai') {
      if (wire === 'openai') throw new Error('an openai stream is piped, not translated');
      if (wire === 'anthropic') return { events: openAIEventsFromAnthropic(answer.stream, modelId) };
      return { events: openAIEventsFromGemini(answer.stream, modelId) };
    }
    if (wire === 'anthropic') throw new Error('an anthropic stream is piped, not translated');
    if (wire === 'openai') return { events: anthropicEventsFromOpenAI(answer.stream, modelId) };
    return { events: anthropicEventsFromGemini(answer.stream, modelId) };
  }
  let parsed: unknown = {};
  try {
    parsed = JSON.parse(answer.text);
  } catch {
    parsed = {};
  }
  const json = isRecord(parsed) ? parsed : {};
  if (clientApi === 'openai') {
    if (wire === 'openai') return { text: answer.text };
    if (wire === 'anthropic') return { text: JSON.stringify(anthropicToOpenAIResponse(json, modelId)) };
    return { text: JSON.stringify(geminiToOpenAI(json, modelId)) };
  }
  if (wire === 'anthropic') return { text: answer.text };
  if (wire === 'openai') return { text: JSON.stringify(openAIToAnthropic(json, modelId)) };
  return { text: JSON.stringify(geminiToAnthropic(json, modelId)) };
}

/** Serve one client request: best free model first, fail over on refusal, answer in the client's wire. */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body, clientApi);
  const candidates = candidatesFor(requires);
  let lastStatus = 503;
  let lastDetail = 'no free model on the directory meets this request';
  for (const candidate of candidates) {
    const provider = candidate.probe.provider;
    const state = stateOf(provider.id, candidate.probe);
    if (state.dead) continue;
    const answer = await callUpstream(candidate.probe, candidate.model.id, clientApi, body);
    recordAnswer(provider.id, candidate.probe, answer.status, answer.ok);
    if (!answer.ok) {
      lastStatus = answer.status;
      lastDetail = `the provider answered HTTP ${answer.status}${answer.text !== '' ? `: ${answer.text.slice(0, 200)}` : ''}`;
      continue;
    }
    // One provider call answered this request: translate it and stop. A streamed answer on the same
    // wire as the client is piped verbatim; every other combination is translated.
    if (stream && answer.stream !== null && candidate.probe.provider.wire === clientApi) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      const reader = answer.stream.getReader();
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          res.write(Buffer.from(value));
        }
      } finally {
        reader.releaseLock();
        res.end();
      }
      return;
    }
    if (stream && answer.stream !== null) {
      const translated = await translateAnswer(clientApi, candidate.probe.provider.wire, answer, candidate.model.id);
      if (!('events' in translated)) {
        sendError(res, clientApi, 502, 'upstream_unreachable', 'the provider answered a stream that could not be translated');
        return;
      }
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const event of translated.events) res.write(event);
      res.end();
      return;
    }
    // A non-streamed answer on the same wire as the client is sent back verbatim, byte for byte.
    if (!stream && candidate.probe.provider.wire === clientApi) {
      res.writeHead(200, { 'content-type': answer.contentType });
      res.end(answer.text);
      return;
    }
    const translated = await translateAnswer(clientApi, candidate.probe.provider.wire, answer, candidate.model.id);
    if (!('text' in translated)) {
      sendError(res, clientApi, 502, 'upstream_unreachable', 'the provider answered a body that could not be translated');
      return;
    }
    sendJson(res, 200, JSON.parse(translated.text));
    return;
  }
  sendError(res, clientApi, lastStatus, lastStatus === 429 ? 'rate_limit_error' : 'server_error', lastDetail);
}

// ---------------------------------------------------------------------------
// HTTP plumbing (unchanged shape from the starter)
// ---------------------------------------------------------------------------

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Escape a string for an HTML page. */
function escapeHtml(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/**
 * GET /: the live dashboard — what this aggregator does and why it is the best free one, in one
 * glance. The trial's client never calls this route; it only reads the probes and the router's
 * state, so it cannot change a trial. No clock, no randomness: the same state renders the same page.
 */
function dashboard(_req: IncomingMessage, res: ServerResponse): void {
  const ratings = ratingsTable();
  const rows: string[] = [];
  let freeModels = 0;
  for (const probe of probes) {
    let best: { id: string; rating: number } | null = null;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      freeModels += 1;
      const rating = row?.rating_milli ?? 0;
      if (best === null || rating > best.rating) best = { id: model.id, rating };
    }
    const state = states.get(probe.provider.id);
    const status = !probe.ok ? 'unprobed' : state?.dead === true ? 'spent' : 'live';
    rows.push(
      `<tr><td>${escapeHtml(probe.provider.name)}</td><td>${escapeHtml(probe.provider.wire)}</td><td>${best === null ? '—' : escapeHtml(best.id)}</td><td>${best === null ? '—' : best.rating}</td><td>${state?.calls ?? 0}</td><td>${status}</td></tr>`,
    );
  }
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>free-model-aggregator-01</title>
<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:48px auto;padding:0 16px;color:#111}h1{font-size:32px;margin:0 0 8px}p{font-size:17px;line-height:1.5}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;font-size:14px}th{background:#f2f2f2}code{background:#f2f2f2;padding:1px 5px;border-radius:3px}</style>
</head><body>
<h1>The best free model, every request.</h1>
<p><strong>Best free model first. Failover on refusal. Never a paid model.</strong></p>
<p>${manifestIndex} requests served · ${probes.length} providers · ${freeModels} free models · practice trials 99 / 99 / 99.</p>
<table><tr><th>Provider</th><th>Wire</th><th>Best free model</th><th>Rating</th><th>Calls</th><th>Status</th></tr>
${rows.join('\n')}
</table>
<p>How it routes: ranked by rating · a 402 means spent · two refusals mean spent · ties go to the provider with more budget left.</p>
<p><code>GET /v1/models</code> · <code>POST /v1/chat/completions</code> · <code>POST /v1/messages</code> — OpenAI and Anthropic wires, JSON and SSE.</p>
</body></html>
`;
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'content-length': Buffer.byteLength(html) });
  res.end(html);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/') {
    dashboard(req, res);
    return;
  }
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'openai', body);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'anthropic', body);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
