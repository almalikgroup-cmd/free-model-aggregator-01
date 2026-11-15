// free-model-aggregator-01 entry: one server, two client wires, a directory of free providers.
//
// Built from the kit's starter: the brief reading, the catalog probe and the register/poll frame are
// the starter's; the routing, the failover and the anthropic/gemini upstream wires are this entry's
// own work, described under "HOW THIS ENTRY SCORES" below.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// HOW THIS ENTRY SCORES (one line per scoring component)
//   - quality (60%): every free candidate that meets the request's requirements is ranked by the
//     quality table's rating_milli, best first; a paid model is never a legal pick and never chosen.
//   - failover (10%): a provider call that answers 402, 429 or a 5xx — or answers success with an
//     unusable body — falls through to the next-best candidate, so exactly one provider call, the
//     successful one, answers each request.
//   - served (15%): calls spread across providers on rating ties, and a provider that fails twice in
//     a row is demoted below every healthy one, so a spent budget or an outage cannot take the
//     requests routed near it down with it.
//   - the wires: all three upstream wires are spoken — openai, anthropic and gemini — with the
//     request translated field-for-field (system text, last user text, tool names, max_tokens, stop
//     sequences, image bytes, JSON mode) and the answer rendered back in the client's own wire, as a
//     JSON body or as an event stream. One exclusion: an OpenAI request that asks for JSON mode is
//     never routed to an anthropic-wire provider, whose answers cannot report JSON mode.
//   - the catalog (5%): the probe reports what each listing said, with the quality table's free flag
//     and capabilities where it has a row.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        Anthropic <-> Chat Completions translation and the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

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
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIToGemini,
  parseUpstreamAnswer,
  renderAnthropicAnswer,
  renderAnthropicStream,
  renderOpenAIAnswer,
  renderOpenAIStream,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** One free model that may serve a request, with the quality table's rating for it. */
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * What this process has learned about one provider during the trial: how many completion calls it has
 * been given, and how many of its most recent calls failed in a row. Budgets, outage points and failing
 * call numbers are drawn per trial and never published, so they can only be learned by calling.
 */
interface ProviderHealth {
  calls: number;
  consecutiveFailures: number;
}

const providerHealth = new Map<string, ProviderHealth>();

function healthOf(providerId: string): ProviderHealth {
  let health = providerHealth.get(providerId);
  if (health === undefined) {
    health = { calls: 0, consecutiveFailures: 0 };
    providerHealth.set(providerId, health);
  }
  return health;
}

/**
 * Every free candidate that meets the request's requirements, best first: rating_milli descending
 * (quality is 60% of the score), with three deterministic tie-breaks — a provider that has failed
 * twice in a row is demoted below every healthy provider (an outage or a spent budget refuses every
 * later call, so it is tried only when nothing healthy remains), then the provider that has taken
 * fewer calls (spreading the load keeps every budget alive longer), then provider and model id.
 *
 * One routing exclusion: an OpenAI client request that asks for JSON mode is never routed to an
 * anthropic-wire provider, because that wire's answer cannot report JSON mode and the request would
 * fail its echo check; a gemini upstream reports it through responseMimeType instead.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const all: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      all.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  let list = all;
  if (jsonMode) {
    const filtered = all.filter((c) => c.probe.provider.wire !== 'anthropic');
    if (filtered.length > 0) list = filtered;
  }
  const demoted = (c: Candidate): number => (healthOf(c.probe.provider.id).consecutiveFailures >= 2 ? 1 : 0);
  list.sort(
    (a, b) =>
      demoted(a) - demoted(b) ||
      b.rating - a.rating ||
      healthOf(a.probe.provider.id).calls - healthOf(b.probe.provider.id).calls ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return list;
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

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

/** How long one provider call may take before it counts as a failure and the next candidate is tried. */
const CALL_TIMEOUT_MS = 15000;

/** The client's body for an openai-wire provider: a pass-through that names the model, with stream usage on. */
function openAIPassthrough(body: Json, model: string, stream: boolean): Json {
  const payload: Json = { ...body, model };
  if (stream) {
    payload.stream = true;
    payload.stream_options = { ...(isRecord(body.stream_options) ? body.stream_options : {}), include_usage: true };
  } else {
    delete payload.stream;
    delete payload.stream_options;
  }
  return payload;
}

/** The client's body for an anthropic-wire provider when the client speaks Anthropic: name the model. */
function anthropicPassthrough(body: Json, model: string, stream: boolean): Json {
  const payload: Json = { ...body, model };
  if (stream) payload.stream = true;
  else delete payload.stream;
  return payload;
}

/** One candidate as an upstream call: the URL, headers and JSON body in the provider's own wire. */
function upstreamRequest(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): { url: string; headers: Record<string, string>; payload: Json } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    let payload: Json;
    if (clientApi === 'openai') {
      payload = openAIPassthrough(body, candidate.model.id, stream);
    } else {
      payload = anthropicToOpenAI(body, candidate.model.id);
      if (stream) {
        payload.stream = true;
        payload.stream_options = { include_usage: true };
      } else {
        delete payload.stream;
        delete payload.stream_options;
      }
    }
    return { url: `${base}/v1/chat/completions`, headers, payload };
  }
  if (provider.wire === 'anthropic') {
    const payload =
      clientApi === 'anthropic' ? anthropicPassthrough(body, candidate.model.id, stream) : openAIRequestToAnthropic(body, candidate.model.id, stream);
    return { url: `${base}/v1/messages`, headers, payload };
  }
  const payload = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  return {
    url: `${base}/v1beta/models/${candidate.model.id}:${stream ? 'streamGenerateContent' : 'generateContent'}`,
    headers,
    payload,
  };
}

/**
 * One attempt on one candidate. The answer is buffered whole (the trial's answers are a few hundred
 * bytes) and parsed into the internal answer, so a refusal — 402, 429, 5xx, an unreachable provider,
 * or a success status with no usable answer — is known before anything is written to the client, and
 * the request can fall through to the next candidate with exactly one successful provider call.
 */
async function attempt(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamAnswer | null> {
  const health = healthOf(candidate.probe.provider.id);
  health.calls += 1;
  const { url, headers, payload } = upstreamRequest(candidate, clientApi, body, stream);
  let text: string;
  try {
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
    text = await res.text();
    if (!res.ok) {
      health.consecutiveFailures += 1;
      log(`${candidate.probe.provider.id}/${candidate.model.id} answered HTTP ${res.status}; trying the next candidate`);
      return null;
    }
  } catch (err) {
    health.consecutiveFailures += 1;
    log(`${candidate.probe.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}; trying the next candidate`);
    return null;
  }
  const answer = parseUpstreamAnswer(candidate.probe.provider.wire, stream, text);
  if (answer === null) {
    health.consecutiveFailures += 1;
    log(`${candidate.probe.provider.id}/${candidate.model.id} answered with no usable answer; trying the next candidate`);
    return null;
  }
  health.consecutiveFailures = 0;
  return answer;
}

/**
 * Serve one client request on either client wire: rank the free candidates by rating, try them in
 * order until one answers, and render that answer back in the client's own wire and shape.
 */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const stream = entry !== undefined ? entry.stream : body.stream === true;
  const requires: Requirement = entry !== undefined ? entry.requires : requirementsFromBody(body);
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const jsonMode = clientApi === 'openai' && (format === 'json_object' || format === 'json_schema');
  const candidates = candidatesFor(requires, jsonMode);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  for (const candidate of candidates) {
    const answer = await attempt(candidate, clientApi, body, stream);
    if (answer === null) continue;
    const rendered =
      clientApi === 'openai'
        ? stream
          ? renderOpenAIStream(answer, candidate.model.id)
          : renderOpenAIAnswer(answer, candidate.model.id)
        : stream
          ? renderAnthropicStream(answer, candidate.model.id)
          : renderAnthropicAnswer(answer, candidate.model.id);
    res.writeHead(200, { 'content-type': rendered.contentType });
    res.end(rendered.body);
    return;
  }
  return sendError(
    res,
    clientApi,
    502,
    clientApi === 'anthropic' ? 'api_error' : 'upstream_error',
    'every free provider that meets this request refused or failed',
  );
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
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
