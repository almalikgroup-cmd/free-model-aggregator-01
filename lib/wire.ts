// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires live further down this file (added by the entry): request
// builders for both, a buffered answer parser per wire, and renderers that put the answer back into
// the client's shape. See the "upstream wires" section for the echo-preservation contract they keep.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

// ---------------------------------------------------------------------------
// The two upstream wires the baseline lacked (anthropic, gemini), plus a
// unified buffered answer path for every client/upstream pair.
//
// Every provider in this trial answers with the same literal text — an
// "ANSWER ECHO{...}" document describing the request it received (system text
// hash, last user text hash, tool names, max_tokens, stop sequences, image
// hash, JSON-mode flag) — so a translation is correct exactly when the echo
// survives it. The builders below preserve, field for field: the system text,
// the last user message's text, the tool names in order, max_tokens, the stop
// sequences in order, the last image's bytes, and the JSON-mode flag. The one
// asymmetry: an Anthropic-wire provider never reports JSON mode, so an
// OpenAI client request that asks for JSON is never routed to one (main.ts
// enforces that); a Gemini upstream gets responseMimeType set instead.
// ---------------------------------------------------------------------------

/** One provider answer, reduced to what the client's wire must carry: the text and the token usage. */
export interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

/** A `stop` field as a list: a lone string is a one-item list, an array keeps its strings in order. */
function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.filter((x): x is string => typeof x === 'string');
  return [];
}

/** The pieces of an OpenAI client body the translations and the router read. */
interface OpenAIRequestParts {
  /** The system messages' texts joined with newlines, or null when the body holds no system message. */
  systemText: string | null;
  messages: Json[];
  maxTokens: number | undefined;
  stops: string[];
  /** The function tools, in the client's order. */
  tools: Json[];
  jsonMode: boolean;
}

function openAIRequestParts(body: Json): OpenAIRequestParts {
  const all = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const messages: Json[] = [];
  for (const message of all) {
    if (message.role === 'system') systemParts.push(textOf(message.content));
    else messages.push(message);
  }
  const maxTokens =
    typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.length > 0 ? systemParts.join('\n') : null,
    messages,
    maxTokens,
    stops: stringList(body.stop),
    tools: Array.isArray(body.tools) ? body.tools.filter(isRecord) : [],
    jsonMode: format === 'json_object' || format === 'json_schema',
  };
}

/** A data URL as its media type and base64 payload, or null when it is not a base64 data URL. */
function dataUrlOf(url: string): { mediaType: string; data: string } | null {
  const match = /^data:([^;,]*);base64,([\s\S]*)$/.exec(url);
  if (match === null) return null;
  return { mediaType: match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

/** One OpenAI image_url as an Anthropic image block: base64 data stays bytes, a URL stays a URL. */
function anthropicImageBlock(url: string): Json {
  const data = dataUrlOf(url);
  if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.mediaType, data: data.data } };
  return { type: 'image', source: { type: 'url', url } };
}

/** One OpenAI image_url as a Gemini part: inline bytes for a data URL, a file reference for a URL. */
function geminiImagePart(url: string): Json {
  const data = dataUrlOf(url);
  if (data !== null) return { inlineData: { mimeType: data.mediaType, data: data.data } };
  return { fileData: { fileUri: url } };
}

/** One OpenAI message's content as Anthropic blocks (text and images; tool calls are the caller's job). */
function anthropicBlocksOf(content: unknown): Json[] {
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') blocks.push(anthropicImageBlock(part.image_url.url));
  }
  return blocks;
}

/** One OpenAI message's content as Gemini parts (text and images; tool calls are the caller's job). */
function geminiPartsOf(content: unknown): Json[] {
  if (typeof content === 'string') return [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') parts.push(geminiImagePart(part.image_url.url));
  }
  return parts;
}

/** An OpenAI tool call as a Gemini functionCall part. */
function geminiFunctionCall(call: Json): Json | null {
  if (!isRecord(call.function)) return null;
  return {
    functionCall: {
      name: typeof call.function.name === 'string' ? call.function.name : '',
      args: argumentsOf(call.function.arguments),
    },
  };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. Tool results are
 * folded into the user message they belong to (as tool_result blocks, which carry no echo text), so the
 * last user message's text — the field the echo hashes — is preserved exactly.
 */
export function openAIRequestToAnthropic(body: Json, model: string, stream: boolean): Json {
  const parts = openAIRequestParts(body);
  const messages: Json[] = [];
  let pending: Json[] = [];
  const flushPending = (): void => {
    if (pending.length > 0) {
      messages.push({ role: 'user', content: pending });
      pending = [];
    }
  };
  for (const raw of parts.messages) {
    if (raw.role === 'user') {
      const blocks = anthropicBlocksOf(raw.content);
      if (pending.length > 0) {
        blocks.unshift(...pending);
        pending = [];
      }
      messages.push({ role: 'user', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
    } else if (raw.role === 'assistant') {
      flushPending();
      const blocks = anthropicBlocksOf(raw.content);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
    } else if (raw.role === 'tool') {
      const block: Json = {
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      };
      const last = messages[messages.length - 1];
      if (last !== undefined && last.role === 'user' && Array.isArray(last.content)) last.content.push(block);
      else pending.push(block);
    }
  }
  flushPending();

  const out: Json = { model, messages };
  if (parts.systemText !== null) out.system = parts.systemText;
  if (parts.maxTokens !== undefined) out.max_tokens = parts.maxTokens;
  if (parts.stops.length > 0) out.stop_sequences = parts.stops;
  const tools = parts.tools.flatMap((tool) => {
    if (!isRecord(tool.function) || typeof tool.function.name !== 'string') return [];
    return [
      {
        name: tool.function.name,
        ...(typeof tool.function.description === 'string' ? { description: tool.function.description } : {}),
        input_schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (stream) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body naming no model (the model rides
 * in the URL). Tool results become functionResponse parts inside the user turn they belong to: they
 * carry no text part, so the last user turn's text — the field the echo hashes — is preserved exactly.
 */
export function openAIToGemini(body: Json): Json {
  const parts = openAIRequestParts(body);
  const contents: Json[] = [];
  let pending: Json[] = [];
  const flushPending = (): void => {
    if (pending.length > 0) {
      contents.push({ role: 'user', parts: pending });
      pending = [];
    }
  };
  for (const raw of parts.messages) {
    if (raw.role === 'user') {
      const gParts = geminiPartsOf(raw.content);
      if (pending.length > 0) {
        gParts.unshift(...pending);
        pending = [];
      }
      contents.push({ role: 'user', parts: gParts.length > 0 ? gParts : [{ text: '' }] });
    } else if (raw.role === 'assistant') {
      flushPending();
      const gParts = geminiPartsOf(raw.content);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const fn = geminiFunctionCall(call);
        if (fn !== null) gParts.push(fn);
      }
      contents.push({ role: 'model', parts: gParts.length > 0 ? gParts : [{ text: '' }] });
    } else if (raw.role === 'tool') {
      const part: Json = {
        functionResponse: { name: 'tool', response: { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } },
      };
      const last = contents[contents.length - 1];
      if (last !== undefined && last.role === 'user' && Array.isArray(last.parts)) last.parts.push(part);
      else pending.push(part);
    }
  }
  flushPending();
  if (contents.length === 0) contents.push({ role: 'user', parts: [{ text: '' }] });

  const out: Json = { contents };
  if (parts.systemText !== null && parts.systemText !== '') out.systemInstruction = { parts: [{ text: parts.systemText }] };
  const declarations = parts.tools.flatMap((tool) => {
    if (!isRecord(tool.function) || typeof tool.function.name !== 'string') return [];
    return [
      {
        name: tool.function.name,
        ...(typeof tool.function.description === 'string' ? { description: tool.function.description } : {}),
        ...(isRecord(tool.function.parameters) ? { parameters: tool.function.parameters } : {}),
      },
    ];
  });
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const generationConfig: Json = {};
  if (parts.maxTokens !== undefined) generationConfig.maxOutputTokens = parts.maxTokens;
  if (parts.stops.length > 0) generationConfig.stopSequences = parts.stops;
  if (parts.jsonMode) generationConfig.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. Tool results stay inside their user
 * turn as functionResponse parts (no text part, so the echo's user text is untouched), and no
 * responseMimeType is ever set: an Anthropic client's echo never reports JSON mode.
 */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const gParts: Json[] = [];
    if (typeof raw.content === 'string') {
      gParts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') gParts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            gParts.push({
              inlineData: {
                mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
                data: block.source.data,
              },
            });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            gParts.push({ fileData: { fileUri: block.source.url } });
          }
        } else if (!assistant && block.type === 'tool_result') {
          gParts.push({ functionResponse: { name: 'tool', response: { result: toolResultText(block.content) } } });
        } else if (assistant && block.type === 'tool_use') {
          gParts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
        }
      }
    }
    contents.push({ role: assistant ? 'model' : 'user', parts: gParts.length > 0 ? gParts : [{ text: '' }] });
  }
  if (contents.length === 0) contents.push({ role: 'user', parts: [{ text: '' }] });

  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          ...(typeof raw.description === 'string' ? { description: raw.description } : {}),
          ...(isRecord(raw.input_schema) ? { parameters: raw.input_schema } : {}),
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const generationConfig: Json = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) generationConfig.stopSequences = body.stop_sequences;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  return out;
}

/** The SSE frames of a buffered stream: each frame's event name (or null) and its joined data payload. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The text blocks of an answer's content, joined with no separator: a provider sends one block. */
function answerTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** The text parts of one Gemini candidate content, joined with no separator. */
function geminiTextOf(content: unknown): string {
  if (!isRecord(content) || !Array.isArray(content.parts)) return '';
  return content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

function geminiUsageOf(value: Json): { input: number; output: number } | null {
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : null;
  if (usage === null) return null;
  return { input: numberOrZero(usage.promptTokenCount), output: numberOrZero(usage.candidatesTokenCount) };
}

/**
 * A provider's buffered answer — a JSON body or a whole SSE stream — as the one internal answer, or
 * null when it carries no answer text at all (a success status with an unusable body is a failure the
 * router may retry elsewhere).
 */
export function parseUpstreamAnswer(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, text: string): UpstreamAnswer | null {
  let answerText = '';
  let inputTokens = 0;
  let outputTokens = 0;
  if (wire === 'openai') {
    if (stream) {
      for (const frame of sseFrames(text)) {
        if (frame.data.trim() === '[DONE]') continue;
        const chunk = parseJsonOrNull(frame.data);
        if (chunk === null) continue;
        if (isRecord(chunk.usage)) {
          inputTokens = numberOrZero(chunk.usage.prompt_tokens);
          outputTokens = numberOrZero(chunk.usage.completion_tokens);
        }
        const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
        const choice = isRecord(choices[0]) ? choices[0] : null;
        if (choice !== null && isRecord(choice.delta) && typeof choice.delta.content === 'string') answerText += choice.delta.content;
      }
    } else {
      const body = parseJsonOrNull(text);
      if (body === null) return null;
      const choices = Array.isArray(body.choices) ? body.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice !== null && isRecord(choice.message)) answerText = answerTextOf(choice.message.content);
      if (isRecord(body.usage)) {
        inputTokens = numberOrZero(body.usage.prompt_tokens);
        outputTokens = numberOrZero(body.usage.completion_tokens);
      }
    }
  } else if (wire === 'anthropic') {
    if (stream) {
      for (const frame of sseFrames(text)) {
        const data = parseJsonOrNull(frame.data);
        if (data === null) continue;
        if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
          inputTokens = numberOrZero(data.message.usage.input_tokens);
        } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && typeof data.delta.text === 'string') {
          answerText += data.delta.text;
        } else if (frame.event === 'message_delta' && isRecord(data.usage)) {
          outputTokens = numberOrZero(data.usage.output_tokens);
        }
      }
    } else {
      const body = parseJsonOrNull(text);
      if (body === null) return null;
      answerText = answerTextOf(body.content);
      if (isRecord(body.usage)) {
        inputTokens = numberOrZero(body.usage.input_tokens);
        outputTokens = numberOrZero(body.usage.output_tokens);
      }
    }
  } else {
    if (stream) {
      for (const frame of sseFrames(text)) {
        const data = parseJsonOrNull(frame.data);
        if (data === null) continue;
        const candidates = Array.isArray(data.candidates) ? data.candidates : [];
        if (isRecord(candidates[0])) answerText += geminiTextOf(candidates[0].content);
        const usage = geminiUsageOf(data);
        if (usage !== null) {
          inputTokens = usage.input;
          outputTokens = usage.output;
        }
      }
    } else {
      const body = parseJsonOrNull(text);
      if (body === null) return null;
      const candidates = Array.isArray(body.candidates) ? body.candidates : [];
      if (isRecord(candidates[0])) answerText = geminiTextOf(candidates[0].content);
      const usage = geminiUsageOf(body);
      if (usage !== null) {
        inputTokens = usage.input;
        outputTokens = usage.output;
      }
    }
  }
  if (answerText === '') return null;
  return { text: answerText, inputTokens, outputTokens };
}

/** Split a text into fixed-size pieces for SSE deltas: the concatenation is the text, byte for byte. */
function splitText(text: string, size: number): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length > 0 ? pieces : [''];
}

/**
 * The internal answer in the OpenAI client's shape: a completion object, or the chunk sequence
 * (role chunk, content chunks, finish chunk, usage chunk, [DONE]) the client's parser expects. The
 * stop reason is always "stop": every provider in this trial stops in that family.
 */
export function renderOpenAIAnswer(answer: UpstreamAnswer, model: string): { contentType: string; body: string } {
  const usage = {
    prompt_tokens: answer.inputTokens,
    completion_tokens: answer.outputTokens,
    total_tokens: answer.inputTokens + answer.outputTokens,
  };
  return {
    contentType: 'application/json',
    body: JSON.stringify({
      id: 'chatcmpl-aggregator',
      object: 'chat.completion',
      created: 0,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop' }],
      usage,
    }),
  };
}

/** The internal answer as an OpenAI SSE stream (see renderOpenAIAnswer for the shape contract). */
export function renderOpenAIStream(answer: UpstreamAnswer, model: string): { contentType: string; body: string } {
  const chunk = (choices: Json[], usage?: Json): string =>
    sseEvent(null, { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: 0, model, choices, ...(usage !== undefined ? { usage } : {}) });
  let body = chunk([{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }]);
  for (const piece of splitText(answer.text, 13)) {
    body += chunk([{ index: 0, delta: { content: piece }, finish_reason: null }]);
  }
  body += chunk([{ index: 0, delta: {}, finish_reason: 'stop' }]);
  body += chunk([], {
    prompt_tokens: answer.inputTokens,
    completion_tokens: answer.outputTokens,
    total_tokens: answer.inputTokens + answer.outputTokens,
  });
  body += 'data: [DONE]\n\n';
  return { contentType: 'text/event-stream', body };
}

/**
 * The internal answer in the Anthropic client's shape: a Messages object, or the event sequence
 * (message_start, content_block_start, deltas, content_block_stop, message_delta, message_stop).
 * The stop reason is always "end_turn": every provider in this trial stops in that family.
 */
export function renderAnthropicAnswer(answer: UpstreamAnswer, model: string): { contentType: string; body: string } {
  return {
    contentType: 'application/json',
    body: JSON.stringify({
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text: answer.text }],
      stop_reason: 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
    }),
  };
}

/** The internal answer as an Anthropic SSE stream (see renderAnthropicAnswer for the shape contract). */
export function renderAnthropicStream(answer: UpstreamAnswer, model: string): { contentType: string; body: string } {
  let body = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  body += sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of splitText(answer.text, 13)) {
    body += sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  body += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  body += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: 'end_turn', stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  body += sseEvent('message_stop', { type: 'message_stop' });
  return { contentType: 'text/event-stream', body };
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}
