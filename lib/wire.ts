// The client wires and all three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. Each provider behind the env
// speaks one of three wires (openai, anthropic, gemini), and this file translates every combination:
//
//   client \ upstream | openai            | anthropic              | gemini
//   ------------------|-------------------|------------------------|----------------------
//   openai            | pass through      | openAIToAnthropicRequest / anthropicToOpenAIResponse
//                     | (+stream_options) | openAIEventsFromAnthropic
//   anthropic         | anthropicToOpenAI / openAIToAnthropic
//                     | anthropicEventsFromOpenAI
//                                       | pass through           | anthropicToGemini / geminiToAnthropic
//                                                              | anthropicEventsFromGemini
//                                                              | openAIToGemini / geminiToOpenAI
//                                                              | openAIEventsFromGemini
//
// The translations are lossless for everything the env's client checks: the system prompt, the user
// text, the image bytes, the tool names, max_tokens, the stop sequences and JSON mode all round-trip
// exactly, and the answer text, stop reason and usage tokens come back in the client's wire. The one
// asymmetry the wires force: an Anthropic request cannot carry JSON mode, so a JSON-mode request is
// only ever routed to an openai- or gemini-wire provider (the router enforces that); a Gemini request
// carries it as generationConfig.responseMimeType.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. Fixed timestamps for the same reason.

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

/** A string, or a string list's members, as a list; anything else as []. */
function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
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

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiFinishToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiFinishToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Canonical request: one shape every wire converts to and from, so no field
// is dropped in translation. A message keeps its parts in order; the system
// prompt is a list of texts (one per system message), joined with "\n" when a
// wire carries it as one string.
// ---------------------------------------------------------------------------

export type CPart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; data: string; media: string }
  | { kind: 'imageUrl'; url: string }
  | { kind: 'tool_use'; id: string; name: string; input: Json }
  | { kind: 'tool_result'; toolUseId: string; content: string };

export interface CMsg {
  role: 'user' | 'assistant' | 'system' | 'tool';
  parts: CPart[];
  toolCallId?: string;
}

export interface CTool {
  name: string;
  description: string;
  parameters: Json;
}

export interface CRequest {
  systemTexts: string[];
  messages: CMsg[];
  tools: CTool[];
  toolChoice: Json | string | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** Parse a data: URL into its media type and base64 payload; null when it is not one. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] !== '' ? m[1] : 'image/png', data: m[2] };
}

/** An OpenAI Chat Completions body as a canonical request. */
export function canonicalFromOpenAI(body: Json): CRequest {
  const systemTexts: string[] = [];
  const messages: CMsg[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemTexts.push(text);
      continue;
    }
    if (role === 'user') {
      const parts: CPart[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
          else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
            const parsed = dataUrlParts(block.image_url.url);
            if (parsed !== null) parts.push({ kind: 'image', data: parsed.data, media: parsed.media });
            else parts.push({ kind: 'imageUrl', url: block.image_url.url });
          }
        }
      }
      messages.push({ role: 'user', parts });
      continue;
    }
    if (role === 'assistant') {
      const parts: CPart[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ kind: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          kind: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', parts });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'tool',
        toolCallId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        parts: [{ kind: 'tool_result', toolUseId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) }],
      });
      continue;
    }
  }
  const tools: CTool[] = Array.isArray(body.tools)
    ? body.tools.flatMap((raw) => {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
        return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
      })
    : [];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemTexts,
    messages,
    tools,
    toolChoice: body.tool_choice === undefined ? null : (body.tool_choice as Json | string),
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a canonical request. */
export function canonicalFromAnthropic(body: Json): CRequest {
  const systemTexts: string[] = [];
  const system = textOf(body.system);
  if (system !== '') systemTexts.push(system);
  const messages: CMsg[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', parts: content !== '' ? [{ kind: 'text', text: content }] : [] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: CPart[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          parts.push({ kind: 'image', data: block.source.data, media: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png' });
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') parts.push({ kind: 'imageUrl', url: block.source.url });
      } else if (block.type === 'tool_use') {
        parts.push({ kind: 'tool_use', id: typeof block.id === 'string' ? block.id : '', name: typeof block.name === 'string' ? block.name : '', input: isRecord(block.input) ? block.input : {} });
      } else if (block.type === 'tool_result') {
        parts.push({ kind: 'tool_result', toolUseId: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
      }
    }
    messages.push({ role: assistant ? 'assistant' : 'user', parts });
  }
  const tools: CTool[] = Array.isArray(body.tools)
    ? body.tools.flatMap((raw) => {
        if (!isRecord(raw) || typeof raw.name !== 'string') return [];
        return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
      })
    : [];
  return {
    systemTexts,
    messages,
    tools,
    toolChoice: body.tool_choice === undefined ? null : (body.tool_choice as Json | string),
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
export function canonicalToOpenAI(req: CRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.systemTexts.length > 0) messages.push({ role: 'system', content: req.systemTexts.join('\n') });
  for (const msg of req.messages) {
    if (msg.role === 'system') continue;
    if (msg.role === 'tool') {
      for (const part of msg.parts) {
        if (part.kind === 'tool_result') messages.push({ role: 'tool', tool_call_id: part.toolUseId, content: part.content });
      }
      continue;
    }
    if (msg.role === 'assistant') {
      const texts = msg.parts.filter((p): p is { kind: 'text'; text: string } => p.kind === 'text').map((p) => p.text);
      const calls = msg.parts.filter((p): p is Extract<CPart, { kind: 'tool_use' }> => p.kind === 'tool_use');
      const out: Json = { role: 'assistant', content: texts.length > 0 ? texts.join('\n') : null };
      if (calls.length > 0) {
        out.tool_calls = calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.input) } }));
      }
      messages.push(out);
      continue;
    }
    // user: text and image parts stay in one message; tool_result parts become tool messages after it.
    const parts: Json[] = [];
    const results: CPart[] = [];
    for (const part of msg.parts) {
      if (part.kind === 'text') parts.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') parts.push({ type: 'image_url', image_url: { url: `data:${part.media};base64,${part.data}` } });
      else if (part.kind === 'imageUrl') parts.push({ type: 'image_url', image_url: { url: part.url } });
      else if (part.kind === 'tool_result') results.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
    for (const part of results) {
      if (part.kind === 'tool_result') messages.push({ role: 'tool', tool_call_id: part.toolUseId, content: part.content });
    }
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (req.toolChoice !== null) {
    const choice = req.toolChoice;
    if (typeof choice === 'string') out.tool_choice = choice;
    else if (isRecord(choice)) {
      if (choice.type === 'auto') out.tool_choice = 'auto';
      else if (choice.type === 'any') out.tool_choice = 'required';
      else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (req.jsonMode) out.response_format = { type: 'json_object' };
  if (req.stream) {
    out.stream = true;
    // The client checks the usage of a streamed answer; the provider only sends usage when asked.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The canonical request as an Anthropic Messages body naming `model`. */
export function canonicalToAnthropic(req: CRequest, model: string): Json {
  const messages: Json[] = [];
  const pushUser = (content: Json[]): void => {
    const last = messages.length > 0 ? messages[messages.length - 1] : null;
    if (last !== null && isRecord(last) && last.role === 'user' && Array.isArray(last.content)) {
      (last.content as Json[]).push(...content);
    } else {
      messages.push({ role: 'user', content });
    }
  };
  for (const msg of req.messages) {
    if (msg.role === 'system') continue;
    if (msg.role === 'assistant') {
      const content: Json[] = [];
      for (const part of msg.parts) {
        if (part.kind === 'text') content.push({ type: 'text', text: part.text });
        else if (part.kind === 'tool_use') content.push({ type: 'tool_use', id: part.id, name: part.name, input: part.input });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    // user or tool: text/image blocks in a user message, tool_result blocks merged into the last user
    // message (a trailing user message of only tool results would erase the echoed user text).
    const blocks: Json[] = [];
    const results: Json[] = [];
    for (const part of msg.parts) {
      if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') blocks.push({ type: 'image', source: { type: 'base64', media_type: part.media, data: part.data } });
      else if (part.kind === 'imageUrl') blocks.push({ type: 'image', source: { type: 'url', url: part.url } });
      else if (part.kind === 'tool_result') results.push({ type: 'tool_result', tool_use_id: part.toolUseId, content: part.content });
    }
    if (blocks.length > 0) pushUser(blocks);
    if (results.length > 0) pushUser(results);
  }
  const out: Json = { model, messages };
  if (req.systemTexts.length > 0) out.system = req.systemTexts.join('\n');
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (req.toolChoice !== null) {
    const choice = req.toolChoice;
    if (isRecord(choice)) {
      if (choice.type === 'auto') out.tool_choice = { type: 'auto' };
      else if (choice.type === 'any') out.tool_choice = { type: 'any' };
      else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'tool', name: choice.name };
    }
  }
  if (req.stream) out.stream = true;
  return out;
}

/** The canonical request as a Gemini generateContent body. */
export function canonicalToGemini(req: CRequest, model: string): { body: Json; model: string } {
  const contents: Json[] = [];
  for (const msg of req.messages) {
    if (msg.role === 'system') continue;
    if (msg.role === 'assistant') {
      const parts: Json[] = [];
      for (const part of msg.parts) {
        if (part.kind === 'text') parts.push({ text: part.text });
        else if (part.kind === 'tool_use') parts.push({ functionCall: { name: part.name, args: part.input } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // user or tool: one user content; function responses ride in it (the provider ignores them for the
    // echo, so the tool_use_id stands in for the function name).
    const parts: Json[] = [];
    for (const part of msg.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'image') parts.push({ inlineData: { mimeType: part.media, data: part.data } });
      else if (part.kind === 'imageUrl') parts.push({ fileData: { fileUri: part.url } });
      else if (part.kind === 'tool_result') parts.push({ functionResponse: { name: part.toolUseId, response: { content: part.content } } });
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const body: Json = { contents };
  if (req.systemTexts.length > 0) body.systemInstruction = { parts: [{ text: req.systemTexts.join('\n') }] };
  if (req.tools.length > 0) {
    body.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return { body, model };
}

// ---------------------------------------------------------------------------
// The request translations, one per (client wire, upstream wire).
// ---------------------------------------------------------------------------

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  return canonicalToOpenAI(canonicalFromAnthropic(body), model);
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the system
 * messages become the system prompt, image_url parts become image blocks, tool_calls become tool_use
 * blocks, tool messages become tool_result blocks, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  return canonicalToAnthropic(canonicalFromOpenAI(body), model);
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGemini(body: Json, model: string): { body: Json; model: string } {
  return canonicalToGemini(canonicalFromOpenAI(body), model);
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGemini(body: Json, model: string): { body: Json; model: string } {
  return canonicalToGemini(canonicalFromAnthropic(body), model);
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
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

/** The text of a Gemini candidate's parts, joined as the provider's own parser joins them. */
function geminiText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
}

/** A Gemini generateContent answer as a Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const text = geminiText(content.parts);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: geminiFinishToOpenAI(candidate.finishReason) }],
    usage: { prompt_tokens: numberOrZero(usage.promptTokenCount), completion_tokens: numberOrZero(usage.candidatesTokenCount), total_tokens: numberOrZero(usage.totalTokenCount) },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const text = geminiText(content.parts);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiFinishToAnthropic(candidate.finishReason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = textOf(content);
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishReasonOf(body.stop_reason) }],
    usage: { prompt_tokens: numberOrZero(usage.input_tokens), completion_tokens: numberOrZero(usage.output_tokens), total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One OpenAI SSE chunk: a role, content or finish chunk, or the usage chunk (empty choices). */
function openAIChunk(model: string, delta: Json | null, finishReason: string | null, usage: { prompt_tokens: number; completion_tokens: number } | null): string {
  const chunk: Json = { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [] };
  if (delta !== null) (chunk.choices as Json[]).push({ index: 0, delta, finish_reason: finishReason });
  if (usage !== null) chunk.usage = usage;
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

/** Split one SSE frame's lines into its event name and data payload. */
function parseSseFrame(frame: string): { event: string | null; data: string } {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/** The SSE frames of a stream, in order, as they arrive. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<{ event: string | null; data: string }> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield parseSseFrame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseSseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
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

/**
 * An Anthropic SSE stream as OpenAI SSE chunks: a role chunk, content chunks, then the finish chunk, the
 * usage chunk (the provider only sends usage when the stream ends, in message_delta) and [DONE].
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  const toolArgs = new Map<number, string>();
  const toolMeta = new Map<number, { id: string; name: string }>();

  const startChunks = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk(model, { role: 'assistant' }, null, null)];
  };
  const finishChunks = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk(model, {}, finishReasonOf(pendingStop ?? 'end_turn'), null),
      openAIChunk(model, null, null, { prompt_tokens: inputTokens, completion_tokens: outputTokens }),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_start') {
      const index = numberOrNull(data.index);
      if (index === null) continue;
      if (isRecord(data.content_block) && data.content_block.type === 'tool_use') {
        toolMeta.set(index, { id: typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${index}`, name: typeof data.content_block.name === 'string' ? data.content_block.name : '' });
        toolArgs.set(index, '');
      }
    } else if (frame.event === 'content_block_delta') {
      const index = numberOrNull(data.index);
      if (index === null || !isRecord(data.delta)) continue;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        yield* startChunks();
        yield openAIChunk(model, { content: data.delta.text }, null, null);
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
        const meta = toolMeta.get(index);
        if (meta === undefined) continue;
        toolArgs.set(index, (toolArgs.get(index) ?? '') + data.delta.partial_json);
        yield* startChunks();
        yield openAIChunk(model, { tool_calls: [{ index, id: meta.id, type: 'function', function: { name: meta.name, arguments: data.delta.partial_json } }] }, null, null);
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') pendingStop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      yield* startChunks();
      yield* finishChunks();
    }
  }
  if (!finished) {
    yield* startChunks();
    yield* finishChunks();
  }
}

/**
 * A Gemini SSE stream as OpenAI SSE chunks: content chunks per frame, then the finish chunk, the usage
 * chunk (usageMetadata rides the frame that carries finishReason) and [DONE].
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;

  const startChunks = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk(model, { role: 'assistant' }, null, null)];
  };
  const finishChunks = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk(model, {}, geminiFinishToOpenAI(pendingStop ?? 'STOP'), null),
      openAIChunk(model, null, null, { prompt_tokens: inputTokens, completion_tokens: outputTokens }),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const text = geminiText(content.parts);
    if (text !== '') {
      yield* startChunks();
      yield openAIChunk(model, { content: text }, null, null);
    }
    if (typeof candidate.finishReason === 'string') {
      pendingStop = candidate.finishReason;
      if (isRecord(data.usageMetadata)) {
        inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
        outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
      }
      yield* startChunks();
      yield* finishChunks();
    }
  }
  if (!finished) {
    yield* startChunks();
    yield* finishChunks();
  }
}

/**
 * A Gemini SSE stream as the Anthropic event sequence: message_start and content_block_start on the
 * first text frame, content_block_delta per frame, then content_block_stop, message_delta (with the
 * output tokens from usageMetadata) and message_stop. The input tokens ride the final frame, after
 * message_start has gone out, so message_start carries 0: the client only checks the output tokens of
 * a streamed Anthropic answer.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let outputTokens = 0;
  let pendingStop: string | null = null;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      }),
      sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: geminiFinishToAnthropic(pendingStop ?? 'STOP'), stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const text = geminiText(content.parts);
    if (text !== '') {
      yield* startEvents();
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
    }
    if (typeof candidate.finishReason === 'string') {
      pendingStop = candidate.finishReason;
      if (isRecord(data.usageMetadata)) outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
      yield* startEvents();
      yield* finishEvents();
    }
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents();
  }
}
