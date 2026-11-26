import { createHash, randomUUID } from 'node:crypto';
import { lstat, mkdir, open, readdir, rename, unlink } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import type { TruncationOptions } from '../types/config.js';
import type { CoreToolResult, TruncatedOutputDetails } from '../types/tool.js';

export type { TruncatedOutputDetails } from '../types/tool.js';

const WRITE_CHUNK_UNITS = 16 * 1024;
const CLEANUP_INTERVAL_MS = 60_000;
const DEFAULT_MAX_AGE_MS = 7 * 24 * 3600 * 1000;
const DEFAULT_MAX_TOTAL_BYTES = 1024 * 1024 * 1024;
const QUOTA_MIN_AGE_MS = 5_000;
const MANAGED_PREFIX = 'agent-core-output.';
const MANAGED_FILE = /^agent-core-output\.[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}\.txt$/;
const inFlightOutputs = new Set<string>();

type ToolOutput = Pick<CoreToolResult, 'content' | 'details'>;
type TextBlock = Extract<ToolOutput['content'][number], { type: 'text' }>;

interface TextPart {
  text: string;
  start: number;
}

interface Edge {
  offset: number;
  breaks: number;
  complete: boolean;
}

function safeFilePart(value: string): string {
  return /^[A-Za-z0-9_-]{1,128}$/.test(value)
    ? value : createHash('sha256').update(value).digest('hex');
}

function legacyFilename(sessionId: string, toolCallId: string): string {
  const session = safeFilePart(sessionId);
  const call = safeFilePart(toolCallId);
  return session.length + call.length <= 210 ? `${session}-${call}`
    : `${session}-${createHash('sha256').update(call).digest('hex')}`;
}

function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
}

function pairedAt(text: string, index: number): boolean {
  return isHighSurrogate(text.charCodeAt(index)) && isLowSurrogate(text.charCodeAt(index + 1));
}

function previewEdge(
  parts: TextPart[], length: number, maxLines: number, maxBytes: number, tail: boolean, preferNewline: boolean,
): Edge {
  let offset = tail ? length : 0;
  let breaks = 0;
  let bytes = 0;
  let boundary: Edge | undefined;
  if (maxLines <= 0 || maxBytes <= 0) return { offset, breaks, complete: false };
  const step = tail ? -1 : 1;
  for (let p = tail ? parts.length - 1 : 0; p >= 0 && p < parts.length; p += step) {
    const { text, start } = parts[p];
    const end = Math.min(text.length, length - start);
    for (let i = tail ? end - 1 : 0; i >= 0 && i < end;) {
      const code = text.charCodeAt(i);
      if (code === 10) {
        boundary = { offset, breaks, complete: true };
        if (breaks + 1 >= maxLines) return boundary;
      }
      const pair = pairedAt(text, tail ? i - 1 : i);
      bytes += pair ? 4 : code < 0x80 ? 1 : code < 0x800 ? 2 : 3;
      if (bytes > maxBytes) {
        if (boundary?.offset === offset || (preferNewline && boundary && boundary.offset !== (tail ? length : 0))) {
          return boundary;
        }
        return { offset, breaks, complete: false };
      }
      if (code === 10) breaks++;
      i += step * (pair ? 2 : 1);
      offset = start + i + Number(tail);
    }
  }
  return { offset, breaks, complete: true };
}

function splitBudget(limit: number): [number, number] {
  if (limit === Infinity) return [Infinity, Infinity];
  const head = Math.max(1, Math.floor(limit * 0.65));
  return [head, limit - head];
}

function* writeChunks(parts: TextPart[]): Iterable<string> {
  let pending = '';
  for (const { text } of parts) {
    for (let start = 0; start < text.length;) {
      let end = Math.min(start + WRITE_CHUNK_UNITS - pending.length, text.length);
      if (pairedAt(text, end - 1)) end--;
      pending += text.slice(start, end);
      start = end;
      if (pending.length >= WRITE_CHUNK_UNITS - 1) {
        yield pending;
        pending = '';
      }
    }
  }
  if (pending) yield pending;
}

function renderPreview(content: ToolOutput['content'], head: Edge, tail: Edge, length: number, note: string) {
  const preview: ToolOutput['content'] = [];
  let offset = -1; // Each text block after the first has an implicit newline.
  let lastHead: TextBlock | undefined;
  let noted = false;
  const addNote = () => {
    if (noted) return;
    noted = true;
    if (lastHead) lastHead.text += '\n\n' + note;
    else preview.push({ type: 'text', text: note });
  };
  for (const block of content) {
    if (block.type !== 'text') {
      preview.push(block);
      continue;
    }
    const start = offset + 1;
    offset = start + block.text.length;
    if (start < head.offset) {
      lastHead = { ...block, text: block.text.slice(0, head.offset - start) };
      preview.push(lastHead);
    }
    if (tail.offset < length && start < length && offset > tail.offset) {
      addNote();
      const suffix = block.text.slice(Math.max(0, tail.offset - start), length - start);
      if (lastHead && preview.at(-1) === lastHead && head.offset > start) lastHead.text += '\n\n' + suffix;
      else preview.push({ ...block, text: suffix });
    }
  }
  addNote();
  return preview;
}

export class OutputTruncator {
  private readonly directory: string;
  private readonly options: TruncationOptions;
  private lastCleanup = -Infinity;
  private cleanupTask?: Promise<void>;

  constructor(dataDir: string, options: TruncationOptions = {}) {
    this.directory = resolve(dataDir, 'tool-output');
    this.options = { ...options };
  }

  async truncate<T extends ToolOutput>(
    sessionId: string, toolCallId: string, result: T, toolName?: string,
  ): Promise<Omit<T, keyof ToolOutput> & ToolOutput> {
    const perTool = this.options.perTool;
    const override = toolName !== undefined && perTool && Object.hasOwn(perTool, toolName) ? perTool[toolName] : undefined;
    const maxLines = Math.max(0, Math.floor(override?.maxLines ?? this.options.maxLines ?? 2000));
    const maxBytes = Math.max(0, Math.floor(override?.maxBytes ?? this.options.maxBytes ?? 50 * 1024));
    const strategy = override?.strategy ?? this.options.strategy ?? 'head-tail';
    const parts: TextPart[] = [];
    let length = 0;
    let totalBytes = 0;
    let lineCount = 1;
    let endsWithNewline = false;
    for (const block of result.content) {
      if (block.type !== 'text') continue;
      if (parts.length) {
        parts.push({ text: '\n', start: length++ });
        totalBytes++;
        lineCount++;
      }
      parts.push({ text: block.text, start: length });
      length += block.text.length;
      totalBytes += Buffer.byteLength(block.text);
      endsWithNewline = block.text.endsWith('\n') || (parts.length > 1 && !block.text.length);
      for (let i = block.text.indexOf('\n'); i !== -1; i = block.text.indexOf('\n', i + 1)) lineCount++;
    }
    // The final newline terminates the last content line.
    const terminator = Number(endsWithNewline);
    const totalLines = lineCount - terminator;
    if (!length || (totalLines <= maxLines && totalBytes <= maxBytes)) return result;

    const previewLength = length - terminator;
    const split = strategy === 'head-tail' && (maxLines >= 2 || (maxLines === 1 && totalLines === 1)) && maxBytes >= 2;
    const [headLines, tailLines] = !split ? [maxLines, 0] : maxLines === 1 ? [1, 1] : splitBudget(maxLines);
    const [headBytes, tailBytes] = split ? splitBudget(maxBytes) : [maxBytes, 0];
    let head = previewEdge(parts, previewLength, headLines, headBytes, false, split);
    const tail = previewEdge(parts, previewLength, tailLines, tailBytes, true, split);
    if (split && head.offset === 0 && tail.offset === previewLength) {
      head = previewEdge(parts, previewLength, maxLines, maxBytes, false, false);
    }
    const firstOmitted = head.breaks + 1 + Number(head.complete && head.offset > 0);
    const lastOmitted = totalLines - tail.breaks - Number(tail.complete && tail.offset < previewLength);
    const omittedLines = Math.max(0, lastOmitted - firstOmitted + 1);
    const managed = this.options.cleanup?.enabled === true;
    const filename = managed ? MANAGED_PREFIX + randomUUID() : legacyFilename(sessionId, toolCallId);
    const targetPath = join(this.directory, filename + '.txt');
    if (managed) inFlightOutputs.add(targetPath);
    try {
      const outputPath = await this.save(targetPath, managed, parts);
      const range = firstOmitted === lastOmitted ? String(firstOmitted) : `${firstOmitted}–${lastOmitted}`;
      const omitted = omittedLines ? `${omittedLines} lines omitted or clipped (lines ${range}).` : 'Text clipped.';
      const note = `[Output truncated. Full output: ${outputPath}]\n` +
        `${totalLines} total lines, ${totalBytes} bytes. ${omitted}\n` +
        `Do not rerun the same command. Use a file read or search tool on ${outputPath} to read ranges or filter the full output.`;

      const metadata: TruncatedOutputDetails = {
        truncated: true, outputPath, totalLines, totalBytes, keptLines: totalLines - omittedLines, omittedLines, strategy,
      };
      const details = result.details;
      const existing = details && typeof details === 'object' && !Array.isArray(details) ? details
        : details === undefined ? {} : { originalDetails: details };
      return { ...result, content: renderPreview(result.content, head, tail, previewLength, note), details: { ...existing, ...metadata } };
    } finally {
      if (managed) inFlightOutputs.delete(targetPath);
    }
  }

  private async save(outputPath: string, managed: boolean, parts: TextPart[]): Promise<string> {
    await mkdir(this.directory, { recursive: true, mode: 0o700 });
    let target = managed ? outputPath + '.tmp' : outputPath;
    const file = await open(target, 'wx', 0o600).catch((error) => {
      if (managed || (error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
      outputPath = `${outputPath.slice(0, -4)}-${randomUUID()}.txt`;
      target = outputPath;
      return open(target, 'wx', 0o600);
    });
    let saved = false;
    try {
      try {
        for (const text of writeChunks(parts)) await file.writeFile(text);
      } finally {
        await file.close();
      }
      if (managed) await rename(target, outputPath);
      saved = true;
    } finally {
      if (!saved) await unlink(target).catch(() => undefined);
    }
    if (managed) this.scheduleCleanup(outputPath);
    return outputPath;
  }

  private scheduleCleanup(justSaved: string): void {
    const now = Date.now();
    if (this.cleanupTask || now - this.lastCleanup < CLEANUP_INTERVAL_MS) return;
    this.lastCleanup = now;
    this.cleanupTask = this.clean(now, justSaved).catch(() => undefined).finally(() => { this.cleanupTask = undefined; });
  }

  /** Wait for any scheduled maintenance before disposing the owning runtime. */
  async flushCleanup(): Promise<void> {
    await this.cleanupTask;
  }

  private async clean(now: number, justSaved: string): Promise<void> {
    const maxAge = this.options.cleanup?.maxAgeMs ?? DEFAULT_MAX_AGE_MS;
    const maxBytes = this.options.cleanup?.maxTotalBytes ?? DEFAULT_MAX_TOTAL_BYTES;
    const files: Array<{ path: string; size: number; mtimeMs: number }> = [];
    let totalBytes = 0;
    for (const name of await readdir(this.directory)) {
      if (!MANAGED_FILE.test(name)) continue;
      const path = join(this.directory, name);
      const stats = await lstat(path).catch(() => undefined);
      if (!stats?.isFile()) continue;
      files.push({ path, size: stats.size, mtimeMs: stats.mtimeMs });
      totalBytes += stats.size;
    }
    files.sort((a, b) => a.mtimeMs - b.mtimeMs || a.path.localeCompare(b.path));
    for (const { path, size, mtimeMs } of files) {
      if (path === justSaved || inFlightOutputs.has(path)) continue;
      const age = now - mtimeMs;
      if (age > maxAge || (age >= QUOTA_MIN_AGE_MS && totalBytes > maxBytes)) {
        if (await unlink(path).then(() => true, () => false)) totalBytes -= size;
      }
    }
  }
}
