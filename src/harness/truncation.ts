import { createHash, randomUUID } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import type { TruncationOptions } from '../types/config.js';
import type { CoreToolResult } from '../types/tool.js';

export interface TruncatedOutputDetails {
  truncated: true;
  outputPath: string;
}

function safeFilePart(value: string): string {
  return /^[A-Za-z0-9_-]{1,128}$/.test(value)
    ? value : createHash('sha256').update(value).digest('hex');
}

function previewLength(text: string, maxLines: number, maxBytes: number): number {
  let lines = 1;
  let bytes = 0;
  let end = 0;
  for (const point of text) {
    bytes += Buffer.byteLength(point);
    if (point === '\n') lines++;
    if (lines > maxLines || bytes > maxBytes) break;
    end += point.length;
  }
  return end;
}

type ToolOutput = Pick<CoreToolResult, 'content' | 'details'>;

export class OutputTruncator {
  private readonly directory: string;
  private readonly maxLines: number;
  private readonly maxBytes: number;

  constructor(dataDir: string, options: TruncationOptions = {}) {
    this.directory = resolve(dataDir, 'tool-output');
    this.maxLines = options.maxLines ?? 2000;
    this.maxBytes = options.maxBytes ?? 50 * 1024;
  }

  async truncate<T extends ToolOutput>(sessionId: string, toolCallId: string, result: T): Promise<Omit<T, keyof ToolOutput> & ToolOutput> {
    const text = result.content.flatMap((block) => block.type === 'text' ? [block.text] : []).join('\n');
    const end = previewLength(text, this.maxLines, this.maxBytes);
    if (end === text.length) return result;

    const outputPath = await this.save(sessionId, toolCallId, text);
    const note = `[Output truncated. Full output: ${outputPath}]`;
    const content: CoreToolResult['content'] = [];
    let remaining = end;
    let lastText = -1;
    for (const block of result.content) {
      if (block.type !== 'text') {
        content.push(block);
      } else if (remaining > 0) {
        lastText = content.length;
        content.push({ ...block, text: block.text.slice(0, remaining) });
        remaining -= block.text.length + 1;
      }
    }
    const last = content[lastText];
    if (last?.type === 'text') last.text += `\n\n${note}`;
    else content.push({ type: 'text', text: note });

    const metadata: TruncatedOutputDetails = { truncated: true, outputPath };
    const details = result.details;
    const existing = details && typeof details === 'object' && !Array.isArray(details) ? details
      : details === undefined ? {} : { originalDetails: details };
    return { ...result, content, details: { ...existing, ...metadata } };
  }

  private async save(sessionId: string, toolCallId: string, text: string): Promise<string> {
    // Keep both sanitized components within the filesystem's filename length limit.
    const session = safeFilePart(sessionId);
    const call = safeFilePart(toolCallId);
    const filename = session.length + call.length <= 210 ? `${session}-${call}`
      : `${session}-${createHash('sha256').update(call).digest('hex')}`;
    await mkdir(this.directory, { recursive: true, mode: 0o700 });
    const outputPath = join(this.directory, filename + '.txt');
    try {
      await writeFile(outputPath, text, { flag: 'wx', mode: 0o600 });
      return outputPath;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
      const unique = join(this.directory, `${filename}-${randomUUID()}.txt`);
      await writeFile(unique, text, { flag: 'wx', mode: 0o600 });
      return unique;
    }
  }
}
