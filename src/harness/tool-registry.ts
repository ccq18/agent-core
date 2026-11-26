import type { CoreToolDefinition } from '../types/tool.js';

export interface ToolSnapshot {
  definitions: ReadonlyMap<string, CoreToolDefinition>;
  visible: CoreToolDefinition[];
}

export class ToolRegistry {
  private readonly groups = new Map<symbol, CoreToolDefinition[]>();
  private readonly builtIns: CoreToolDefinition[];
  private revision = 0;

  constructor(builtIns: readonly CoreToolDefinition[] = []) {
    this.builtIns = [...builtIns];
  }

  get version(): number {
    return this.revision;
  }

  definitionFingerprint(): string {
    let fingerprint = '';
    for (const group of [this.builtIns, ...this.groups.values()]) {
      for (const tool of group) {
        fingerprint += JSON.stringify([tool.name, tool.readOnly, tool.executionMode, tool.permissionAction]);
      }
    }
    return fingerprint;
  }

  invalidate(): void {
    this.revision++;
  }

  register(tools: readonly CoreToolDefinition[]): () => void {
    const id = Symbol();
    this.groups.set(id, [...tools]);
    this.invalidate();
    return () => { if (this.groups.delete(id)) this.invalidate(); };
  }

  snapshot(include: (tool: CoreToolDefinition) => boolean = () => true): ToolSnapshot {
    const tools = new Map(this.builtIns.map((tool) => [tool.name, tool]));
    for (const group of this.groups.values()) {
      for (const tool of group) tools.set(tool.name, tool);
    }
    return { definitions: tools, visible: [...tools.values()].filter(include) };
  }

  clear(): void {
    if (this.groups.size) this.invalidate();
    this.groups.clear();
  }
}
