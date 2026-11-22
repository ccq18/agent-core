import type { CoreToolDefinition } from '../types/tool.js';

export interface ToolSnapshot {
  definitions: ReadonlyMap<string, CoreToolDefinition>;
  visible: CoreToolDefinition[];
}

export class ToolRegistry {
  private readonly groups = new Map<symbol, CoreToolDefinition[]>();
  private readonly builtIns: CoreToolDefinition[];

  constructor(builtIns: readonly CoreToolDefinition[] = []) {
    this.builtIns = [...builtIns];
  }

  register(tools: readonly CoreToolDefinition[]): () => void {
    const id = Symbol();
    this.groups.set(id, [...tools]);
    return () => { this.groups.delete(id); };
  }

  snapshot(include: (tool: CoreToolDefinition) => boolean = () => true): ToolSnapshot {
    const tools = new Map(this.builtIns.map((tool) => [tool.name, tool]));
    for (const group of this.groups.values()) {
      for (const tool of group) tools.set(tool.name, tool);
    }
    return { definitions: tools, visible: [...tools.values()].filter(include) };
  }

  clear(): void {
    this.groups.clear();
  }
}
