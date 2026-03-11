type SDK = {
  invokeFunction<I, O>(fn: string, input: I, timeout?: number): Promise<O>;
};

export class StateKV {
  constructor(private sdk: SDK) {}

  async get<T>(scope: string, key: string): Promise<T | null> {
    try {
      const result = await this.sdk.invokeFunction<
        { scope: string; key: string },
        { value: T | null }
      >("state::get", { scope, key });
      return result?.value ?? null;
    } catch {
      return null;
    }
  }

  async set<T>(scope: string, key: string, data: T): Promise<void> {
    await this.sdk.invokeFunction("state::set", { scope, key, value: data });
  }

  async list<T>(scope: string): Promise<T[]> {
    try {
      const result = await this.sdk.invokeFunction<
        { scope: string },
        { items: T[] }
      >("state::list", { scope });
      return result?.items ?? [];
    } catch {
      return [];
    }
  }

  async delete(scope: string, key: string): Promise<void> {
    await this.sdk.invokeFunction("state::delete", { scope, key });
  }

  async count(scope: string): Promise<number> {
    const items = await this.list(scope);
    return items.length;
  }
}
