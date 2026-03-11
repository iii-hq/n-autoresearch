type SDK = {
  trigger<O>(fn: string, input: unknown, timeout?: number): Promise<O>;
};

export class StateKV {
  constructor(private sdk: SDK) {}

  async get<T>(scope: string, key: string): Promise<T | null> {
    try {
      return await this.sdk.trigger<T | null>(
        "state::get", { scope, key }
      );
    } catch {
      return null;
    }
  }

  async set<T>(scope: string, key: string, value: T): Promise<void> {
    await this.sdk.trigger("state::set", { scope, key, value });
  }

  async list<T>(scope: string): Promise<T[]> {
    try {
      return await this.sdk.trigger<T[]>(
        "state::list", { scope }
      );
    } catch {
      return [];
    }
  }

  async delete(scope: string, key: string): Promise<void> {
    await this.sdk.trigger("state::delete", { scope, key });
  }

  async count(scope: string): Promise<number> {
    const items = await this.list(scope);
    return items.length;
  }
}
