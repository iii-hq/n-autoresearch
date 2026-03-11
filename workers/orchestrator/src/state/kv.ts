type SDK = {
  trigger<I, O>(fn: string, input: I, timeout?: number): Promise<O>;
};

export class StateKV {
  constructor(private sdk: SDK) {}

  async get<T>(scope: string, key: string): Promise<T | null> {
    try {
      return await this.sdk.trigger<{ scope: string; key: string }, T | null>(
        "state::get", { scope, key }
      );
    } catch {
      return null;
    }
  }

  async set<T>(scope: string, key: string, data: T): Promise<void> {
    await this.sdk.trigger<{ scope: string; key: string; data: T }, T>(
      "state::set", { scope, key, data }
    );
  }

  async list<T>(scope: string): Promise<T[]> {
    try {
      return await this.sdk.trigger<{ scope: string }, T[]>(
        "state::list", { scope }
      );
    } catch {
      return [];
    }
  }

  async delete(scope: string, key: string): Promise<void> {
    await this.sdk.trigger<{ scope: string; key: string }, void>(
      "state::delete", { scope, key }
    );
  }

  async count(scope: string): Promise<number> {
    const items = await this.list(scope);
    return items.length;
  }
}
