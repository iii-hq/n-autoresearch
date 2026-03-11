import { registerWorker } from "iii-sdk";

const sdk = registerWorker("ws://localhost:49134", {
  workerName: "test-state",
});

setTimeout(async () => {
  try {
    console.log("Testing state::set...");
    const setResult = await sdk.trigger("state::set", {
      scope: "test",
      key: "hello",
      value: { msg: "world" },
    });
    console.log("state::set result:", JSON.stringify(setResult));

    console.log("Testing state::get...");
    const getResult = await sdk.trigger("state::get", {
      scope: "test",
      key: "hello",
    });
    console.log("state::get result:", JSON.stringify(getResult));

    console.log("Testing state::list...");
    const listResult = await sdk.trigger("state::list", { scope: "test" });
    console.log("state::list result:", JSON.stringify(listResult));

    console.log("All state tests passed!");
  } catch (err) {
    console.error("State test failed:", err.message);
  }
  process.exit(0);
}, 2000);
