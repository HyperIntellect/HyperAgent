#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

function loadFixture(filePath) {
  const raw = fs.readFileSync(filePath, "utf-8");
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed)) {
    throw new Error(`Fixture ${filePath} must be a JSON array`);
  }
  return parsed;
}

function reduceEvents(events) {
  const seenEventIds = new Set();
  const state = {
    content: "",
    stageCount: 0,
    toolCallCount: 0,
    toolResultCount: 0,
    sourceCount: 0,
    duplicateCount: 0,
    consideredCount: 0,
  };

  const dedupeTypes = new Set(["token", "tool_call", "tool_result", "stage", "reasoning", "source"]);
  for (const event of events) {
    const eventId = String(event.event_id || "");
    if (dedupeTypes.has(event.type)) {
      state.consideredCount += 1;
      if (eventId) {
        if (seenEventIds.has(eventId)) {
          state.duplicateCount += 1;
          continue;
        }
        seenEventIds.add(eventId);
      }
    }

    if (event.type === "token") {
      const token = typeof event.data === "string" ? event.data : (event.content || "");
      state.content += token;
    } else if (event.type === "stage") {
      state.stageCount += 1;
    } else if (event.type === "tool_call") {
      state.toolCallCount += 1;
    } else if (event.type === "tool_result") {
      state.toolResultCount += 1;
    } else if (event.type === "source") {
      state.sourceCount += 1;
    }
  }
  return state;
}

function digestState(state) {
  return JSON.stringify({
    content: state.content,
    stageCount: state.stageCount,
    toolCallCount: state.toolCallCount,
    toolResultCount: state.toolResultCount,
    sourceCount: state.sourceCount,
  });
}

function main() {
  const fixturesDir = path.resolve(process.cwd(), "evals/fixtures");
  if (!fs.existsSync(fixturesDir)) {
    console.log(`No fixture dir found at ${fixturesDir}; skipping replay check.`);
    process.exit(0);
  }

  const fixtureFiles = fs
    .readdirSync(fixturesDir)
    .filter((name) => name.endsWith(".json"))
    .map((name) => path.join(fixturesDir, name));

  if (fixtureFiles.length === 0) {
    console.log("No replay fixtures found; skipping replay check.");
    process.exit(0);
  }

  let totalDuplicates = 0;
  let totalConsidered = 0;
  for (const filePath of fixtureFiles) {
    const events = loadFixture(filePath);
    const first = reduceEvents(events);
    const second = reduceEvents(events);
    if (digestState(first) !== digestState(second)) {
      console.error(`Replay mismatch for fixture ${path.basename(filePath)}`);
      process.exit(1);
    }
    totalDuplicates += first.duplicateCount;
    totalConsidered += first.consideredCount;
  }

  const duplicateRate = totalConsidered ? totalDuplicates / totalConsidered : 0;
  console.log(
    JSON.stringify(
      {
        fixtures: fixtureFiles.length,
        duplicateRate,
      },
      null,
      2,
    ),
  );
}

main();

