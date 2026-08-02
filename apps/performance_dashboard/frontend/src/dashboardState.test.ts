import assert from "node:assert/strict";
import test from "node:test";

import {
  ALL_COHORTS,
  readDashboardUrl,
  resolveAdvancedFilterSelection,
  resolveCohortSelection,
  writeDashboardUrl
} from "./dashboardState.ts";

test("dashboard URL state round-trips exact cohort and cascading filters", () => {
  const original = new URL("https://dashboard.example/performance?unrelated=kept");
  const written = writeDashboardUrl(original, {
    days: 30,
    model: "wan-t2v",
    gpu: "gpu:abc",
    cohort: "v2:def",
    source: "scheduled_main",
    hardware: "hw-l40s",
    software: "sw-cu130",
    recipe: "recipe-sp2"
  });

  assert.deepEqual(readDashboardUrl(written), {
    days: 30,
    model: "wan-t2v",
    gpu: "gpu:abc",
    cohort: "v2:def",
    source: "scheduled_main",
    hardware: "hw-l40s",
    software: "sw-cu130",
    recipe: "recipe-sp2"
  });
  assert.equal(written.searchParams.get("unrelated"), "kept");
});

test("missing and stale cohort values fall back to the server default", () => {
  assert.equal(resolveCohortSelection(null, ["v2:active"], "v2:active"), "v2:active");
  assert.equal(resolveCohortSelection("v2:stale", ["v2:active"], "v2:active"), "v2:active");
});

test("all cohorts remains an explicit shareable selection", () => {
  assert.equal(resolveCohortSelection(ALL_COHORTS, ["v2:active"], "v2:active"), ALL_COHORTS);
  const written = writeDashboardUrl(new URL("https://dashboard.example/performance"), {
    days: 90,
    model: "",
    gpu: "",
    cohort: ALL_COHORTS,
    source: "",
    hardware: "",
    software: "",
    recipe: ""
  });
  assert.equal(written.searchParams.get("cohort"), ALL_COHORTS);
});

test("invalid URL values safely use dashboard defaults", () => {
  const state = readDashboardUrl(new URL("https://dashboard.example/performance?days=-2&source=nightly"));

  assert.equal(state.days, 90);
  assert.equal(state.source, "");
  assert.equal(state.hardware, "");
  assert.equal(state.software, "");
  assert.equal(state.recipe, "");
});

test("stale advanced filters clear while valid raw IDs remain selected", () => {
  assert.equal(resolveAdvancedFilterSelection("hw-active", ["hw-active", "hw-other"]), "hw-active");
  assert.equal(resolveAdvancedFilterSelection("hw-stale", ["hw-active"]), "");
  assert.equal(resolveAdvancedFilterSelection("", ["hw-active"]), "");
});
