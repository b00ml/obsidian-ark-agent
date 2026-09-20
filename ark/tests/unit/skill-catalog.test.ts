import assert from "node:assert/strict";
import { test } from "node:test";
import { SKILL_CATALOG, skillCatalog } from "../../src/skill-catalog";

test("skill catalog exposes stable, non-empty prompts", () => {
  const catalog = skillCatalog();
  assert.equal(catalog.length, SKILL_CATALOG.length);
  assert.ok(catalog.length >= 5);
  assert.equal(new Set(catalog.map((skill) => skill.label)).size, catalog.length);
  assert.ok(catalog.every((skill) => skill.icon && skill.label && skill.prompt));
});

test("skill catalog returns copies so UI edits cannot mutate the source", () => {
  const catalog = skillCatalog();
  catalog[0].label = "changed";
  assert.notEqual(SKILL_CATALOG[0].label, "changed");
});
