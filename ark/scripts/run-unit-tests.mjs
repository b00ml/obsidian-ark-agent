// ark 单元测试入口：零新依赖 —— 复用 devDependencies 里已有的 esbuild，
// 把 tests/unit/*.test.ts 打成 CJS（把 "obsidian" 指向 .uiharness 里现成的桩），
// 再交给 node 内置的 test runner 执行。
//
// 为什么不是 vitest：本项目只需要"纯函数级"断言，esbuild + node:test 已经够用，
// 引入测试框架会多一份依赖与配置，收益为零。
import { readdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

import builtins from "builtin-modules";
import esbuild from "esbuild";

const here = dirname(fileURLToPath(import.meta.url));
const arkRoot = join(here, "..");
const outDir = join(arkRoot, ".unit-dist");

rmSync(outDir, { recursive: true, force: true });

await esbuild.build({
  entryPoints: [join(arkRoot, "tests/unit/*.test.ts")],
  bundle: true,
  format: "cjs",
  platform: "node",
  target: "es2020",
  outdir: outDir,
  outExtension: { ".js": ".cjs" },
  // Obsidian API 在纯逻辑测试里只需要"能被解析"；真机行为由 .uiharness 与人工验收覆盖。
  alias: { obsidian: join(arkRoot, ".uiharness/obsidian-stub.ts") },
  external: ["node:test", "node:assert", "node:assert/strict", ...builtins],
  loader: { ".st": "text" },
  logLevel: "warning",
});

// 显式列出产物文件：Node 24 对"目录参数"的处理是当作模块 require，不会自动发现子文件。
const files = readdirSync(outDir).filter((f) => f.endsWith(".cjs")).map((f) => join(outDir, f));
if (!files.length) {
  console.error("[test:unit] 没有产出任何测试文件（tests/unit/*.test.ts 是否为空？）");
  process.exit(1);
}
const run = spawnSync(process.execPath, ["--test", ...files], { stdio: "inherit" });
process.exit(run.status ?? 1);
