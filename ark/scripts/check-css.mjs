// CSS 结构校验：括号/花括号平衡。
// 背景：一个缺失的 ")" 会让 CSS 解析器吞掉其后整份样式表，
// 表现为"其他页面 UI 全乱"，且构建、TypeScript 都不会报错。
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const file = join(here, "..", "styles.css");
const source = readFileSync(file, "utf8");

/** 去掉注释，但保留换行以便行号准确。 */
function stripComments(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, " "));
}

const cleaned = stripComments(source);
const errors = [];
let braces = 0;
let parens = 0;
let line = 1;
let parenStartLine = 1;

for (let i = 0; i < cleaned.length; i++) {
  const ch = cleaned[i];
  if (ch === "\n") line += 1;
  else if (ch === "(") {
    if (parens === 0) parenStartLine = line;
    parens += 1;
  } else if (ch === ")") {
    parens -= 1;
    if (parens < 0) {
      errors.push(`第 ${line} 行：出现多余的 ")"`);
      parens = 0;
    }
  } else if (ch === "{") {
    if (parens > 0) {
      errors.push(`第 ${line} 行：第 ${parenStartLine} 行开始的 "(" 未闭合就遇到 "{"`);
    }
    braces += 1;
  } else if (ch === "}") {
    if (parens > 0) {
      errors.push(`第 ${parenStartLine} 行：括号未闭合（缺 ")"），这会让解析器吞掉后续全部样式`);
      parens = 0;
    }
    braces -= 1;
    if (braces < 0) {
      errors.push(`第 ${line} 行：出现多余的 "}"`);
      braces = 0;
    }
  }
}

if (parens !== 0) errors.push(`文件结束时仍有 ${parens} 个 "(" 未闭合（第 ${parenStartLine} 行开始）`);
if (braces !== 0) errors.push(`文件结束时花括号不平衡，差值 ${braces}`);

if (errors.length) {
  console.error("[check-css] 发现 CSS 结构错误：");
  for (const e of errors) console.error(`  - ${e}`);
  process.exit(1);
}

console.log("[check-css] OK");
