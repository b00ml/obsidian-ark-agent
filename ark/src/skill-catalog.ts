export interface ArkSkill {
  icon: string;
  label: string;
  prompt: string;
}

/** Product-visible skill entrypoints. The prompt is prefilled only; sending remains explicit. */
export const SKILL_CATALOG: readonly ArkSkill[] = [
  { icon: "🎬", label: "BV 号转写", prompt: "按 skills/bili-video-summarizer 的流程处理这个视频：" },
  { icon: "📥", label: "处理收件箱", prompt: "按 skills/inbox-processor 的流程处理收件箱队列：" },
  { icon: "🔍", label: "设计批判性审查", prompt: "按 skills/design-critical-review 的清单审查当前设计：" },
  { icon: "🗒️", label: "每日回顾", prompt: "生成今天的每日回顾笔记（读取今日 Inbox 变动与已完成任务）：" },
  { icon: "📊", label: "周报回顾", prompt: "生成上周的周报回顾（汇总 wiki/ 与 Inbox/ 的新增）：" },
  { icon: "🧪", label: "快速研究草稿", prompt: "进入快速研究草稿模式，先定义问题、范围和资料，再用 rag_retrieve(envelope=true) 获取证据：" },
];

export function skillCatalog(): ArkSkill[] {
  return SKILL_CATALOG.map((skill) => ({ ...skill }));
}
