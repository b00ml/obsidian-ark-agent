import type { SpaceOSView } from "../view";
import { getGreeting } from "../utils";
import { getSkin } from "../skins";
import { todoCount, ideaCount } from "../stats";
import { renderAgentWorkbench } from "../agent-workbench";

/** AI 工作台（OPT-109 W1 升级）：HUD 头部保留（日期/统计/状态数据口径），主体升级为
 *  Agent 工作台（agent-workbench.ts）——页面即 Agent 的家，CRT 收敛为快速指令入口。
 *  原欢迎区/信息卡/健康雷达已移出（git 历史可溯；皮肤组件样式保留待 W2 复用）。 */
export function renderArk(view: SpaceOSView, mount: HTMLElement) {
  const s = view.plugin.data.settings;
  const skin = getSkin(s.skin);
  const b = skin.ark;
  mount.addClass("space-ark-full");

  mount.createDiv({ cls: "ark-hud" }, (hud) => {
    const left = hud.createDiv({ cls: "ark-hud-left" });
    left.createDiv({ cls: "ark-hud-label", text: b.hudLabel });
    left.createDiv({ cls: "ark-hud-value", text: b.hudValue });

    const center = hud.createDiv({ cls: "ark-hud-center hud-mono", text: new Date().toLocaleDateString("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit" }) });

    const right = hud.createDiv({ cls: "ark-hud-right" });
    const d = view.plugin.data;
    const stats = [
      { label: b.stats[0], num: todoCount(view.plugin), color: "var(--space-cyan)" },
      { label: b.stats[1], num: 0, color: "var(--space-yellow)" },
      { label: b.stats[2], num: d.database.length, color: "var(--space-purple)" },
      { label: b.stats[3], num: ideaCount(view.plugin), color: "var(--space-green)" },
    ];
    stats.forEach((st) => {
      const stat = right.createDiv({ cls: "hud-stat" });
      stat.createDiv({ cls: "hud-stat-num", text: String(st.num), attr: { style: `color:${st.color}` } });
      stat.createDiv({ cls: "hud-stat-label", text: st.label });
    });
  });

  const greet = mount.createDiv({ cls: "space-ark-center" });
  greet.createDiv({ cls: "space-ark-greeting", text: `${getGreeting()}${s.captainName ? "，" + s.captainName : ""}` });

  renderAgentWorkbench(view, mount);
}
