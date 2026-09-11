/** 逐词 Diff（W3/OPT-067，Claudian 形态）：润色结果与原文的词级对比，零外部依赖。
 *
 * 切词粒度：CJK 单字成段（中文改动逐字可见）、英文/数字成词、空白与标点成段。
 * 算法：修剪公共前后缀 → 中段 LCS DP（规模护栏：token 乘积 >4M 时退化为整段替换，
 * 典型选区几百 token 远够）；回溯后合并相邻同类型段。
 */

export type DiffSeg = { type: "same" | "ins" | "del"; text: string };

function tokenize(text: string): string[] {
  return text.match(/[\u4e00-\u9fff]|[A-Za-z0-9_]+|\s+|[^\sA-Za-z0-9_\u4e00-\u9fff]+/g) ?? [];
}

export function wordDiff(a: string, b: string): DiffSeg[] {
  const A = tokenize(a);
  const B = tokenize(b);
  const n = A.length;
  const m = B.length;

  // 1) 修剪公共前后缀
  let pre = 0;
  while (pre < n && pre < m && A[pre] === B[pre]) pre++;
  let suf = 0;
  while (suf < n - pre && suf < m - pre && A[n - 1 - suf] === B[m - 1 - suf]) suf++;

  const segs: DiffSeg[] = [];
  if (pre) segs.push({ type: "same", text: A.slice(0, pre).join("") });

  // 2) 中段 LCS（带规模护栏）
  const midA = A.slice(pre, n - suf);
  const midB = B.slice(pre, m - suf);
  const N = midA.length;
  const M = midB.length;
  if (N > 0 || M > 0) {
    if (N * M > 4_000_000) {
      if (N) segs.push({ type: "del", text: midA.join("") });
      if (M) segs.push({ type: "ins", text: midB.join("") });
    } else {
      const width = M + 1;
      const dp = new Uint32Array((N + 1) * width);
      for (let i = 1; i <= N; i++) {
        for (let j = 1; j <= M; j++) {
          dp[i * width + j] = midA[i - 1] === midB[j - 1]
            ? dp[(i - 1) * width + j - 1] + 1
            : Math.max(dp[(i - 1) * width + j], dp[i * width + j - 1]);
        }
      }
      const rev: DiffSeg[] = [];
      let i = N;
      let j = M;
      while (i > 0 && j > 0) {
        if (midA[i - 1] === midB[j - 1]) {
          rev.push({ type: "same", text: midA[i - 1] });
          i--; j--;
        } else if (dp[(i - 1) * width + j] >= dp[i * width + j - 1]) {
          rev.push({ type: "del", text: midA[i - 1] });
          i--;
        } else {
          rev.push({ type: "ins", text: midB[j - 1] });
          j--;
        }
      }
      while (i > 0) { rev.push({ type: "del", text: midA[--i] }); }
      while (j > 0) { rev.push({ type: "ins", text: midB[--j] }); }
      rev.reverse();
      for (const s of rev) {
        const last = segs[segs.length - 1];
        if (last && last.type === s.type) last.text += s.text;
        else segs.push({ type: s.type, text: s.text });
      }
    }
  }

  // 3) 公共后缀
  if (suf) {
    const tail = A.slice(n - suf).join("");
    const last = segs[segs.length - 1];
    if (last && last.type === "same") last.text += tail;
    else segs.push({ type: "same", text: tail });
  }
  return segs.filter((s) => s.text.length > 0);
}

/** 把逐词 Diff 渲染进容器（删除=红删除线，新增=绿底） */
export function renderWordDiff(el: HTMLElement, original: string, modified: string): void {
  for (const s of wordDiff(original, modified)) {
    if (s.type === "same") el.createSpan({ text: s.text });
    else if (s.type === "del") el.createSpan({ cls: "wdiff-del", text: s.text });
    else el.createSpan({ cls: "wdiff-ins", text: s.text });
  }
}
