分析视频《{{title}}》的字幕，找出最适合截图的时间点。

规则：
- 最多 {{max_count}} 张截图
- 优先选择：代码展示、图表出现、关键操作步骤、PPT 切换
- 返回 JSON 数组: [{"timestamp": 秒, "reason": "原因"}]

字幕文本（含 [M:SS] 时间戳）：
{{transcript}}

只返回 JSON，不要其他文字。
