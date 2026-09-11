"""外部 Agent 连接层（P2-1/OPT-112，学 Zed Agent Client Protocol）。

- acp_client.AcpClient：ACP stdio 客户端（newline 分帧 JSON-RPC 2.0）——
  spawn 外部 agent 进程、initialize/session-new/session-prompt、收集
  session/update 流；agent 反向请求（权限/fs）fail-closed 拒绝。
- acp_agent.ExternalAgent：配置驱动的单 agent 外观——惰性启动、会话复用、
  崩溃自愈、事件循环迁移自愈。
"""
from agentlab.connectors.acp_client import AcpClient, AcpError  # noqa: F401
from agentlab.connectors.acp_agent import ExternalAgent  # noqa: F401
