# 测试与验收

本文档给出部署和改动后的推荐验收命令。默认使用 `mamba run -n nervos-brain ...`，不要求安装项目到 Python 环境。

## 基础检查

```bash
bash -n bootstrap_qdrant_server.sh restart_telegram_bot.sh
mamba run -n nervos-brain python -m py_compile \
  scripts/run_discord_bot.py \
  scripts/run_github_docs_ingest.py \
  scripts/run_github_code_ingest.py \
  scripts/migrate_qdrant_server_from_archive.py \
  scripts/run_talk_mcp_server.py \
  scripts/run_talk_forum_ingest.py \
  src/nervos_brain/tool_runtime/telegram_bot_runtime.py \
  src/nervos_brain/tool_runtime/discord_bot_runtime.py \
  src/nervos_brain/tool_runtime/talk_mcp_adapter.py
```

确认 Git LFS 已拉取三套 archive DB：

```bash
git lfs ls-files | rg 'data/(archive|forum_talk/archive|github_code/archive)\.db'
file data/archive.db data/forum_talk/archive.db data/github_code/archive.db
```

## Qdrant 和检索

```bash
curl http://127.0.0.1:6333/collections
mamba run -n nervos-brain pytest tests/test_qdrant_server_migration.py tests/test_retrieval_unit.py -q
```

如果修改多 backend 或统一检索行为：

```bash
mamba run -n nervos-brain pytest tests/test_composite_retriever.py tests/test_retrieval_unit.py tests/test_full_graph.py -q
```

## Telegram / Discord runtime

Telegram 相关：

```bash
mamba run -n nervos-brain pytest tests/test_telegram_bot_runtime.py tests/test_telegram_bot_protocol_adapter.py tests/test_feedback.py -q
```

Discord 相关：

```bash
mamba run -n nervos-brain pytest tests/test_discord_bot_runtime.py tests/test_discord_bot_protocol_adapter.py tests/test_platform_formatter.py -q
```

## 格式化输出

修改 Markdown、分段、entities、Discord message payload 时：

```bash
mamba run -n nervos-brain pytest tests/test_platform_formatter.py tests/test_telegram_bot_protocol_adapter.py tests/test_discord_bot_protocol_adapter.py -q
```

## Talk MCP 和论坛增量更新

```bash
mamba run -n nervos-brain pytest tests/test_talk_mcp_adapter.py tests/test_talk_forum_timer_templates.py tests/test_discourse_parallel.py tests/test_tool_runtime.py -q
```

## Graph / Prompt / Reflection

```bash
mamba run -n nervos-brain pytest tests/test_full_graph.py tests/test_reflection_module.py tests/test_week4_graph_runtime.py -q
```

## 部署验收清单

- `git lfs pull` 已执行。
- `data/archive.db`、`data/forum_talk/archive.db`、`data/github_code/archive.db` 都不是 LFS pointer。
- `config.yaml` 已创建但未提交。
- `docker compose -f docker-compose.qdrant.yml ps` 显示 Qdrant 正常。
- `curl http://127.0.0.1:6333/collections` 能看到三套 collection。
- Telegram Bot 能启动，日志无 token/config 缺失错误。
- Discord Bot 能启动，Developer Portal 已开启 `MESSAGE CONTENT INTENT`，guild/channel 限制符合预期。
- Talk forum ingest timer 如需启用，`systemctl --user list-timers` 能看到下一次执行时间。

## 人工验收建议

- Telegram 群里 mention Bot 提一个 CKB 基础问题，确认只在被唤起时回复。
- Reply Bot 上一条消息并发送“用小白版解释一下”，确认回答围绕被 reply 的消息，而不是普通最近历史。
- 提一个需要资料的问题，例如 CCC / Fiber / Spore DOB，确认回答走检索并带引用。
- Discord 频道里 mention Bot 测试同类问题，确认 Developer Portal intents 和频道权限正确。
- 测试长代码块、链接、列表和引用，确认 Telegram / Discord 格式没有被分段破坏。
- 如启用 CSAT 或 `/feedback`，确认本地 feedback 文件有记录，且不会提交到 Git。
