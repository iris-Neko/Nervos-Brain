# 交付说明

本文档面向项目接收方，用于快速了解 Nervos Brain 当前可交付能力、运行依赖、数据更新方式、私密配置清单、常用运维命令和后续扩展方向。

更细的部署步骤见 [新服务器部署](deployment.md)，配置字段见 [配置说明](configuration.md)，数据重建见 [检索数据与 Qdrant 重建](retrieval-data.md)，运行排障见 [故障排查](troubleshooting.md)。

## 1. 当前交付内容

公开仓库包含：

```text
代码与测试
README 和 docs 工程文档
config.yaml.example 配置模板
environment.yml Python/mamba 环境
Docker Qdrant compose 文件
Telegram / Discord Bot runtime
Nervos Talk MCP 只读查询服务
Talk forum 定时增量更新模板
GitHub docs/code 增量更新脚本与 systemd timer 模板
SQLite archive DB 与 source JSONL 等公开检索数据
Qdrant server 重建脚本
```

公开仓库不包含：

```text
config.yaml
LLM API key
Telegram Bot token
Discord Bot token
Qdrant API key
GitHub token
真实群聊记录
runtime debug events
feedback.jsonl
memory DB
runtime logs
Docker Qdrant server 运行目录
Telegram MCP session state
```

## 2. 主要能力边界

当前系统是面向 CKB/Nervos 的 Telegram / Discord Agentic RAG Bot，主要能力包括：

1. 统一检索 docs、Nervos Talk forum、GitHub code 三类资料。
2. 基于 full graph 做信息缺口判断、检索规划、证据合并、回答生成和反思检查。
3. Telegram 群聊按 mention / reply / command 触发，普通群聊默认不插话。
4. Discord guild 内按 mention / reply 触发，可配置 guild/channel allowlist。
5. Telegram / Discord 支持长消息分段、Markdown/代码块格式保护和 plain fallback。
6. 支持 CSAT、feedback、debug event 和用户/群/线程隔离的短期 memory。
7. 支持 Nervos Talk MCP 只读实时查询公开论坛内容。
8. 支持 Talk forum、GitHub docs/code 的定时增量更新。

当前不承诺的能力：

```text
不自动访问私有论坛分类
不代表用户发 Telegram/Discord 普通消息
不在普通 Bot 问答请求中触发爬虫写库
不内置真实 Bot token 或 LLM key
不保证所有 CKB/生态 API 永远最新，仍需通过检索和定期更新维护
不建议同一个 Telegram/Discord Bot token 启动多个 runtime 进程
```

## 3. 运行依赖

推荐环境：

```text
Linux 服务器
git + git-lfs
mamba / miniforge
Docker + Docker Compose
可访问 LLM provider 的网络
可访问 Telegram / Discord / GitHub / talk.nervos.org 的网络
```

Python 环境默认名称：

```text
nervos-brain
```

创建或更新环境：

```bash
mamba env create -f environment.yml
# 已存在时：
mamba env update -n nervos-brain -f environment.yml --prune
```

## 4. 接收方需要准备的私密配置

最低需要：

```text
LLM API key
LLM API base，如果使用 OpenAI-compatible endpoint
Telegram Bot token，如果启用 Telegram
Discord Bot token，如果启用 Discord
```

可选：

```text
GitHub token：降低 GitHub 增量更新 rate limit 风险
Qdrant API key：仅远程/云端 Qdrant 需要
Nervos Talk API key / API user：公开只读查询不需要，访问私有分类或遇到限流时才需要
Telegram API ID / API Hash：仅 Telegram MCP 需要，不是 Telegram Bot runtime 所需
```

建议通过环境变量或本地 `config.yaml` 填写真实值，不要写入公开仓库。

## 5. 标准部署路径

从 fresh clone 到可运行的最短路径：

```bash
git lfs install
git clone https://github.com/iris-Neko/Nervos-Brain.git
cd Nervos-Brain
git checkout dev
git lfs pull

mamba env create -f environment.yml
cp config.yaml.example config.yaml
```

填写 `config.yaml` 或环境变量后，启动 Qdrant 并重建三套 collection：

```bash
bash bootstrap_qdrant_server.sh
```

启动 Telegram：

```bash
export TELEGRAM_BOT_TOKEN="<TELEGRAM_BOT_TOKEN>"
bash restart_telegram_bot.sh
```

启动 Discord：

```bash
export DISCORD_BOT_TOKEN="<DISCORD_BOT_TOKEN>"
mamba run -n nervos-brain python scripts/run_discord_bot.py
```

Discord 首次部署要在 Developer Portal 开启 `MESSAGE CONTENT INTENT`，并确认 Bot role 有查看频道、读取消息历史和发送消息权限。

## 6. 检索数据和数据库关系

项目使用三套检索数据：

```text
retrieval              -> nervos_docs
retrieval_forum_talk   -> nervos_talk_user_discussions
retrieval_github_code  -> nervos_github_code
```

关系可以这样理解：

```text
SQLite archive DB = 可发布、可迁移、可重建的资料源
Docker Qdrant server = 运行时向量索引服务
```

新机器不需要提交或拷贝 `data/qdrant_server/`。部署时从 Git LFS 拉取 archive DB，再用脚本重建 Qdrant collection。

必须确认：

```bash
git lfs pull
git lfs ls-files
file data/archive.db data/forum_talk/archive.db data/github_code/archive.db
```

正常情况下三个 archive DB 应显示为 SQLite database，而不是 Git LFS pointer 文本。

## 7. 数据更新方式

Talk forum 日常增量：

```bash
mamba run -n nervos-brain python scripts/run_talk_forum_ingest.py --latest-pages 3 --incremental
```

GitHub docs/code 日常增量：

```bash
export GITHUB_TOKEN="<GITHUB_TOKEN>"
mamba run -n nervos-brain python scripts/run_github_docs_ingest.py --incremental
mamba run -n nervos-brain python scripts/run_github_code_ingest.py --incremental
```

推荐使用 `deploy/systemd/` 下的 user timer：

```text
nervos-talk-forum-ingest.timer   # 默认 24 小时一次
nervos-github-ingest.timer       # 默认每周一次
```

增量更新可以在 Bot 运行时执行。Qdrant 向量检索通常能直接看到新增内容；如果希望 BM25/fuzzy/exact 内存索引也马上刷新，增量完成后重启 Bot 即可。

## 8. 常用运维命令

Telegram：

```bash
bash restart_telegram_bot.sh
ps -eo pid,args | grep run_telegram_bot_polling.py | grep -v grep
tail -n 100 data/logs/telegram_bot_polling.stderr.log
```

Discord：

```bash
mamba run -n nervos-brain python scripts/run_discord_bot.py
ps -eo pid,args | grep run_discord_bot.py | grep -v grep
```

Qdrant：

```bash
docker compose -f docker-compose.qdrant.yml ps
curl http://127.0.0.1:6333/collections
bash bootstrap_qdrant_server.sh
```

Talk forum timer：

```bash
systemctl --user status nervos-talk-forum-ingest.timer
journalctl --user -u nervos-talk-forum-ingest.service -n 100
```

GitHub ingest timer：

```bash
systemctl --user status nervos-github-ingest.timer
journalctl --user -u nervos-github-ingest.service -n 100
```

## 9. 推荐验收

基础检查：

```bash
bash -n bootstrap_qdrant_server.sh restart_telegram_bot.sh
mamba run -n nervos-brain python -m py_compile \
  scripts/run_discord_bot.py \
  scripts/run_talk_mcp_server.py \
  scripts/run_talk_forum_ingest.py \
  scripts/run_github_docs_ingest.py \
  scripts/run_github_code_ingest.py \
  scripts/migrate_qdrant_server_from_archive.py
```

核心测试：

```bash
mamba run -n nervos-brain pytest tests/test_qdrant_server_migration.py tests/test_retrieval_unit.py -q
mamba run -n nervos-brain pytest tests/test_telegram_bot_runtime.py tests/test_telegram_bot_protocol_adapter.py -q
mamba run -n nervos-brain pytest tests/test_discord_bot_runtime.py tests/test_discord_bot_protocol_adapter.py tests/test_platform_formatter.py -q
mamba run -n nervos-brain pytest tests/test_talk_mcp_adapter.py tests/test_talk_forum_timer_templates.py -q
```

人工验收建议：

1. Telegram 群里 mention Bot 提问一个 CKB 基础问题。
2. Reply Bot 上一条消息，发送“用小白版解释一下”，确认回答围绕被 reply 消息。
3. 提一个需要资料的问题，例如 CCC / Fiber / Spore DOB，确认回答有检索和引用。
4. Discord 频道里 mention Bot 测试同类问题。
5. 发送长代码或长列表回答，确认 Telegram / Discord 格式正常。
6. 点一次 CSAT 或发送 `/feedback`，确认 feedback 文件有记录。

## 10. 后续可扩展方向

短期可做：

1. 完成 fresh clone 部署演练和最终文档校对。
2. 收集更多真实用户反馈，扩充 `evaluation/human_collected_cases.jsonl`。
3. 继续优化复杂问题响应时间和模型路由成本。
4. 给 Discord 增加类似 Telegram 的 restart/systemd 运行脚本。
5. 在服务器上长期验证 GitHub 大仓库增量更新稳定性。

中期可做：

1. 增加统一任务队列，提升多用户并发下的可控性和可观测性。
2. 增加更细粒度的回答质量评测和自动回归报告。
3. 支持云端 Qdrant 的备份、恢复和 API key 部署模式。
4. 把 Talk MCP 接入更多 Agent 工具调用场景。
5. 增加更完善的运行监控、告警和日志轮转策略。
