# 运行与运维

本文档记录 Telegram、Discord、Qdrant 和 Talk forum 增量更新的日常操作。

## Telegram Bot

启动或重启：

```bash
export TELEGRAM_BOT_TOKEN="<TELEGRAM_BOT_TOKEN>"
bash restart_telegram_bot.sh
```

常用环境变量：

```text
ENV_NAME=nervos-brain
ENV_RUNNER_BIN=conda
PROJECT_ROOT=/optional/repo/root
DEBUG=1
DRY_RUN=1
DROP_PENDING_ON_START=1
```

查看日志：

```bash
tail -n 100 data/logs/telegram_bot_polling.stdout.log
tail -n 100 data/logs/telegram_bot_polling.stderr.log
```

查看进程：

```bash
ps -eo pid,args | grep scripts/run_telegram_bot_polling.py | grep -v grep
```

如果系统支持 user systemd，`restart_telegram_bot.sh` 会优先用 `systemd-run --user` 管理进程；否则回退到 `nohup`。同一个 Telegram Bot token 不要启动多个 polling 进程；多群聊由一个进程按 `chat_id` 区分处理，响应范围由 `telegram_bot.allowed_chat_ids` 控制。

## Discord Bot

启动：

```bash
export DISCORD_BOT_TOKEN="<DISCORD_BOT_TOKEN>"
conda run -n nervos-brain python scripts/run_discord_bot.py
```

该命令是前台运行，适合首次验证。长期运行请用 `tmux`、`systemd`、`supervisor` 或部署方已有进程管理；当前仓库暂未提供 Discord 专用 restart 脚本。Discord Developer Portal 必须开启 `MESSAGE CONTENT INTENT`。同一个 Discord Bot token 不建议启动多个 gateway 进程，响应范围由 `discord_bot.allowed_guild_ids` 和 `discord_bot.allowed_channel_ids` 控制。

Discord runtime 数据默认写入：

```text
data/discord_bot/memory.db
data/discord_bot/feedback.jsonl
data/discord_bot/debug_events.jsonl
```

这些都是本地 runtime 私有数据，不提交。

## 用户命令

Telegram 和 Discord 都支持用户侧 `/fast` 一次性加速开关：

```text
/fast
/fast off
/fast status
```

`/fast` 不会立刻进入问答，也不是群内排队优先；它只会让当前用户的下一次正常请求尝试使用更快的模型 API 服务层级（内部为 `service_tier="priority"`）。下一条请求开始处理后，这个标记会自动清除；其他用户不受影响。它不改变模型档位、检索策略或回答 prompt，因此更适合临时赶时间的问题。priority 通常更贵，默认不开启。

状态文件默认写入 `data/runtime/fast_mode_state.json`，只保存 `platform + user_id` 的待消费标记，不保存聊天内容或密钥。该目录是本地 runtime 私有数据，不提交。

Telegram 还支持 `/chatid` 诊断命令，用于确认 Bot 当前所在群组和 Topic：

```text
/chatid@NBCKB_Bot
```

Bot 会直接返回群标题、`chat_id`，以及当前 Topic 的 `message_thread_id`（如果消息位于 Topic 中）。该命令不进入 Graph，也不受 `allowed_chat_ids` 和 Topic 白名单限制，方便在别人管理的新群里先确认 ID；其它普通消息仍然遵守原有白名单和 mention 规则。

## Qdrant Docker server

启动：

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

查看状态：

```bash
docker compose -f docker-compose.qdrant.yml ps
curl http://127.0.0.1:6333/collections
```

重建 collection：

```bash
conda run -n nervos-brain python scripts/migrate_qdrant_server_from_archive.py \
  --public-default-backends \
  --recreate
```

一键 bootstrap：

```bash
bash bootstrap_qdrant_server.sh
```

## GitHub docs/code 每周增量更新

GitHub 文档和代码库默认每周增量更新一次。增量逻辑会检查每个目标 repo 的 default branch 最新 commit：未变化的 repo 会跳过，变化的 repo 会重新爬取并清理该 repo 旧 commit 的记录。

一次性手动执行：

```bash
conda run -n nervos-brain python scripts/run_github_docs_ingest.py --incremental
conda run -n nervos-brain python scripts/run_github_code_ingest.py --incremental
```

推荐配置 `GITHUB_TOKEN` 以降低 rate limit 风险：

```bash
export GITHUB_TOKEN="<GITHUB_TOKEN>"
```

安装 systemd user timer：

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/nervos-github-ingest.service ~/.config/systemd/user/
cp deploy/systemd/nervos-github-ingest.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now nervos-github-ingest.timer
```

该 timer 默认每周运行一次，不会自动重启 Bot。和 Talk 增量一样，Qdrant 向量检索通常能直接看到新增内容；如果希望 BM25/fuzzy/exact 也马上刷新，增量结束后手动重启 Bot。

运行 state 默认写入 `data/ingest_state/`，这是本机私有游标，不提交。它记录“这台机器上次检查到了哪个 commit”，提交后会让别人的服务器误以为自己已经更新过。

公开数据版本说明写入 `data/manifests/`，用于记录当前公开数据来自哪些 repo/branch/commit，以及每个 repo 写入了多少条记录。manifest 不含密钥或机器状态，可以作为交付验收清单提交。

## Talk forum 24 小时增量更新

一次性手动执行：

```bash
conda run -n nervos-brain python scripts/run_talk_forum_ingest.py --latest-pages 3 --incremental
```

增量更新不需要先停止 Bot。它会把新论坛内容写入 SQLite archive 和 Qdrant server；后续向量检索通常可以直接看到新数据。需要注意的是，正在运行的 Bot 进程里的 BM25/fuzzy/exact 相关索引是启动时从 archive DB 构建的内存快照。如果希望所有检索路径都立刻使用新增内容，增量更新完成后重启 Bot 即可，通常耗时不到一分钟。

安装 systemd user timer：

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/nervos-talk-forum-ingest.service ~/.config/systemd/user/
cp deploy/systemd/nervos-talk-forum-ingest.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now nervos-talk-forum-ingest.timer
```

默认频率是 24 小时。该 service 是模板，安装前必须确认或编辑：

```ini
Environment=PROJECT_ROOT=%h/path/to/Nervos-Brain
Environment=ENV_RUNNER_BIN=%h/miniconda3/bin/conda
Environment=ENV_NAME=nervos-brain
Environment=TALK_LATEST_PAGES=3
```

查看 timer 和日志：

```bash
systemctl --user status nervos-talk-forum-ingest.timer
systemctl --user list-timers nervos-talk-forum-ingest.timer
journalctl --user -u nervos-talk-forum-ingest.service -n 100
```

## 运行边界

- Telegram/Discord 普通问答不触发爬虫。
- Telegram/Discord 并发由各自 `max_worker_threads` 控制；超过 worker 数的请求会排队。
- Nervos Talk MCP 是只读实时查询工具，不写 archive 或 Qdrant。
- Talk forum 写库由定时增量 ingest 服务负责。
- `data/logs/`、`data/telegram_bot/`、`data/discord_bot/`、`data/qdrant_server/` 不提交。
