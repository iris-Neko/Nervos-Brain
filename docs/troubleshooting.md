# 故障排查

## Docker sock permission denied

现象：

```text
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

检查：

```bash
docker ps
id
```

修复：把当前用户加入 Docker group 后重新登录，或在测试环境临时使用有 Docker 权限的用户执行部署。具体命令因系统发行版不同而不同。

## 忘记 git lfs pull

现象：Qdrant migration 报 archive DB 不是 SQLite，或提示 Git LFS pointer。

检查：

```bash
git lfs status
git lfs ls-files
file data/archive.db data/forum_talk/archive.db data/github_code/archive.db
```

修复：

```bash
git lfs pull
```

修复后再运行 `file data/archive.db data/forum_talk/archive.db data/github_code/archive.db`，正常情况下应显示 SQLite database；如果仍显示文本文件或 Git LFS pointer，说明 LFS 对象没有成功拉下。

## Qdrant 没启动

现象：连接 `http://127.0.0.1:6333` 失败。

检查：

```bash
docker compose -f docker-compose.qdrant.yml ps
curl http://127.0.0.1:6333/collections
```

修复：

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

## Qdrant collection 为空或缺失

现象：Bot 能启动，但检索没有结果；`/collections` 中缺少 `nervos_docs`、`nervos_talk_user_discussions` 或 `nervos_github_code`。

检查：

```bash
curl http://127.0.0.1:6333/collections
```

修复：

```bash
bash bootstrap_qdrant_server.sh
```

或手动重建：

```bash
mamba run -n nervos-brain python scripts/migrate_qdrant_server_from_archive.py \
  --public-default-backends \
  --recreate
```

## config 缺 key 或 token

现象：Bot 启动时报 LLM key、Telegram token 或 Discord token 缺失。

检查：

```bash
test -f config.yaml && echo ok
printenv TELEGRAM_BOT_TOKEN
printenv DISCORD_BOT_TOKEN
```

修复：复制配置模板并填写本地密钥，或通过环境变量提供 token：

```bash
cp config.yaml.example config.yaml
export TELEGRAM_BOT_TOKEN="<TELEGRAM_BOT_TOKEN>"
export DISCORD_BOT_TOKEN="<DISCORD_BOT_TOKEN>"
```

不要把真实值提交到 Git。

## Telegram Bot 没反应

检查顺序：

```bash
ps -eo pid,args | grep run_telegram_bot_polling.py | grep -v grep
tail -n 100 data/logs/telegram_bot_polling.stderr.log
curl http://127.0.0.1:6333/collections
```

常见原因：

```text
Telegram Bot token 错误
allowed_chat_ids 限制不匹配
群聊中没有 mention Bot 或 reply Bot
Qdrant 未启动或 collection 为空
LLM provider 配置错误
```

## Discord Bot 没反应

检查顺序：

```bash
ps -eo pid,args | grep run_discord_bot.py | grep -v grep
printenv DISCORD_BOT_TOKEN
curl http://127.0.0.1:6333/collections
```

常见原因：

```text
Discord Bot token 错误
allowed_guild_ids / allowed_channel_ids 限制不匹配
Guild 消息没有 mention Bot，且 mention_only_in_guild=true
Bot invite 权限或 intents 配置不完整
Discord Developer Portal 未开启 MESSAGE CONTENT INTENT
Qdrant 未启动或 collection 为空
LLM provider 配置错误
```

如果启动时报 `PrivilegedIntentsRequired`，进入 Discord Developer Portal，打开当前 Application 的 Bot 页面，在 Privileged Gateway Intents 中启用 `MESSAGE CONTENT INTENT`，保存后重启 Discord runtime。

如果 Bot 已启动但频道里没有响应，还要确认邀请链接和服务器权限：Bot role 需要能查看目标频道、读取消息历史、发送消息；如果 `mention_only_in_guild=true`，服务器频道里需要 mention Bot 或 reply Bot。

## Telegram/Discord token 错误

现象：平台 API 返回 unauthorized、forbidden 或 login failure。

检查：确认环境变量或 `config.yaml` 中 token 是当前 Bot 的 token。

修复：重新设置环境变量后重启 runtime：

```bash
export TELEGRAM_BOT_TOKEN="<TELEGRAM_BOT_TOKEN>"
bash restart_telegram_bot.sh

export DISCORD_BOT_TOKEN="<DISCORD_BOT_TOKEN>"
mamba run -n nervos-brain python scripts/run_discord_bot.py
```

## mamba 环境名错误

现象：`mamba run -n nervos-brain ...` 报环境不存在。

检查：

```bash
mamba info -e
```

修复：创建正确环境，或临时用环境变量指定脚本使用的环境名：

```bash
mamba env create -f environment.yml
ENV_NAME=nervos-brain bash restart_telegram_bot.sh
```

## Markdown 长消息异常

Telegram 端长回复应走 `text + entities` 发送路径，避免 MarkdownV2 被硬切。Discord 端走 Discord-specific Markdown 分段。

如果长代码块、链接或列表异常，优先跑格式化测试：

```bash
mamba run -n nervos-brain pytest tests/test_platform_formatter.py tests/test_telegram_bot_protocol_adapter.py tests/test_discord_bot_protocol_adapter.py -q
```

## Talk forum 增量更新失败

检查手动命令：

```bash
mamba run -n nervos-brain python scripts/run_talk_forum_ingest.py --latest-pages 3 --incremental
```

如果使用 systemd user timer：

```bash
systemctl --user status nervos-talk-forum-ingest.timer
journalctl --user -u nervos-talk-forum-ingest.service -n 100
```

确认 Qdrant server 正常运行，并且 `config.yaml` 或默认 retrieval 配置指向正确 collection。


## GitHub 增量更新失败

常见原因：GitHub API rate limit、git clone 超时、网络抖动、Qdrant 未启动或 state 与目标列表不兼容。

检查手动命令：

```bash
mamba run -n nervos-brain python scripts/run_github_docs_ingest.py --incremental --no-ingest
mamba run -n nervos-brain python scripts/run_github_code_ingest.py --incremental --no-ingest
```

推荐设置 GitHub token：

```bash
export GITHUB_TOKEN="<GITHUB_TOKEN>"
```

如果怀疑本机 state 卡住，可以重置 state 后重新检查：

```bash
mamba run -n nervos-brain python scripts/run_github_docs_ingest.py --incremental --reset-state
mamba run -n nervos-brain python scripts/run_github_code_ingest.py --incremental --reset-state
```

注意：`data/ingest_state/` 是本机运行游标，不提交；`data/manifests/` 是公开数据版本说明，可以在确认无私密信息后提交。

## Bot 一直显示 typing 但不出消息

现象：Telegram 或 Discord 中看到 Bot 一直处于 typing 状态，迟迟没有最终回答。

常见原因：

```text
LLM provider 响应很慢或网络超时
复杂问题触发多轮检索、反思和重写，耗时超过预期
Qdrant 或 GitHub/Talk 外部请求阻塞
Bot 进程仍在处理前一个长请求，后续请求排队
```

检查：

```bash
tail -n 100 data/logs/telegram_bot_polling.stderr.log
tail -n 20 data/telegram_bot/debug_events.jsonl
ps -eo pid,args | grep run_telegram_bot_polling.py | grep -v grep
curl http://127.0.0.1:6333/collections
```

处理：

```text
先等当前请求完成，复杂请求可能较慢。
如果日志显示 LLM 或网络超时，重试同一问题。
如果进程卡死或长期无新日志，可以重启 Telegram Bot。
如果频繁发生，检查 config 中 target_elapsed_ms / max_elapsed_ms、模型路由和网络质量。
```

重启 Telegram：

```bash
bash restart_telegram_bot.sh
```

Discord 目前通常以前台方式运行；如果卡住，停止当前进程后重新执行：

```bash
mamba run -n nervos-brain python scripts/run_discord_bot.py
```
