# 新服务器部署

本文档描述从 fresh clone 到 Bot 可运行的标准流程。默认目标环境是 Linux、Conda/Mamba、Docker、Docker Qdrant server 和 `nervos-brain` Python 环境。

## 1. 准备系统依赖

需要先准备：

```text
git
git-lfs
conda、mamba 或 micromamba
Docker 和 Docker Compose
```

确认命令可用：

```bash
git --version
git lfs version
conda --version  # 或 mamba --version / micromamba --version
docker --version
docker compose version
```

如果 `docker ps` 报 `permission denied while trying to connect to the docker API`，说明当前用户没有 Docker daemon 权限。处理方式见 [故障排查](troubleshooting.md)。

## 2. Clone 仓库和数据

```bash
git lfs install
git clone https://github.com/iris-Neko/Nervos-Brain.git
cd Nervos-Brain
git checkout dev  # 当前交付/部署分支
git lfs pull
```

`git lfs pull` 很重要。三套 archive DB 和部分 Qdrant fallback 文件由 LFS 管理，如果只拿到 LFS pointer，后续迁移会失败。三套 archive DB 是：

```text
data/archive.db
data/forum_talk/archive.db
data/github_code/archive.db
```

## 3. 创建 Python 环境

```bash
conda env create -f environment.yml
# 如果 nervos-brain 环境已存在，改用：
# conda env update -n nervos-brain -f environment.yml --prune
conda run -n nervos-brain python --version
```

如果机器安装了 `mamba` 或 `micromamba`，上面的 `conda` 可以替换为对应命令。项目自带脚本会按 `mamba`、`micromamba`、`conda` 的顺序自动探测；也可以用 `ENV_RUNNER_BIN=/path/to/conda` 显式指定。

本项目测试和脚本默认从仓库运行，不要求 `pip install -e .`。

## 4. 创建本地配置

```bash
cp config.yaml.example config.yaml
```

在 `config.yaml` 中填写 LLM provider、Telegram Bot、Discord Bot 等本地配置。最低必填项：

```text
llm.api_key 或对应 provider 的 API key
llm.api_base，如果使用 OpenAI-compatible endpoint
TELEGRAM_BOT_TOKEN，如果启动 Telegram Bot
DISCORD_BOT_TOKEN，如果启动 Discord Bot
```

也可以用环境变量提供平台 token：

```bash
export TELEGRAM_BOT_TOKEN="<TELEGRAM_BOT_TOKEN>"
export DISCORD_BOT_TOKEN="<DISCORD_BOT_TOKEN>"
```

`config.yaml` 不应提交。`config.yaml.example` 只是模板，runtime 不会把它当正式配置 fallback。

## 5. 启动 Qdrant server 并重建 collection

推荐部署使用 Docker Qdrant server，避免 Telegram 和 Discord 多进程同时访问本地 Qdrant 目录产生锁冲突。

```bash
bash bootstrap_qdrant_server.sh
```

脚本会执行：

```text
1. docker compose -f docker-compose.qdrant.yml up -d
2. 等待 http://127.0.0.1:6333 可用
3. 从 SQLite archive DB 重建 docs/forum/github_code 三套 collection
4. 打印 Qdrant collection 状态
```

手动检查：

```bash
curl http://127.0.0.1:6333/collections
```

## 6. 启动 Telegram Bot

```bash
export TELEGRAM_BOT_TOKEN="<TELEGRAM_BOT_TOKEN>"
bash restart_telegram_bot.sh
```

日志默认在：

```text
data/logs/telegram_bot_polling.stdout.log
data/logs/telegram_bot_polling.stderr.log
```

## 7. 启动 Discord Bot

```bash
export DISCORD_BOT_TOKEN="<DISCORD_BOT_TOKEN>"
conda run -n nervos-brain python scripts/run_discord_bot.py
```

这个命令是前台运行，适合首次验证。长期运行请用 `tmux`、`systemd`、`supervisor` 或部署方已有进程管理；当前仓库暂未提供 Discord 专用 restart 脚本。Discord 配置见 `config.yaml.example` 的 `discord_bot` 区块。

Discord Developer Portal 必须为该 Bot 开启 `MESSAGE CONTENT INTENT`，否则 runtime 会在连接 Discord Gateway 时抛出 `PrivilegedIntentsRequired`。邀请 Bot 到服务器时，至少需要让它能读取频道消息并发送消息；如果频道有更细权限控制，还要确认 Bot role 能访问目标 channel。

## 8. 多群和并发边界

一个 Telegram Bot token 只应启动一个 polling 进程。该进程可以同时服务多个群聊；用 `telegram_bot.allowed_chat_ids` 白名单控制允许响应的群。为空时不限制群；正式部署建议显式填写允许的群。超出 `telegram_bot.max_worker_threads` 的并发请求会排队。

Discord 也不建议用同一个 Bot token 启多个 gateway 进程。首次部署保持一个 Discord runtime 进程即可；可用 `discord_bot.allowed_guild_ids` 和 `discord_bot.allowed_channel_ids` 控制响应范围。并发由 `discord_bot.max_worker_threads` 控制，超出后排队。

## 9. 部署后验收

建议至少运行：

```bash
bash -n bootstrap_qdrant_server.sh restart_telegram_bot.sh
conda run -n nervos-brain pytest tests/test_qdrant_server_migration.py tests/test_retrieval_unit.py -q
conda run -n nervos-brain python -m py_compile \
  scripts/migrate_qdrant_server_from_archive.py \
  scripts/run_talk_mcp_server.py \
  scripts/run_talk_forum_ingest.py
```

更多验收项目见 [测试与验收](testing-and-acceptance.md)。

交付接手清单和能力边界见 [交付说明](handover.md)。
