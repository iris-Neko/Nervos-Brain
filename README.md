# Nervos Brain

Nervos Brain 是面向 CKB/Nervos 的 Telegram / Discord Agentic RAG Bot。它把官方文档、Nervos Talk 论坛讨论和 GitHub 源码资料接入统一检索流程，并通过 LangGraph 完成信息缺口判断、检索规划、证据合并、回答生成、反思检查和平台格式化。

默认部署目标是 Linux + Conda/Mamba + Docker Qdrant server + Telegram/Discord Bot runtime。

## 快速部署

```bash
git lfs install
git clone https://github.com/iris-Neko/Nervos-Brain.git
cd Nervos-Brain
git checkout dev  # 当前交付/部署分支
git lfs pull

conda env create -f environment.yml
# 如果环境已存在，改用：
# conda env update -n nervos-brain -f environment.yml --prune
cp config.yaml.example config.yaml
```

如果机器安装了 `mamba` 或 `micromamba`，也可以把上面的 `conda` 换成对应命令。项目自带脚本会自动探测 `mamba`、`micromamba`、`conda`；需要指定时可设置 `ENV_RUNNER_BIN=/path/to/conda`。

编辑 `config.yaml` 或通过环境变量填写本地密钥。最低必填项是 LLM API 配置，以及要启动的平台 Bot token：

```bash
export TELEGRAM_BOT_TOKEN="<TELEGRAM_BOT_TOKEN>"
export DISCORD_BOT_TOKEN="<DISCORD_BOT_TOKEN>"
```

启动 Docker Qdrant 并从 SQLite archive 重建三套 collection：

```bash
bash bootstrap_qdrant_server.sh
```

启动 Telegram Bot：

```bash
bash restart_telegram_bot.sh
```

启动 Discord Bot：

```bash
conda run -n nervos-brain python scripts/run_discord_bot.py
```

更完整的 fresh clone 部署流程见 [docs/deployment.md](docs/deployment.md)。

## 文档入口

- [工程文档索引](docs/README.md)
- [交付说明](docs/handover.md)
- [新服务器部署](docs/deployment.md)
- [配置说明](docs/configuration.md)
- [检索数据与 Qdrant 重建](docs/retrieval-data.md)
- [运行与运维](docs/runtime-operations.md)
- [MCP 服务](docs/mcp.md)
- [测试与验收](docs/testing-and-acceptance.md)
- [故障排查](docs/troubleshooting.md)

## 系统组成

- `graph_engine`: full graph、prompt、模型路由、检索规划、回答生成和反思。
- `tool_runtime`: Telegram/Discord runtime、平台 adapter、工具执行、反馈和 MCP transport。
- `retrieval`: Qdrant 浅层向量库、SQLite archive 深层文本库、BM25/fuzzy/exact/vector 检索和多 backend 合并。
- `ingestion`: GitHub docs/code、Discourse/Nervos Talk、JSONL/web text 数据入库。
- `memory`: 用户/群/线程隔离的消息、事实和 AskUser checkpoint。
- `response_normalizer`: 引用规范化、Markdown 清理、Telegram/Discord 分段格式化。

## 检索数据

当前公开部署使用三套检索 backend：

```text
retrieval              -> nervos_docs
retrieval_forum_talk   -> nervos_talk_user_discussions
retrieval_github_code  -> nervos_github_code
```

SQLite archive DB 是可复现源数据，Docker Qdrant server 是运行时向量索引服务。`data/qdrant_server/` 是 Docker 运行目录，不提交；新机器 clone 后通过 `bootstrap_qdrant_server.sh` 从 archive DB 重建 Qdrant collection。可以简单理解为：archive DB 是可发布的资料源，Qdrant 是从资料源生成的搜索索引。

三套 archive DB 都由 Git LFS 管理。fresh clone 后必须运行：

```bash
git lfs pull
```

## 配置与安全边界

`config.yaml.example` 是模板，`config.yaml` 是本地私有配置，不提交。公开仓库不包含：

```text
LLM API key
Telegram / Discord Bot token
config.yaml
.env*
群聊记录
debug events
feedback.jsonl
memory DB
runtime logs
Docker Qdrant server 运行目录
```

配置里的路径应保持相对路径，运行时由 `src/nervos_brain/pathing.py` 解析。不要把本机绝对路径写进公开配置或文档。

## 常用命令

```bash
bash -n bootstrap_qdrant_server.sh restart_telegram_bot.sh
conda run -n nervos-brain python -m py_compile \
  scripts/migrate_qdrant_server_from_archive.py \
  scripts/run_talk_mcp_server.py \
  scripts/run_talk_forum_ingest.py
# 全量测试较重，需要时再运行：
# conda run -n nervos-brain pytest -q
```

更多测试和验收命令见 [docs/testing-and-acceptance.md](docs/testing-and-acceptance.md)。

交付接手清单见 [docs/handover.md](docs/handover.md)。
