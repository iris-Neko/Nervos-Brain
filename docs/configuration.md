# 配置说明

正式运行读取 `config.yaml`。`config.yaml.example` 只是模板，不作为 runtime fallback。部署时应复制模板并填写本地配置：

```bash
cp config.yaml.example config.yaml
```

## 路径规则

配置里的路径应保持相对路径，例如：

```yaml
logging:
  log_dir: "data/logs"
telegram_bot:
  memory_db: "data/telegram_bot/memory.db"
```

运行时由 `src/nervos_brain/pathing.py` 统一解析。不要把本机绝对路径写入公开配置或文档。

## LLM 配置

`llm` 是默认模型配置，`llm_profiles` 是 graph 节点动态路由使用的模型档位。

常见必填项：

```yaml
llm:
  model: "openai/gpt-5.4"
  api_key: "<LLM_API_KEY>"
  api_base: "https://your-openai-compatible-endpoint/v1"
```

当前业务档位包括：

```text
router
low
mini_high
medium
high
```

`mini_high` 是低成本深思考档，适合 InfoGap、检索规划和轻量反思。`medium` / `high` 用于更复杂的最终回答、代码/协议/架构和一致性检查。

## Retrieval 配置

主要有三套 backend：

```text
retrieval              # docs / official docs
retrieval_forum_talk   # Nervos Talk forum
retrieval_github_code  # GitHub source code
```

`retrieval_backends` 决定 runtime 是否使用多库统一检索。默认应包含 docs、forum 和 github_code 三库，使 `qdrant_search` 能一次查询全部 configured backends。

推荐部署使用 Qdrant server：

```yaml
retrieval:
  qdrant_url: "http://127.0.0.1:6333"
```

如果清空 `qdrant_url`，系统会回退到 `qdrant_path` 本地目录模式。线上 Telegram/Discord 多进程部署不建议使用本地目录模式。

### Embedding（向量路）

入库、重建 collection 和查询三处都走 `retrieval/embedding.py` 里同一个后端，在 `retrieval` 段配置，`retrieval_forum_talk` / `retrieval_github_code` 继承它的全部设置（三个库必须用同一个模型，各段只覆盖路径和 collection 名）：

```yaml
retrieval:
  embedding_backend: local          # local | api | hash | none
  embedding_model: "BAAI/bge-m3"    # local: sentence-transformers 模型名；api: litellm 模型名
  vector_size: 1024                 # 必须等于模型输出维度，不一致会直接报错
  embedding_batch_size: 64
  embedding_normalize: true         # L2 归一化，Qdrant 用 Cosine
  # embedding_query_prefix: "query: "     # e5 这类模型需要
  # embedding_doc_prefix: "passage: "
  # embedding_device: "cuda"        # local 后端；空 = 自动
  # embedding_api_base / embedding_api_key_env: api 后端；留空则复用 llm 段
  # embedding_cache_db: "data/embedding_cache/embeddings.db"  # 文档向量缓存，空 = 不缓存
  # enable_vector: true             # false = 关闭向量路，其余四路照常
```

| 后端 | 说明 |
|---|---|
| `local` | 本机 sentence-transformers。可选依赖：`pip install -e ".[local-embedding]"`，GPU 需要装对应的 torch。bge-m3（fp16，batch 64）在 RTX 3080 上编码 4 万条约 2.3 分钟，连同本地 Qdrant 写入重建三个库总共约 5 分钟。 |
| `api` | OpenAI 兼容的 `/embeddings`（`litellm.embedding`，批量调用，带重试）。网关需要提供 `/embeddings` 接口。 |
| `hash` | SHA-256 伪向量，**只给测试用**，没有任何语义。pytest 之外它会让向量路自动关闭（可用 `embedding_allow_hash: true` 或环境变量 `NERVOS_ALLOW_HASH_EMBEDDING=1` 强行打开，只用于评测对照）。 |
| `none` | 不建语义向量，向量路关闭，其余四路照常。 |

旧开关 `use_real_embedding` 仍然可读：`embedding_backend` 为空时，`true` 按 `api`、`false` 按 `hash` 处理。因为旧默认值是 `false`，没写新配置的老部署升级后向量路会被自动关闭（日志里有一条 error），而不是继续把哈希向量混进 RRF。

启动时每个库都会检查 Qdrant collection 的向量维度和 payload 里的 `embedding_model` 标签与当前后端是否一致，不一致就在日志里报 `vector path DISABLED ... rebuild it with migrate_qdrant_server_from_archive.py --recreate` 并关闭这个库的向量路，不会再静默返回空结果。换模型或维度之后必须重建 collection（见 [retrieval-data.md](retrieval-data.md)）。

## Telegram 配置

`telegram_bot` 控制轮询、过滤、memory、feedback、debug 和耗时预算。

常见项：

```yaml
telegram_bot:
  offset_file: "data/telegram_bot/offset.txt"
  memory_db: "data/telegram_bot/memory.db"
  feedback_file: "data/telegram_bot/feedback.jsonl"
  debug_log_file: "data/telegram_bot/debug_events.jsonl"
  allowed_chat_ids: []
  append_csat: true
```

`TELEGRAM_BOT_TOKEN` 可以通过环境变量提供。

## Discord 配置

`discord_bot` 控制 Discord runtime、memory、feedback、debug、allowed guild/channel 和并发。

常见项：

```yaml
discord_bot:
  memory_db: "data/discord_bot/memory.db"
  feedback_file: "data/discord_bot/feedback.jsonl"
  debug_log_file: "data/discord_bot/debug_events.jsonl"
  mention_only_in_guild: true
  allowed_channel_ids: []
  allowed_guild_ids: []
```

`DISCORD_BOT_TOKEN` 可以通过环境变量提供。

## Logging 配置

`logging` 控制控制台和文件日志：

```yaml
logging:
  level: "INFO"
  third_party_level: "WARNING"
  log_dir: "data/logs"
  file: true
```

`data/logs/` 是 runtime 私有数据，不提交。

## 不应提交的配置

不要提交：

```text
config.yaml
.env*
LLM API key
Telegram Bot token
Discord Bot token
Qdrant API key
runtime logs
memory DB
feedback/debug 文件
```
