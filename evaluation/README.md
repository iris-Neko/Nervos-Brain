# Evaluation Datasets

这里存放固定评测样例，用于 prompt、模型路由、检索策略、数据源和格式化改动后的回答质量回归。

## 文件划分

```text
ai_generated_cases.jsonl      # 人工设计 / AI 辅助生成的基准样例
human_collected_cases.jsonl   # 真实内测或人工收集的 bad case
```

`ai_generated_cases.jsonl` 主要用于覆盖已知产品能力，例如多轮补参、资料推荐、开发指导和排障。

`human_collected_cases.jsonl` 用于后续沉淀真实坏例子，例如 CCC 教程、TS/JS CKB 转账、Spore DOB、Fiber、xUDT、Nervos Talk 社区讨论等。这个文件可以为空；收集到样例后按 JSONL 一行一个 case 追加。

## Case 格式

```json
{
  "case_id": "human-ccc-transfer-001",
  "category": "bad_case",
  "title": "TS/JS user asks for CCC CKB transfer tutorial",
  "conversation": [
    {"role": "user", "content": "我是 TS/JS 开发者，想用 CCC 写 CKB 转账小应用，给我最简教程。"}
  ],
  "expected_signals": {
    "must_use_tools": ["qdrant_search"],
    "should_not": ["all_todo_skeleton", "go_only_tutorial"]
  },
  "success_criteria": [
    "主动检索 CCC / TS/JS 相关资料",
    "优先给现成库和可运行路径",
    "核心 SDK 调用不能全部写成 TODO 占位"
  ],
  "notes": "从真实内测 bad case 整理，去除用户隐私、群聊原文和 token。"
}
```

## 检索评测

回答质量样例（上面的 case）没有自动打分。检索本身有一套独立的离线评测：

```text
retrieval_queries.jsonl   # 59 条查询：qid、group（zh/en/id）、kind、query、来源
retrieval_pool.jsonl      # 评审池：每条查询的候选文档（标题 + 正文前 1200 字），顺序随机，不带任何配置信息
retrieval_qrels.jsonl     # 相关性标注：{qid, judgments: {"<库>::<anchor>": 0/1/2}}
reports/                  # 评测报告（Markdown + JSON）
```

查询来源：18 条取自上面两个 case 文件里的用户问题，其余是手写的典型查询——中文概念、英文概念、函数名、报错名和论坛讨论各若干。`group` 是语言/形态（中文 zh、英文 en、标识符/报错 id），`kind` 是细分类型。

### 标注方法（pooling）

1. 对 A/B/C/D 四种配置各跑一遍，取每条查询融合后的前 10 条，合并去重成候选池（平均每条查询 24 个候选，共 1,417 个）。候选顺序随机打乱，标注时看不到它来自哪个配置、排第几。
2. 标注者只看“标题 + 正文前 1200 字”，对“能否回答这条查询的信息需求”打 0/1/2：2 = 直接回答或就是被问对象的定义/说明/实现；1 = 有实质帮助但只是片段或相邻主题；0 = 无关或只是偶然出现关键词。标识符类查询：定义、核心实现、专门说明 = 2，有逻辑的调用/示例 = 1 或 2，仅 import 或无关配置里出现 = 0。
3. 标注由 6 个 LLM 子代理（Claude Sonnet）并行完成，每个分一部分查询。
4. 抽查：随机抽 12 条查询（约 20%，中文 5、英文 4、标识符 3）交给另一个独立的 LLM 子代理在不同的候选顺序下重新标注，不看原标注。290 个候选上，三档完全一致 92.4%，相关/不相关二分类一致 93.1%，Cohen's κ 0.86（三档）/ 0.86（二分类）。
5. 局限：标注者都是 LLM，系统性偏差（比如对代码文件、论坛短回复的口径）两遍抽查发现不了；pooling 只覆盖四种配置各自前 10 的结果，所有配置都没召回的相关文档不在标注范围内，所以 Recall 是“相对于池中已知相关文档”的值，绝对数偏高，用来比较配置而不是估计真实召回率。没有做人工抽查，欢迎补。

### 配置与指标

| 配置 | 说明 |
|---|---|
| A | 现状：64 维哈希向量 + BM25 + exact + fuzzy |
| B | 关掉向量路：BM25 + exact + fuzzy |
| C | bge-m3 真实向量 + BM25 + exact + fuzzy（RRF 融合） |
| D | 只用 bge-m3 向量路 |

指标：Recall@5/10（池中相关文档，等级 ≥1，被前 K 条找回的比例）、MRR@10、nDCG@10（增益 2^等级−1）、Precision@5、Noise@5（前 5 条里不相关的比例），按中文 / 英文 / 标识符及查询类型分组，另报平均和 P95 耗时。LLM 生成正则的那一路依赖 LLM 调用，评测里不开。

### 复现

```bash
# 1. 重建两套评测用的嵌入式 Qdrant 目录（真实向量 / 哈希对照），不碰线上 server 和仓库里提交的 data/qdrant_*
python scripts/migrate_qdrant_server_from_archive.py --public-default-backends --local \
  --qdrant-root data/qdrant_eval --recreate \
  --embedding-backend local --embedding-model BAAI/bge-m3 --vector-size 1024
python scripts/migrate_qdrant_server_from_archive.py --public-default-backends --local \
  --qdrant-root data/qdrant_eval_hash --recreate --embedding-backend hash --vector-size 64

# 2. （只在改了查询集或要重标时）生成评审池，然后标注 retrieval_pool.jsonl，写 retrieval_qrels.jsonl
python scripts/eval_retrieval.py pool

# 3. 打分并生成报告
python scripts/eval_retrieval.py run --report evaluation/reports/retrieval_eval_<日期>
```

`run` 会覆盖同名报告；仓库里的报告在自动生成的表格后面追加了人工撰写的“解读”一节，重跑时请换一个 `--report` 名字。`data/qdrant_eval*/` 已在 `.gitignore` 里。哈希对照库必须重建而不是直接用仓库里提交的 `data/qdrant_*`：后者是早期快照，条数比 archive DB 少（例如 docs 1,240 对 1,573），会让 A 配置吃亏。

## 安全边界

不要写入真实 token、私钥、API key、群聊原文、debug log、用户隐私或无法公开的客户信息。真实对话应做脱敏和概括，只保留评测需要的技术诉求、失败模式和验收标准。
