# ============================================================
# retrieval_protocols.py —— 检索系统相关协议
# ============================================================
# 这个文件定义的是"主动检索闭环"相关的数据结构。
#
# 什么是主动检索？
#   传统 RAG：用户问一句 → 系统查一次 → 直接回答。
#   主动检索：用户问一句 → 系统先分析"我缺什么信息" → 制定检索计划
#            → 执行检索 → 检查证据够不够 → 不够就再查 → 够了才回答。
#
# 这个文件里的类型就是这个流程中每一步传递的"表格格式"：
#   1. InfoNeed     = "我还缺什么信息"
#   2. PayloadFilter = "按什么条件筛选"
#   3. RetrievalStep = "检索计划中的一个步骤"
#   4. RetrievalPlan = "完整的检索计划"
#   5. Evidence      = "找到的一条证据"
#   6. EvidenceConflict = "两条证据互相矛盾"
# ============================================================

from __future__ import annotations

from typing import Dict, List, Literal, NotRequired, TypedDict


# ============================================================
# InfoNeed（信息缺口）
# ============================================================
# 什么是信息缺口？
#   系统的 Planner 节点会先分析用户的问题，判断：
#   "要回答这个问题，我还缺哪些信息？"
#   每一个缺口就是一个 InfoNeed。
#
# 举例：
#   用户问："怎么用 Fiber SDK 发交易？"
#   系统可能识别出两个缺口：
#     1. kind="version_unknown", question="用户用的是哪个版本的 Fiber？"
#     2. kind="concept_gap", question="Fiber 发交易的 API 是什么？"

# 信息缺口的种类
InfoNeedKind = Literal[
    "missing_param",         # 用户缺少必要参数（比如没说用什么语言）
    "version_unknown",       # 版本不明（不知道用的哪个版本）
    "concept_gap",           # 概念/背景缺口（需要查文档补知识）
    "error_trace",           # 缺少报错日志/调用栈
    "latest_spec",           # 需要最新的 RFC/规范
    "historical_consensus",  # 需要论坛/社区的历史共识
]


class InfoNeed(TypedDict):
    # 缺口种类
    kind: InfoNeedKind

    # 给 Planner 看的问题描述："要解决什么"
    question: str

    # 是否必须解决：True = 不解决就不能回答
    required: bool

    # 缺口来自公开资料还是只能由用户提供的私有/现场信息
    availability: NotRequired[Literal["public", "user_owned"]]

    # 说明这个缺口如何直接完成核心交付物
    purpose: NotRequired[str]

    # 可选的提示信息，帮助检索更精准
    # 例如 {"sdk": "js", "version": ">=0.3"}
    hints: NotRequired[Dict[str, str]]


# ============================================================
# PayloadFilter（元数据过滤条件）
# ============================================================
# 什么是 PayloadFilter？
#   在 Qdrant 向量数据库里，每条文档除了正文向量，还带有元数据。
#   比如：来源是 "rfcs"、类型是 "doc"、版本是 ">=0.3"。
#   PayloadFilter 就是告诉检索引擎：
#   "我只要满足这些条件的结果，其它的不要给我。"
#
# 举例：
#   {"source": "rfcs", "lang": "en", "version": ">=0.3"}
#   意思是：只要来自 RFC 文档的、英文的、版本 >= 0.3 的结果。

class PayloadFilter(TypedDict):
    # 来源：例如 "rfcs"、"fiber"、"ccc"、"talk"
    source: NotRequired[str]

    # 类型：例如 "doc"（文档）、"code"（代码）、"forum"（论坛帖子）
    type: NotRequired[str]

    # 版本要求：例如 ">=0.3"
    version: NotRequired[str]

    # 语言：例如 "zh"（中文）、"en"（英文）
    lang: NotRequired[str]

    # 主题：例如 "fiber"、"cell-model"
    topic: NotRequired[str]


# ============================================================
# RetrievalStep（检索步骤）
# ============================================================
# 什么是检索步骤？
#   检索计划可能包含多个步骤，比如：
#     第 1 步：在 Qdrant 里搜 "Fiber 发交易 API"
#     第 2 步：在 Nervos Talk 里搜 "Fiber transaction 最新讨论"
#     第 3 步：在 GitHub 里搜 "Fiber SDK 代码示例"
#   每一步就是一个 RetrievalStep。

# 可用的工具名称（白名单）
# 只有这四个工具允许被调用，其它的一律拒绝
ToolName = Literal[
    "qdrant_search",     # 在向量数据库中搜索
    "discourse_query",   # 在 Nervos Talk 论坛中搜索
    "github_search",     # 在 GitHub 仓库中搜索
    "memory_fetch",      # 从记忆系统中取数据
]


class RetrievalStep(TypedDict):
    # 步骤 ID，例如 "step_1"、"step_2"
    step_id: str

    # 用哪个工具执行这个步骤
    tool: ToolName

    # 搜索用的查询语句
    query: str

    # 可选：元数据过滤条件
    filters: NotRequired[PayloadFilter]

    # 可选：返回前 k 条结果（默认一般是 5）
    top_k: NotRequired[int]

    # 可选：时间范围，格式 "YYYY-MM..YYYY-MM"
    # 例如 "2024-01..2026-12"
    time_range: NotRequired[str]

    # 可选：依赖的其他步骤 ID 列表
    # 用于表示"先做完 step_1 才能做 step_2"这种顺序关系
    depends_on: NotRequired[List[str]]


# ============================================================
# RetrievalPlan（检索计划）
# ============================================================
# 什么是检索计划？
#   把上面的多个 RetrievalStep 组织起来，形成一个完整的检索计划。
#   包括：
#     - 为什么要这样查（rationale）
#     - 哪些步骤可以并行执行（parallel_groups）
#     - 总共允许消耗多少资源（budget）

class RetrievalPlan(TypedDict):
    # 计划 ID
    plan_id: str

    # 检索理由：Planner 解释"为什么要这样查"
    rationale: str

    # 检索步骤列表
    steps: List[RetrievalStep]

    # 并行分组：哪些步骤可以同时执行
    # 例如 [["step_1", "step_2"], ["step_3"]]
    # 意思是 step_1 和 step_2 可以同时跑，跑完再跑 step_3
    parallel_groups: List[List[str]]

    # 本次检索的资源预算
    # 这里引用的是 graph_protocols.py 里的 TokenBudget
    # 但为了避免循环导入，这里先用 Dict 表示
    # 后续在 graph_protocols.py 里会正式定义 TokenBudget
    budget: Dict[str, int]


# ============================================================
# Evidence（证据）
# ============================================================
# 什么是证据？
#   不管是从 Qdrant、Nervos Talk、GitHub 还是记忆里查到的东西，
#   都必须统一转换成 Evidence 格式，才能进入后续的"打分→回答"流程。
#
# 为什么要统一？
#   因为后续的 Grader（打分员）和 AnswerComposer（回答组装器）
#   不关心"这条结果原来是从哪个工具查出来的"，
#   它们只关心"这条证据是否回答了用户的问题"。
#
# 强制规则：
#   每条 Evidence 必须有来源（source）、链接（url）、锚点（anchor）。
#   缺任何一个都不允许进入 AnswerComposer。

# 证据来源
EvidenceSource = Literal[
    "qdrant",      # 来自向量数据库
    "discourse",   # 来自 Nervos Talk 论坛
    "github",      # 来自 GitHub
    "memory",      # 来自记忆系统
]


class Evidence(TypedDict):
    # 证据的唯一 ID
    id: str

    # 来源
    source: EvidenceSource

    # 标题（文档标题 / 帖子标题 / 文件名）
    title: str

    # 链接（可以点击打开原文的 URL）
    url: str

    # 锚点：精确定位到原文的哪个位置
    # 例如：
    #   代码文件："L42-L58"（第 42~58 行）
    #   论坛帖子："post#47"（第 47 楼）
    #   文档章节："section:4.1"
    anchor: str

    # 摘录片段（不超过 1200 字符，超出会被截断）
    snippet: str

    # 相关度评分（0~1，越高越相关）
    score: float

    # 元数据字典：source / type / version / lang / url / anchor
    # 缺失的字段必须补齐为 "unknown"
    payload: Dict[str, str]

    # 稳定哈希值：用于去重
    # 计算方式：sha256(canonical_json({url, anchor, snippet_norm, payload_norm}))
    hash: str

    # 检索到这条证据的时间（毫秒时间戳）
    retrieved_ts_ms: int


# ============================================================
# EvidenceConflict（证据冲突）
# ============================================================
# 什么是证据冲突？
#   有时候两条证据会互相矛盾。
#   比如：
#     证据 A 说："Fiber SDK 最新版本是 0.2"
#     证据 B 说："Fiber SDK 最新版本是 0.3"
#   系统必须记录这种冲突，让后续的 Grader 或 Replanner 来处理。

class EvidenceConflict(TypedDict):
    # 第一条证据的 ID
    a_id: str

    # 第二条证据的 ID
    b_id: str

    # 冲突原因
    # 例如 "version_mismatch"（版本不一致）
    # 例如 "contradiction"（内容直接矛盾）
    reason: str
