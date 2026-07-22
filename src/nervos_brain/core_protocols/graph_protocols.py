# ============================================================
# graph_protocols.py —— LangGraph 图状态协议
# ============================================================
# 这个文件定义的是 LangGraph 工作流中所有节点共享的"大状态包"。
#
# 什么是 LangGraph？
#   LangGraph 是一个用图（Graph）来编排 AI 工作流的框架。
#   你可以把它想象成一条流水线：
#     Planner → Execute → Grader → (如果不够好) Replanner → Execute → ...
#   每个节点都是流水线上的一个工位。
#
# 什么是 GraphState？
#   所有工位共用的一张大表。
#   每个节点都可以读这张表、也可以往上面写。
#   比如：
#     - Planner 往表上写"我发现缺这些信息"（info_needs）
#     - Execute 往表上写"我查到了这些证据"（evidence）
#     - Grader 往表上写"这些证据有冲突"（conflicts）
#
# 什么是 TokenBudget？
#   本轮回答的资源预算。
#   防止系统无限制地调用工具、塞入太多证据、消耗太多 token。
#   就像给你一个固定的"零花钱"，花完了就得停。
# ============================================================

from typing import Any, List, Literal, NotRequired, TypedDict

from .memory_protocols import (
    ChannelMemoryKey,
    Fact,
    MemoryPointer,
    ThreadKey,
    UserMemoryKey,
)
from .message_protocols import MessageEnvelope
from .retrieval_protocols import Evidence, EvidenceConflict, InfoNeed, RetrievalPlan
from .turn_protocols import TurnContract


# ============================================================
# TokenBudget（Token 预算）
# ============================================================
# 为什么需要预算？
#   LLM 的 API 调用是按 token 收费的。
#   如果不设预算，一个复杂问题可能消耗几万 token，成本爆炸。
#   TokenBudget 就是告诉系统：
#   "这一轮最多用这么多资源，超了就停下来用已有的证据回答。"

class TokenBudget(TypedDict):
    # prompt 最多允许占用多少 token
    max_prompt_tokens: int

    # 最多允许注入多少条证据片段
    max_evidence_chunks: int

    # 最多允许注入多少条事实卡片
    max_memory_facts: int

    # 最多允许调用多少次工具
    max_tool_calls: int

    # （可选）预回答反思最多轮次
    max_reflection_rounds_pre: NotRequired[int]

    # （可选）后回答反思最多轮次
    max_reflection_rounds_post: NotRequired[int]

    # （可选）信息不足时最大检索跳数
    max_hops: NotRequired[int]

    # （可选）触发主动追问的不确定度阈值
    ask_user_uncertainty_threshold: NotRequired[float]


# ============================================================
# GraphState（图工作流共享状态）
# ============================================================
# 这是整个系统最核心的数据结构之一。
# LangGraph 里的每一个节点（Planner、Execute、Grader、Replanner、
# AnswerComposer）都会读写这个状态。
#
# 你可以把它想象成一个"任务大白板"：
#   - 左上角写着"这次任务的基本信息"（request_id、user_message）
#   - 中间写着"记忆相关的指针"（memory_pointers、memory_facts）
#   - 右边写着"检索进展"（info_needs、evidence、conflicts）
#   - 下面写着"资源预算和控制开关"（budget、route、locale）

class GraphState(TypedDict):
    # -------- 基本信息 --------

    # 本次请求的唯一 ID
    request_id: str

    # 用户发来的原始消息（MessageEnvelope 格式）
    user_message: MessageEnvelope

    # -------- 记忆相关 --------

    # 用户记忆钥匙（必填：每个请求都必须知道是哪个用户）
    user_memory_key: UserMemoryKey

    # 频道记忆钥匙（可选：私聊场景可能没有频道）
    channel_memory_key: NotRequired[ChannelMemoryKey]

    # 线程钥匙（可选：不是所有对话都在线程里）
    thread_key: NotRequired[ThreadKey]

    # 记忆指针列表：告诉后续节点"去哪里取相关记忆"
    memory_pointers: List[MemoryPointer]

    # 事实卡片列表：高频短小的信息，直接注入 prompt
    memory_facts: List[Fact]

    # -------- 检索闭环 --------

    # 信息缺口列表：Planner 分析出"还缺什么"
    info_needs: List[InfoNeed]

    # 检索计划（可选：第一轮 Planner 还没生成计划时为空）
    retrieval_plan: NotRequired[RetrievalPlan]

    # 已收集的证据列表
    evidence: List[Evidence]

    # 证据冲突列表
    conflicts: List[EvidenceConflict]

    # 当前是第几轮检索（用于限制"查了又查"的次数）
    retry_count: int

    # -------- 控制开关 --------

    # 本轮的资源预算
    budget: TokenBudget

    # 路由决策：走缓存还是走完整图流程
    # "cache" = 这个问题之前答过，直接返回缓存
    # "graph" = 需要走完整的检索 → 打分 → 回答流程
    route: Literal["cache", "graph"]

    # 用户的语言（例如 "zh-CN"、"en"）
    # 用于决定回答用什么语言
    locale: str

    # Platform-provided account preference; never a semantic language decision.
    platform_locale_hint: NotRequired[str]

    # Shared product policy instance injected at the protocol boundary.
    _product_policy: NotRequired[Any]

    # 当前用户回合的唯一语义契约
    turn_contract: NotRequired[TurnContract]

    # contract 解析出的最终回答语言
    response_locale: NotRequired[str]

    # 经过语义选择后允许下游使用的上下文
    selected_context: NotRequired[List[dict]]

    # Turn Interpreter / Compliance 的观测与预算字段
    turn_interpretation_passes: NotRequired[int]
    contract_repair_count: NotRequired[int]
    compliance_replacement_count: NotRequired[int]

    # 线程补参恢复后的“合并问题”（原问题 + 用户补充）
    # 未恢复时可不填，默认等于 user_message.content
    resolved_question: NotRequired[str]

    # -------- 通用反思（可选）--------

    # 当前反思阶段：pre_answer / post_answer
    reflection_stage: NotRequired[str]

    # 当前反思动作决策：
    # continue_retrieval / ask_user / revise_answer / accept_answer
    reflection_decision: NotRequired[str]

    # 反思理由
    reflection_reasoning: NotRequired[str]

    # 不确定度（0~1）
    uncertainty_score: NotRequired[float]

    # 反思提示（例如 next_query / revise_instructions / clarify_question）
    reflection_hints: NotRequired[dict]

    # 当前阶段反思轮次
    reflection_round: NotRequired[int]

    # 当前已执行检索 hop 数
    hop_count: NotRequired[int]
