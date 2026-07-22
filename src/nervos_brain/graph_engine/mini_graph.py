# ============================================================
# mini_graph.py —— M3-T6 用 LangGraph 串起最小流程
# ============================================================
# 这个文件把三个节点用 LangGraph 串成一个最小工作流。
#
# 工作流长这样：
#
#   开始
#    │
#    ▼
#   Turn Interpreter（消费已解析的任务契约）
#    │
#    ├── 信息不够 ──► AskUser（反问用户）──► 结束
#    │
#    └── 信息足够 ──► AnswerComposer（组装回答）──► 结束
#
# 这就是最小版的"主动检索闭环"雏形：
#   先判断 → 再决定走哪条路 → 最后输出
#
# 什么是 StateGraph？
#   LangGraph 的核心类。你可以把它想象成"流水线图纸"：
#   - add_node() = 在图纸上画一个工位
#   - add_edge() = 在两个工位之间连一条线
#   - add_conditional_edges() = 在工位后面画一个分叉路口
#   - compile() = 把图纸变成可以运行的流水线
#
# 什么是 Annotated[list, operator.add]？
#   LangGraph 的状态合并规则。
#   普通字段（str、int）= 后写覆盖前写。
#   列表字段如果用 Annotated[list, operator.add]
#     = 后写的内容会 append 到列表末尾，而不是覆盖。
#   我们这里的 evidence、info_needs 等列表字段暂时不需要
#   追加合并，所以先用最简单的方式。
# ============================================================

from typing import Annotated, Any, List

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from .nodes import answer_composer, ask_user, turn_interpreter


# ============================================================
# MiniGraphState —— 最小图状态
# ============================================================
# 为什么不直接用 core_protocols 里的 GraphState？
#   因为 LangGraph 的 StateGraph 要求状态类型的每个字段
#   都必须有默认值或 Annotated 标注来告诉它"怎么合并"。
#   而 core_protocols 里的 GraphState 是纯协议定义，
#   不带 LangGraph 特有的合并注解。
#
#   所以我们在这里定义一个"运行时用的最小状态"，
#   它是 GraphState 的一个简化版本，专门给 LangGraph 用。
#   后续 Milestone 7 会把它升级成完整版。

class MiniGraphState(TypedDict, total=False):
    """
    LangGraph 运行时用的最小状态。

    total=False 的意思是：所有字段都是可选的。
    这样 LangGraph 在合并节点返回值时不会因为缺字段报错。
    """
    # 请求 ID
    request_id: str

    # 用户消息（简化版，直接用 dict）
    user_message: dict

    # 用户记忆钥匙
    user_memory_key: dict

    # 频道记忆钥匙
    channel_memory_key: dict

    # 记忆指针列表
    memory_pointers: list

    # 事实卡片列表
    memory_facts: list

    # 信息缺口列表
    info_needs: list

    # 已收集的证据列表
    evidence: list

    # 证据冲突列表
    conflicts: list

    # 重试次数
    retry_count: int

    # Token 预算
    budget: dict

    # 路由决策
    route: str

    # 用户语言
    locale: str

    # ---- 以下是节点之间传递的内部字段 ----

    # Turn Interpreter 的路由判断结果
    _route_decision: str

    # 最终回答（由 AskUser 或 AnswerComposer 写入）
    _final_response: dict


# ============================================================
# 路由函数
# ============================================================

def route_after_assessment(state: MiniGraphState) -> str:
    """
    条件路由：根据 Turn Interpreter 的判断结果决定走哪条路。

    返回值是下一个节点的名字（字符串）。
    """
    decision = state.get("_route_decision", "answer")

    if decision == "ask_user":
        return "ask_user"
    else:
        return "answer_composer"


# ============================================================
# 构建最小图
# ============================================================

def build_mini_graph() -> Any:
    """
    构建并编译最小 LangGraph 工作流。

    返回：
        一个编译好的 LangGraph 可运行对象。
        可以直接用 graph.invoke(state) 来运行。
    """
    # 第 1 步：创建一个 StateGraph，指定状态类型
    graph = StateGraph(MiniGraphState)

    # 第 2 步：添加节点（流水线上的工位）
    graph.add_node("turn_interpreter", turn_interpreter)
    graph.add_node("ask_user", ask_user)
    graph.add_node("answer_composer", answer_composer)

    # 第 3 步：设置入口（流水线从哪里开始）
    graph.set_entry_point("turn_interpreter")

    # 第 4 步：添加条件边（分叉路口）
    # Turn Interpreter 执行完后，根据判断结果走不同的路
    graph.add_conditional_edges(
        "turn_interpreter",            # 从哪个节点出发
        route_after_assessment,        # 用哪个函数来决定走哪条路
        {
            "ask_user": "ask_user",              # 如果函数返回 "ask_user" → 去 ask_user 节点
            "answer_composer": "answer_composer", # 如果函数返回 "answer_composer" → 去 answer_composer 节点
        },
    )

    # 第 5 步：添加终止边（这两个节点执行完就结束）
    graph.add_edge("ask_user", END)
    graph.add_edge("answer_composer", END)

    # 第 6 步：编译图（把图纸变成可运行的流水线）
    compiled = graph.compile()

    return compiled
