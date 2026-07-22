# ============================================================
# nodes.py —— M3-T3/T4/T5 LangGraph 最小节点
# ============================================================
# 这个文件定义了 LangGraph 工作流中的三个最小节点（Mock 版本）。
#
# 什么是节点？
#   LangGraph 的工作流就像一条流水线，每个节点是流水线上的一个工位。
#   每个节点做一件事，然后把结果写回共享状态（GraphState）。
#
# M3 阶段的节点全是"Mock"（假的）：
#   - 不会真的调用 LLM
#   - 不会真的去 Qdrant 检索
#   - 只是按照固定规则做判断和返回
#
# 为什么先用 Mock？
#   因为我们现在要验证的是"流水线本身能不能跑通"，
#   不是"LLM 回答得好不好"。
#   先把管道接好，再换真零件。
#
# 这个文件里有三个节点：
#   1. turn_interpreter  —— 根据已解析的缺口选择下一步
#   2. ask_user           —— 反问用户补参数
#   3. answer_composer    —— 组装最终回答
# ============================================================

from nervos_brain.core_protocols import (
    AssistantResponse,
    GraphState,
)


def turn_interpreter(state: GraphState) -> dict:
    """
    M3-T3: Turn Interpreter 的最小示例节点。

    这个节点的职责：
      消费已经由上游解释得到的 info_needs 列表，
      判断最小示例图下一步是否需要用户输入。

    判断逻辑（Mock 版本很简单）：
      - 如果 info_needs 里有 required=True 且 availability=user_owned 的缺口 → 需要用户输入
      - 如果没有这类缺口 → 继续回答

    返回：
      一个字典，LangGraph 会把它合并到 GraphState 里。
      这里我们返回一个自定义字段 _route_decision，
      用于后续的条件路由判断。

    注意：
      节点函数的参数必须是 GraphState（或其子集），
      返回值必须是一个字典（LangGraph 会自动合并到状态里）。
    """
    info_needs = state.get("info_needs", [])

    # 检查是否有必须解决的信息缺口
    has_required_gap = any(
        need.get("required", False)
        and need.get("availability", "public") == "user_owned"
        for need in info_needs
    )

    if has_required_gap:
        # 信息不够，需要反问用户
        print("[TurnInterpreter] required user-owned information is missing")
        return {"_route_decision": "ask_user"}
    else:
        print("[TurnInterpreter] no required user-owned information is missing")
        return {"_route_decision": "answer"}


def ask_user(state: GraphState) -> dict:
    """
    M3-T4: 反问用户节点（Mock 版本）

    这个节点的职责：
      当系统发现缺少必要参数时，生成一个"反问"回答，
      告诉用户"我需要你补充什么信息"。

    Mock 逻辑：
      从 info_needs 里找到第一个 required=True 的缺口，
      把它的 question 字段拿出来，包装成 AssistantResponse。
    """
    info_needs = state.get("info_needs", [])
    request_id = state.get("request_id", "unknown")

    # 找到第一个必须解决的缺口
    question = "请问您能提供更多信息吗？"
    for need in info_needs:
        if need.get("required", False):
            question = need.get("question", question)
            break

    print(f"[AskUser] 反问用户: {question}")

    # 构造一个"反问"类型的 AssistantResponse
    response: AssistantResponse = {
        "request_id": request_id,
        "text": question,
        "citations": [],
        "need_user_input": True,
        "ask_user_question": question,
    }

    return {"_final_response": response}


def answer_composer(state: GraphState) -> dict:
    """
    M3-T5: 回答组装节点（Mock 版本）

    这个节点的职责：
      把已有的 Evidence 组装成最终回答。

    Mock 逻辑：
      1. 从 evidence 列表里取出所有证据的 snippet
      2. 拼成一段回答文本
      3. 为每条证据生成一个 Citation
      4. 包装成 AssistantResponse

    注意：
      真实版本会调用 LLM 来生成自然语言回答。
      Mock 版本只是简单拼接，用于验证流程能否跑通。
    """
    evidence_list = state.get("evidence", [])
    request_id = state.get("request_id", "unknown")

    if not evidence_list:
        # 没有证据，返回一个"我不知道"的回答
        print("[AnswerComposer] 没有找到相关证据")
        response: AssistantResponse = {
            "request_id": request_id,
            "text": "抱歉，我没有找到相关信息。",
            "citations": [],
        }
        return {"_final_response": response}

    # 拼接回答文本：把每条证据的 snippet 编号后拼起来
    text_parts = []
    citations = []

    for i, ev in enumerate(evidence_list, start=1):
        label = f"[{i}]"
        snippet = ev.get("snippet", "")
        title = ev.get("title", "未知来源")

        # 在回答里引用这条证据
        text_parts.append(f"{snippet} {label}")

        # 生成对应的 Citation
        citations.append({
            "label": label,
            "url": ev.get("url", ""),
            "anchor": ev.get("anchor", ""),
            "title": title,
        })

    # 拼成最终文本
    final_text = "\n\n".join(text_parts)

    print(f"[AnswerComposer] 组装了 {len(citations)} 条引用的回答")

    response: AssistantResponse = {
        "request_id": request_id,
        "text": final_text,
        "citations": citations,
    }

    return {"_final_response": response}
