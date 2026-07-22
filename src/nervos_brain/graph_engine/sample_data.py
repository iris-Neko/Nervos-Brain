# ============================================================
# sample_data.py —— M3-T2 GraphState 最小样例数据
# ============================================================
# 这个文件提供一个"假的但格式完全正确"的 GraphState 样例。
#
# 为什么需要样例数据？
#   在 M3 阶段，我们还没有真实的 Qdrant、MCP、用户消息。
#   但我们要验证 LangGraph 流程能不能跑通。
#   所以先造一组"假数据"，格式完全按照协议来，
#   这样就能测试"流水线本身"有没有问题，而不被外部服务拖住。
#
# 这里提供两个样例：
#   1. 信息充足的场景 → 应该直接回答
#   2. 缺参数的场景  → 应该反问用户
# ============================================================

import time

from nervos_brain.core_protocols import GraphState


def make_sufficient_state() -> GraphState:
    """
    构造一个"信息充足"的 GraphState 样例。

    场景：用户问 "CKB 的 Cell 模型是什么？"
    系统已经有足够的证据可以直接回答。
    """
    now_ms = int(time.time() * 1000)

    return GraphState(
        # -------- 基本信息 --------
        request_id="req_demo_001",
        user_message={
            "kind": "message",
            "ts_ms": now_ms,
            "message_id": "msg_demo_001",
            "context": {
                "platform": "discord",
                "user_id": "user_alice",
                "guild_id": "guild_nervos",
                "channel_id": "channel_dev",
            },
            "content": "CKB 的 Cell 模型是什么？",
            "platform_locale_hint": "zh-CN",
        },

        # -------- 记忆相关 --------
        user_memory_key={
            "platform": "discord",
            "user_id": "user_alice",
        },
        channel_memory_key={
            "platform": "discord",
            "guild_id": "guild_nervos",
            "channel_id": "channel_dev",
        },
        memory_pointers=[],   # 目前还没有记忆系统，所以为空
        memory_facts=[],      # 目前还没有事实卡片，所以为空

        # -------- 检索闭环 --------
        info_needs=[],        # Planner 还没分析，先为空
        evidence=[            # 假设已经有一条证据
            {
                "id": "ev_demo_001",
                "source": "qdrant",
                "title": "CKB Cell Model 介绍",
                "url": "https://docs.nervos.org/docs/basics/concepts/cell-model",
                "anchor": "section:overview",
                "snippet": (
                    "Cell 是 CKB 的基本数据存储单元。"
                    "每个 Cell 包含 capacity、lock script、type script 和 data 四个字段。"
                    "它类似于比特币的 UTXO，但功能更强大，可以存储任意数据。"
                ),
                "score": 0.95,
                "payload": {
                    "source": "docs",
                    "type": "doc",
                    "version": "latest",
                    "lang": "zh",
                },
                "hash": "sample_hash_001",
                "retrieved_ts_ms": now_ms,
            },
        ],
        conflicts=[],         # 没有证据冲突
        retry_count=0,        # 第一轮

        # -------- 控制开关 --------
        budget={
            "max_prompt_tokens": 4000,
            "max_evidence_chunks": 5,
            "max_memory_facts": 3,
            "max_tool_calls": 3,
        },
        route="graph",
        locale="zh-CN",
    )


def make_insufficient_state() -> GraphState:
    """
    构造一个"缺参数"的 GraphState 样例。

    场景：用户问 "Fiber SDK 怎么发交易？"
    但用户没说用的是哪个语言的 SDK，系统需要反问。
    """
    now_ms = int(time.time() * 1000)

    return GraphState(
        # -------- 基本信息 --------
        request_id="req_demo_002",
        user_message={
            "kind": "message",
            "ts_ms": now_ms,
            "message_id": "msg_demo_002",
            "context": {
                "platform": "telegram",
                "user_id": "user_bob",
            },
            "content": "Fiber SDK 怎么发交易？",
        },

        # -------- 记忆相关 --------
        user_memory_key={
            "platform": "telegram",
            "user_id": "user_bob",
        },
        memory_pointers=[],
        memory_facts=[],

        # -------- 检索闭环 --------
        info_needs=[
            {
                "kind": "missing_param",
                "question": "用户使用的是哪个语言的 Fiber SDK？(JavaScript / Rust / Go)",
                "required": True,
                "availability": "user_owned",
            },
        ],
        evidence=[],          # 还没开始查，因为缺参数
        conflicts=[],
        retry_count=0,

        # -------- 控制开关 --------
        budget={
            "max_prompt_tokens": 4000,
            "max_evidence_chunks": 5,
            "max_memory_facts": 3,
            "max_tool_calls": 3,
        },
        route="graph",
        locale="zh-CN",
    )
