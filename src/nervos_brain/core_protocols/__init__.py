# ============================================================
# core_protocols 包的统一导出入口
# ============================================================
# 作用：
#   让外部模块可以直接写：
#     from nervos_brain.core_protocols import MessageEnvelope, Evidence
#   而不用记住每个类型藏在哪个子文件里。
#
# 用法：
#   后续每新增一个协议文件，就在这里加一行 import。
# ============================================================

# ---------- common（公共小类型）----------
from .common import MessageKind
from .common import Platform
from .common import RenderMode

# ---------- message（消息协议）----------
from .message_protocols import Attachment
from .message_protocols import AttachmentKind
from .message_protocols import ConversationContext
from .message_protocols import MessageBusEnvelope
from .message_protocols import MessageBusPriority
from .message_protocols import MessageEnvelope
from .message_protocols import OutboundMessage
from .message_protocols import OutboundMessageSegment
from .message_protocols import PlatformCapabilities

# ---------- memory（记忆协议）----------
from .memory_protocols import ChannelMemoryKey
from .memory_protocols import Fact
from .memory_protocols import MemoryPointer
from .memory_protocols import MemoryPointerKind
from .memory_protocols import ThreadKey
from .memory_protocols import UserMemoryKey

# ---------- retrieval（检索协议）----------
from .retrieval_protocols import Evidence
from .retrieval_protocols import EvidenceConflict
from .retrieval_protocols import EvidenceSource
from .retrieval_protocols import InfoNeed
from .retrieval_protocols import InfoNeedKind
from .retrieval_protocols import PayloadFilter
from .retrieval_protocols import RetrievalPlan
from .retrieval_protocols import RetrievalStep

# ---------- tool（工具调用协议）----------
from .tool_protocols import ErrorCode
from .tool_protocols import ToolCallRequest
from .tool_protocols import ToolCallResult
from .tool_protocols import ToolCallStatus
from .tool_protocols import ToolCallWarning
from .tool_protocols import ToolError
from .tool_protocols import ToolName

# ---------- response（回复协议）----------
from .response_protocols import AssistantResponse
from .response_protocols import Citation

# ---------- graph（图状态协议）----------
from .graph_protocols import GraphState
from .graph_protocols import TokenBudget

# ---------- turn（single-turn semantic contract）----------
from .turn_protocols import (
    TurnConfidence,
    TurnContext,
    TurnContract,
    TurnInfoNeed,
    TurnLanguage,
    TurnPolicy,
)
from .turn_protocols import TurnInfoNeed
from .turn_protocols import TurnLanguage
from .turn_protocols import is_turn_contract
