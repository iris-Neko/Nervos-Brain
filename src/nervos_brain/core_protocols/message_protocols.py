from typing import Any, List, Literal, NotRequired, TypedDict

from .common import MessageKind
from .common import Platform
from .common import RenderMode


# 对话上下文：
# 描述“这条消息来自哪里、是谁发的、处在哪个频道/线程里”
class ConversationContext(TypedDict):
    # 平台：Discord 或 Telegram
    platform: Platform

    # 发消息的用户 ID
    user_id: str

    # 服务器 / 群组 ID
    # 有些平台或场景可能没有，所以用 NotRequired
    guild_id: NotRequired[str]

    # 频道 / 聊天 ID
    channel_id: NotRequired[str]

    # 线程 ID
    # 用于 AskUser 挂起恢复这类场景
    thread_id: NotRequired[str]


# 附件类型：
# 这里只在 message_protocols.py 里使用，所以先放本文件即可
AttachmentKind = Literal["image", "file", "link"]


# 单个附件的数据结构
class Attachment(TypedDict):
    # 附件类型：图片 / 文件 / 链接
    kind: AttachmentKind

    # 附件地址
    url: str

    # 附件名字，可选
    name: NotRequired[str]


# 入站消息统一结构：
# 不管消息来自 Discord 还是 Telegram，进入系统后都整理成这个格式
class MessageEnvelope(TypedDict):
    # 消息种类：普通消息 or 命令消息
    kind: MessageKind

    # 毫秒时间戳
    ts_ms: int

    # 当前消息 ID
    message_id: str

    # 对话上下文
    context: ConversationContext

    # 消息正文
    content: str

    # 如果这是对某条消息的回复，就记录被回复消息的 ID
    reply_to_message_id: NotRequired[str]

    # 被回复消息的简短内容快照，用于 Telegram/Discord follow-up 消歧
    reply_to_content: NotRequired[str]

    # 被回复消息的发送者角色，例如 user / assistant / bot
    reply_to_role: NotRequired[str]

    # 附件列表，可选
    attachments: NotRequired[List[Attachment]]

    # 如果是命令消息，例如 /tldr，就把命令名放这里
    command: NotRequired[str]

    # 命令参数，例如链接或主题词
    command_args: NotRequired[str]

    # Platform account preference only; never the semantic response locale.
    platform_locale_hint: NotRequired[str]


# 消息在内部总线中的优先级
MessageBusPriority = Literal["high", "normal", "low"]


# 消息进入 MessageBus 后的外层包装
class MessageBusEnvelope(TypedDict):
    # 本次请求的唯一 ID
    request_id: str

    # 去重键：避免重复处理同一条消息
    dedup_key: str

    # 接收到消息的时间
    received_ts_ms: int

    # 当前是第几次尝试处理
    attempt: int

    # 优先级
    priority: MessageBusPriority

    # 真正的消息内容
    message: MessageEnvelope


# 平台能力说明：
# 告诉系统“这个平台最多发多长、支不支持流式输出、支不支持评分”
class PlatformCapabilities(TypedDict):
    # 平台名
    platform: Platform

    # 当前发送模式
    render_mode: RenderMode

    # 单段最大字符数
    max_chars_per_segment: int

    # 最多允许发几段
    max_segments: int

    # 是否支持流式输出
    supports_streaming: bool

    # 是否支持内联评分按钮
    supports_inline_csat: bool


# 单个出站消息分段
class OutboundMessageSegment(TypedDict):
    # 分段 ID
    segment_id: str

    # 第几段，从 0 或 1 开始都可以，但要统一
    index: int

    # 这一段要发出去的正文
    text: str

    # 这一段的字符数
    char_count: int

    # 这一段中出现了哪些引用编号
    citation_labels: List[str]

    # 平台特定的发送实体，例如 Telegram MessageEntity
    entities: NotRequired[List[dict[str, Any]]]

    # 如果为 false，即使 render_mode=markdown 也不要使用 parse_mode。
    parse_mode_enabled: NotRequired[bool]


# 最终发往平台的消息结构
class OutboundMessage(TypedDict):
    # 请求 ID
    request_id: str

    # 发给哪个上下文
    context: ConversationContext

    # 如果要回复某条消息，就填它
    reply_to_message_id: NotRequired[str]

    # 拆分后的消息段列表
    segments: List[OutboundMessageSegment]

    # 当前渲染模式
    render_mode: RenderMode

    # 是否在结尾附带评分按钮
    append_csat: bool
