# Turn Contract 通用化改造执行规格

> 文档状态：已实现
> 适用范围：共享 Graph、Telegram、Discord、Prompt、上下文、语言、身份、路由、追问、检索和回答质检
> 目标读者：负责直接修改代码、补测试、提交和部署的实现模型

## 0. 执行要求

这是一份完整实现规格，不是讨论稿。执行者必须完成代码、测试、提交和部署验证，不能只改 Prompt，也不能只替换语言正则。

实现时遵守以下约束：

1. 保留工作区中与本任务无关的现有修改，不回滚、不覆盖、不顺手重构。
2. 最终运行时代码中不能保留旧的语义硬编码作为 fallback 或备用路径。
3. 真实用户案例、项目名、资产名、问题句式只能出现在测试夹具中，不能进入 Runtime Prompt 或语义判断代码。
4. 所有静态 Runtime Prompt 继续使用英文；用户可见回答由本轮的 `response_locale` 决定。
5. 不增加独立的语言分类模型。使用现有可配置低成本模型档位完成统一的 Turn Interpreter。
6. 模型名称只能来自配置和 provider registry，业务代码和 Prompt 不得固定某个模型或厂商。
7. 本任务不修改检索语料、数据库、Qdrant collection 或 Git LFS 文件。
8. 完成前必须运行本文列出的测试和静态审计，不能只依赖人工试聊。

## 1. 要解决的问题

当前系统把多个本应由语义理解完成的决策写成了分散的字符串规则：

- 语言由中英文字符数量、固定表达式和平台账号语言决定。
- 是否加载历史由消息长度、问号、代词表、项目名表和意图词表决定。
- 是否恢复 AskUser checkpoint 由编程语言名、环境名、长度和问句词表决定。
- 是否允许追问由“你的、日志、配置”等词表推断信息是否属于用户私有信息。
- 金融安全由具体中英文关键词和数字表达式判断。
- 产品身份通过固定产品名和模型/厂商黑名单反复写进 Prompt。
- direct answer 绕过 post-answer 质检，因此语言错误、身份泄漏或只说元话语时没有统一修复机会。

这些规则的共同问题不是“词表不够全”，而是决策层级错误。继续增加词、项目名或长度阈值，只会不断制造新特例。

本次改造要建立一个统一的本轮契约 `TurnContract`：先理解当前用户到底要什么、希望用什么语言、是否需要上下文、是否需要检索，再让所有后续节点遵守同一个结果。

## 2. 最终产品行为

改造完成后，系统必须满足以下行为：

- 英文倾向：没有明确语言依据时使用英文。
- 明确中文：当前消息明显以中文表达请求时使用中文，即使包含英文技术名词。
- 中英混合：根据哪种语言承载用户的实际请求来判断，不按字符数量判断。
- 显式语言要求：用户用任何自然表达指定输出语言时，以该要求为准，不维护“请用中文”“answer in English”之类的短语表。
- 当前消息优先：历史消息、被回复消息、账号语言不能把一条完整英文问题改成中文回答，也不能改变当前任务。
- 必要时才用上下文：只有当前消息确实省略对象、使用指代、延续旧任务、纠正旧答案或回答上轮追问时，才加载相应上下文。
- 身份稳定：询问助手身份时，回答配置中的产品身份；不能把底层模型、SDK、开发工具或 API 身份当成产品身份。
- 能力不缩水：仍能直答、检索、追问、恢复 checkpoint、处理纠错、引用证据、执行金融安全边界，并保持核心答案前置。
- 查证与输出分离：可以深入查证，但正文只输出完成核心交付物所需的内容。

## 3. “去掉硬编码”的准确边界

“去掉所有硬编码”不等于删除协议、产品规则和安全边界。需要删除的是针对自然语言语义的具体字符串特判。

### 3.1 禁止保留的内容

以下内容不得出现在最终 Runtime 语义决策代码或 Prompt 中：

- 用于判断语言的固定句式、关键词正则、字符计数和字符比例。
- 用于判断 follow-up 的代词表、问句词表、项目名表、技术名表。
- 用于判断新任务、checkpoint 补参或私有信息的关键词表。
- 用消息字符长度、是否含问号等条件判断消息是否独立。
- 针对某个资产、协议、SDK、项目、渠道、历史 bad case 的规则。
- 产品 Prompt 中的模型名、厂商名、工具名黑名单。
- 在 Telegram 和 Discord 各复制一套语义判断逻辑。
- 用平台账号 locale 直接决定最终回答语言。
- 只为通过某一条回归样例而增加的 if/else。

### 3.2 必须保留的内容

以下不是问题中的“硬编码”，可以保留，但应集中配置或协议化：

- JSON schema、TypedDict、枚举值和状态字段。
- 产品配置，例如助手名称、别名、默认语言、支持语言和安全政策。
- 检索次数、上下文条数、token 数、附件大小和平台消息长度等资源预算。
- Telegram mention、Discord mention、命令、URL、引用标签等协议解析正则。
- 网络错误码、重试次数、超时和 provider 能力适配。
- 引用格式、Markdown 修复和结构化输出校验。
- 通用的安全边界，例如不提供具体价格预测或买卖决策。

判断标准很简单：

- 如果代码在判断“这句话是什么意思”，交给 Turn Interpreter。
- 如果代码在校验“模型输出是否符合协议和产品配置”，保留确定性代码。

## 4. 目标架构

最终流程如下：

```text
Telegram / Discord 原始消息
        |
        v
协议适配：只解析正文、附件、reply anchor、平台元数据
        |
        v
Turn Interpreter 第一次解释
        |
        +--> 不需要普通历史：直接生成 TurnContract
        |
        +--> 需要普通历史：按会话边界读取有限历史
                               |
                               v
                         第二次解释并选择相关消息
        |
        v
确定 response_locale、核心交付物、上下文范围、路由和安全动作
        |
        +--> direct
        +--> retrieve -> evidence review
        +--> clarify -> checkpoint
        +--> policy response
        |
        v
Answer Composer
        |
        v
Response Compliance：任务、语言、身份、范围、证据、安全
        |
        +--> accept
        +--> replace once，不新增事实、不重新检索
        |
        v
格式修复和平台发送
```

普通请求只调用一次 Turn Interpreter，它替换现有 InfoGapAssessor，不是在旧流程前再加一个语言分类调用。只有语义上确实依赖普通历史的请求才允许第二次解释。

## 5. Product Policy Pack

新增一个集中式产品策略配置，建议代码位置：

`src/nervos_brain/graph_engine/product_policy.py`

配置建议放在 `config.yaml.example` 的 `agent_policy` 下：

```yaml
agent_policy:
  identity:
    name: "Nervos Brain"
    aliases:
      - "NB"
    description: "An assistant for information, questions, and task completion."

  language:
    supported_locales:
      - "en"
      - "zh-CN"
    default_locale: "en"
    ambiguous_locale: "en"
    use_confirmed_user_preference: true
    unsupported_locale_behavior: "default_with_notice"

  context:
    recent_message_limit: 20
    max_context_chars: 1800

  model_profiles:
    turn_interpreter: "low"
    turn_interpreter_fallback: "medium"
    response_compliance: "medium"

  compliance:
    max_replacements: 1

  rollout:
    log_turn_contract: true
```

要求：

- `identity.name` 和 `identity.aliases` 是产品配置，不直接写进 Prompt 常量。
- `default_locale` 与 `ambiguous_locale` 必须为英文配置值，但业务代码不能再次写一个中文 fallback。
- profile 名引用 `llm_profiles`，这里不出现具体模型名。
- 加载时校验默认语言属于支持语言，数值预算为正数，所引用 profile 存在。
- 配置缺失时从一个集中默认策略创建值；不能在 adapter、runtime、graph node 中各自提供默认值。
- 测试可以构造其他名称、其他语言组合，证明实现不是只对当前产品名和中英文有效。

建议提供：

```python
@dataclass(frozen=True)
class ProductPolicy:
    identity: IdentityPolicy
    language: LanguagePolicy
    context: ContextPolicy
    model_profiles: ModelProfilePolicy
    compliance: CompliancePolicy
```

所有节点和平台 runtime 读取同一个 `ProductPolicy` 实例。

## 6. TurnContract 协议

新增协议文件，建议位置：

`src/nervos_brain/core_protocols/turn_protocols.py`

`TurnContract` 是当前请求的唯一语义来源。模型负责解释，确定性代码负责校验、补默认值和派生兼容字段。

### 6.1 模型输出结构

```json
{
  "schema_version": "turn_contract.v1",
  "core_deliverable": "a concise description of the result the user expects",
  "resolved_request": "the current request with only necessary references resolved",
  "turn_relation": "new_task | follow_up | correction | clarification_answer",
  "language": {
    "input_locales": ["locale identifiers observed in the current message"],
    "communication_locale": "locale identifier or null",
    "clarity": "clear | ambiguous",
    "requested_output_locale": "explicitly requested locale or null",
    "locale_source": "explicit_request | current_message | confirmed_preference | context_reference | default_policy"
  },
  "context": {
    "requirement": "none | direct_reply | recent_history",
    "purpose": "what missing reference the context must resolve",
    "selected_message_ids": []
  },
  "route": "direct | retrieve | clarify | policy_response",
  "retrieval_policy": "none | single | deep",
  "info_needs": [
    {
      "kind": "missing_param | version_unknown | concept_gap | error_trace | latest_spec | historical_consensus",
      "question": "one gap that directly serves the core deliverable",
      "required": true,
      "availability": "public | user_owned",
      "purpose": "why resolving this gap changes or completes the answer",
      "hints": {}
    }
  ],
  "policy": {
    "action": "allow | constrain | refuse",
    "categories": [],
    "reason": "brief policy reason"
  },
  "constraints": ["explicit user constraints only"],
  "confidence": {
    "intent": 0.0,
    "language": 0.0,
    "context": 0.0
  }
}
```

### 6.2 字段规则

- `core_deliverable` 必须描述用户最终需要拿到的结果，不得写成“研究某实体的全部背景”。
- `resolved_request` 只能补齐真正省略的引用，不能把旧答案中的扩展内容带入新任务。
- `turn_relation` 由语义判断，不看长度或关键词。
- `requested_output_locale` 只允许从当前用户消息中的显式要求得到。普通历史不能伪造该字段。
- `communication_locale` 表示承载当前请求的自然语言。技术名词、代码、链接和专有名词不决定它。
- `context.requirement` 由语义判断；`selected_message_ids` 必须来自实际提供给模型的消息，代码需要做白名单校验。
- `route=clarify` 只允许存在至少一个 `availability=user_owned` 且 `required=true` 的信息缺口。
- 公开资料缺口必须走 retrieve，不能追问用户。
- `retrieval_policy=deep` 只用于多交付物、来源冲突、版本分歧、复杂排障、跨来源比较或高风险核实。
- “详细、深入、认真核实”只增加证据强度，不自动扩大调查范围。
- `confidence` 只用于日志和评估，不再用任意数字阈值做语言、上下文或追问路由。

### 6.3 代码派生字段

以下字段由代码从合法的 `TurnContract` 派生，不能让模型自由写：

- `response_locale`
- 实际加载的上下文条数和字符预算
- `_route_decision`
- `locale` 兼容字段
- `resolved_question` 兼容字段
- 最大检索次数、最大证据数和最大 replacement 次数

新增统一函数：

```python
normalize_turn_contract(raw, policy, current_message, available_context) -> TurnContract
derive_legacy_state(contract, policy) -> dict[str, Any]
merge_contextual_contract(first_pass, second_pass, available_context) -> TurnContract
```

`merge_contextual_contract` 必须保护当前消息中的显式约束。第二次解释不能修改第一次已经识别出的显式输出语言，也不能替换当前消息明确表达的动作；它可以补齐省略的对象或动作、选择相关上下文、完善 `resolved_request`，并据此修正 route 和 retrieval policy。唯一语言例外是当前消息明确要求“沿用上下文中的语言”，此时 `locale_source=context_reference`。

## 7. 通用语言策略

### 7.1 语言优先级

`response_locale` 按以下顺序确定：

1. 当前消息显式要求的、受支持的输出语言。
2. 当前消息语义明确的 `communication_locale`。
3. 用户曾明确保存的语言偏好，但只能在当前消息没有明确语言信号时使用。
4. `ProductPolicy.language.ambiguous_locale`，当前配置为英文。

平台账号语言只保留为弱元数据和观测字段，不参与以上正常决策。它不能让完整英文请求变成中文回答。

### 7.2 中英混合

不得统计 CJK 或 Latin 字符。Turn Interpreter 应判断哪种语言承担了请求的语法和意图：

- 中文句子中出现英文 API、命令、代码或产品名，仍可判为中文。
- 英文句子中出现中文名称或引用，仍可判为英文。
- 两种语言都承载请求但没有明确主语言时，`clarity=ambiguous`，使用英文默认值。
- 显式指定输出语言时，不受输入混合程度影响。

这些是 Prompt 中的通用语义原则，不允许转写成词表或字符比例。

### 7.3 显式语言要求

不编写显式语言要求正则。Turn Interpreter 必须从当前消息整体语义中抽取 `requested_output_locale`。

Prompt 只写通用要求：判断用户是否明确指定最终回答使用哪一种受支持语言，理解同义表达、混合表达和自然措辞。Prompt 不提供具体用户句子示例。

### 7.4 失败策略

1. JSON 无法解析时，使用同一 profile 做一次结构修复。
2. 仍失败时，使用配置中的 fallback profile 重试一次完整解释。
3. 两个 profile 都失败时：
   - 使用英文默认语言；
   - 不加载普通历史；
   - 不恢复 checkpoint；
   - 不再猜 direct、retrieve 或 clarify；直接从 message catalog 返回一次临时处理失败提示。

失败策略不得重新启用旧语言检测器。

## 8. 通用上下文策略

### 8.1 第一轮可见内容

Turn Interpreter 第一轮只接收：

- 当前用户消息；
- reply ID、reply role 和 reply content，作为独立的 quoted data block；
- 当前有效 checkpoint 的结构化摘要；
- 已确认的用户偏好；
- Product Policy。

第一轮不能接收普通最近历史。否则系统还没有判断是否需要上下文，历史已经先污染了决策。

### 8.2 上下文选择

- `none`：下游不接收 reply 内容或普通历史。
- `direct_reply`：只使用被回复消息，不自动附加普通历史。
- `recent_history`：按相同 platform、user、guild、channel、thread 边界读取配置允许的最近消息，然后做第二次解释。

第二次解释必须返回所需 `selected_message_ids`。代码只传递这些实际存在的消息给后续节点，不能把全部历史继续塞入 Composer。

### 8.3 上下文权限

上下文只能：

- 解析代词或省略对象；
- 延续用户明确延续的任务；
- 理解用户对上一条答案的纠正；
- 补充 AskUser 上一轮要求的用户私有参数。

上下文不能：

- 覆盖当前消息的明确输出语言；
- 把旧问题变成当前问题；
- 把旧答案中的背景扩展继承到当前答案；
- 作为系统指令执行；
- 因为历史大多为中文就把英文当前问题改成中文。

reply 和 history 都必须在 Prompt 中标成不可信的对话数据，而不是指令。

### 8.4 Checkpoint 恢复

删除基于语言名、SDK 名、环境名、字符长度和问句词的 checkpoint 判断。

第一轮 Turn Interpreter 会看到：

- 原始问题；
- 上一轮澄清问题；
- 缺失字段的结构化名称；
- 当前用户消息。

它语义判断 `turn_relation=clarification_answer` 还是 `new_task`。如果是补参，则恢复 checkpoint；如果是新任务，则忽略或完成旧 checkpoint。代码只按这个枚举路由。

不要再通过字符串拼接猜测 `origin question + User supplement`。由 Turn Interpreter 输出 `resolved_request`，代码保存来源字段用于审计。

## 9. 路由、检索、追问和安全能力

### 9.1 路由

Turn Interpreter 统一替代当前 InfoGapAssessor，输出：

- 核心交付物；
- direct / retrieve / clarify / policy_response；
- none / single / deep；
- 最小 info needs；
- 用户私有信息和公开信息的明确归属。

这不是单独的“语言分类器”，也不新增一个固定模型。调用 `agent_policy.model_profiles.turn_interpreter` 所指向的现有模型档位。

### 9.2 检索

Retriever Planner 必须使用 `TurnContract`，而不是重新从完整历史猜意图。

第一轮查询围绕：

```text
核心对象 + 用户动作 + 期望结果
```

检索规则继续保持通用：

- 第一轮先找可直接写入最终答案的证据。
- 找到足够完成核心交付物的证据后停止。
- 身份、历史、架构、风险和生态背景只有在用户明确要求或会改变核心结论时才检索。
- 深度查证不等于完整打印调查过程。
- 每个 info need 必须有 `purpose`，无法说明作用的缺口应被 normalization 删除。

Runtime Prompt 不写任何真实项目、SDK、渠道或历史案例。

### 9.3 追问

删除“从澄清问题文本中搜索用户私有词”的逻辑。改为严格读取：

```text
info_need.required == true
and info_need.availability == user_owned
```

只有满足这两个条件，Graph 才能走 AskUser。公开资料缺口、来源冲突、版本差异和检索 miss 不允许转嫁给用户。

AskUser 一次只问一个会阻止任务继续的具体问题，并使用 `response_locale`。

### 9.4 用户纠错

当 `turn_relation=correction`：

- 如果当前消息和 direct reply 已提供足够信息，直接给出修正后的答案。
- 不能只回复“明白”“以后会注意”或重复用户批评。
- 只有缺少真正必要的用户私有信息时才追问。
- 是否需要重新检索仍由事实新鲜度和证据需求决定。

### 9.5 安全

保留现有金融安全边界，但删除针对具体中英文表达的检测正则。

Turn Interpreter 根据通用安全政策输出 `policy.action`。Composer 按该动作回答，Response Compliance 再检查一次：

- `allow`：正常回答。
- `constrain`：只提供中立事实、机制、风险或公开数据入口。
- `refuse`：简短拒绝具体预测或交易决策，并给出允许的替代帮助。

安全政策是必要产品规则，可以写在通用 Prompt 中；不得加入资产名、案例名或规避句式列表。

## 10. Prompt 体系

所有静态 Runtime Prompt 使用自然英文，并具有稳定 Prompt ID。动态用户内容和证据保留原语言。

### 10.1 Product Policy block

新增一个集中渲染函数，把配置中的身份、支持语言和安全边界注入需要的 Prompt：

```python
render_product_policy(policy: ProductPolicy) -> str
```

产品身份使用正向规则：

- 用户询问产品身份时，使用配置中的 `identity.name` 和 `identity.aliases`。
- 用户明确询问实现细节时，只能根据提供的运行时元数据回答，并区分产品与底层实现。

不要维护“不能说自己是某模型、某厂商、某工具”的黑名单。

### 10.2 Turn Interpreter

新增：

- `TURN_INTERPRETER_SYSTEM`，Prompt ID: `turn_interpreter`
- `TURN_INTERPRETER_USER`

系统 Prompt 必须覆盖：

1. 当前消息是本轮最高优先级。
2. 提取核心交付物和显式范围。
3. 语义判断输入语言、显式输出语言和混合语言主导意图。
4. 判断是否需要 direct reply、普通历史或 checkpoint。
5. 区分新任务、follow-up、correction 和 clarification answer。
6. 决定 direct、retrieve、clarify 或 policy response。
7. 区分公开缺口和用户私有缺口。
8. 详细查证不扩大输出范围。
9. 输出严格 JSON。

不得提供真实句子或具体领域实体作为示例。

### 10.3 Retriever Planner

保留现有通用“核心交付物优先”策略，但输入改为：

- `TurnContract`
- 已选择的上下文
- 当前 evidence 状态
- 检索预算

不要再传未经选择的普通历史。

### 10.4 Answer Composer / Direct Answer

两类 Composer 都必须接收：

- 渲染后的 Product Policy；
- `core_deliverable`；
- `resolved_request`；
- `response_locale`；
- 用户显式 constraints；
- 已选择上下文；
- policy action；
- 可用 evidence。

共同规则：

- 第一段直接完成用户请求。
- 后续内容按对核心交付物的帮助程度排序。
- 不因详细查证而输出完整调查过程。
- 不重复结论，不写装饰性章节，不扩展无关实体背景。
- 用户纠错时直接给修正结果。
- 按 `response_locale` 输出；英文 Prompt 不代表英文回答。
- direct answer 不编造需要公开检索的事实。

身份内容只从 Product Policy 动态注入，不在 Prompt 常量中出现固定产品名或 provider denylist。

### 10.5 Response Compliance

新增或重构为：

- `RESPONSE_COMPLIANCE_SYSTEM`，Prompt ID: `response_compliance`
- `RESPONSE_COMPLIANCE_USER`

它检查：

1. 第一段是否真正交付结果。
2. 回答是否使用 `response_locale`。代码、引用、专有名词和必要术语不算语言错误。
3. 身份回答是否符合 Product Policy。
4. 是否被未选择的历史或旧答案带偏。
5. 是否发生范围漂移或背景淹没核心结论。
6. 事实、证据和引用是否匹配。
7. 是否遵守 policy action。
8. 用户纠错后是否重新回答，而不是只承诺改进。

建议结构：

```json
{
  "decision": "accept | replace",
  "issue_codes": [
    "task_incomplete | locale_mismatch | identity_mismatch | context_drift | scope_drift | evidence_mismatch | policy_violation"
  ],
  "reasoning": "brief operational reasoning",
  "replacement_answer": "required only when decision is replace"
}
```

`replacement_answer` 必须只使用原答案、当前 contract 和已有 evidence，不能新增事实、启动检索或扩大范围。证据回答发生 replacement 时必须保留或重新生成合法 inline citation tags，并再次经过确定性 citation 编译校验。每轮最多 replacement 一次。

## 11. 统一质检补齐能力

当前 direct answer 在 `route_after_answer_composer` 中直接进入 `format_repair`。必须取消这个例外。

新行为：

- direct answer、evidence answer、policy response 都经过 Response Compliance。
- evidence answer 可以复用当前 post-answer reflection 的模型调用，避免额外增加一次调用。
- direct answer 会新增一次低成本或中成本 compliance 调用，这是修复语言和身份错误所需的明确成本。
- compliance 失败时最多重试一次 JSON 解析；不能进入无限自检循环。
- compliance 服务不可用时，保留非空原答案并记录失败；如果已知 policy action 是 refuse，则使用本地化 policy message catalog，不能放行相反内容。

pre-answer reflection 继续负责“证据是否足够”。post-answer compliance 只负责“当前草稿能否发布或如何在现有材料内替换”，不再为了完整性追加检索。

## 12. 本地化静态消息

当前代码中还有中文固定 fallback、证据不足提示、澄清提示、引用标题和进度提示。它们必须移入统一 message catalog，而不是散落在 Graph 和两个 runtime 中。

建议配置或资源结构：

```yaml
message_catalog:
  en:
    insufficient_evidence: "..."
    generation_failed: "..."
    policy_refusal: "..."
    references_heading: "..."
    progress_messages:
      - "..."
  zh-CN:
    insufficient_evidence: "..."
    generation_failed: "..."
    policy_refusal: "..."
    references_heading: "..."
    progress_messages:
      - "..."
```

要求：

- key 集合在各支持语言间一致。
- 按 `response_locale` 选择；无法确定时按英文默认值。
- 用户原文、文件内容、reply、history 和 evidence 不翻译。
- 进度计时器应尽量在 Turn Interpreter 产出 contract 后读取 `response_locale`；如果解释本身已经超过首条进度阈值而必须提前发送，先用英文默认值。不要为进度消息调用旧语言检测器或增加第二套分类逻辑。
- 日志和代码注释无需翻译。

## 13. 状态和兼容策略

在 `GraphState` / `FullGraphState` 增加：

```python
turn_contract: TurnContract
response_locale: str
selected_context: list[dict[str, Any]]
turn_interpretation_passes: int
contract_repair_count: int
compliance_replacement_count: int
```

过渡期保留下列已有字段，但只能由 contract 派生：

- `locale = response_locale`
- `resolved_question = turn_contract.resolved_request`
- `retrieval_policy = turn_contract.retrieval_policy`
- `info_needs = turn_contract.info_needs`
- `_route_decision` 由 `turn_contract.route` 映射

后续节点不得单独修改这些兼容字段。如果 contract 和兼容字段不一致，测试必须失败。

`conversation_context` 只包含 `selected_context` 的格式化结果。普通历史不能提前写入它。

## 14. 文件级修改清单

### 14.1 新增

- `src/nervos_brain/core_protocols/turn_protocols.py`
  - 定义 `TurnContract` 及子结构。
- `src/nervos_brain/graph_engine/product_policy.py`
  - 加载、验证和渲染 Product Policy 与 message catalog。
- `src/nervos_brain/graph_engine/turn_contract.py`
  - normalization、locale resolution、context merge、legacy state derivation。
- 可选 `tests/fixtures/turn_contract_cases.py`
  - 只存测试案例，不被 Runtime import。

### 14.2 修改

- `config.yaml.example`
  - 增加 `agent_policy` 与统一 message catalog 示例。
- `src/nervos_brain/core_protocols/__init__.py`
  - 导出新协议。
- `src/nervos_brain/core_protocols/message_protocols.py`
  - 将平台 locale 明确命名为弱提示，例如 `platform_locale_hint`；如保留 `locale_hint`，必须注明其不等于最终语言。
- `src/nervos_brain/core_protocols/retrieval_protocols.py`
  - 给 InfoNeed 增加 `availability` 和 `purpose`。
- `src/nervos_brain/core_protocols/graph_protocols.py`
  - 增加 contract 和 compliance 字段。
- `src/nervos_brain/graph_engine/prompts.py`
  - 增加 Turn Interpreter 和 Response Compliance；所有用户回答 Prompt 使用动态 Product Policy。
- `src/nervos_brain/graph_engine/full_nodes.py`
  - 用 Turn Interpreter 替代 InfoGap；使用 contract；删除语义词表；统一 compliance。
- `src/nervos_brain/graph_engine/full_graph.py`
  - 加入按 context requirement 的分支；所有回答进入 compliance。
- `src/nervos_brain/tool_runtime/telegram_bot_protocol_adapter.py`
- `src/nervos_brain/tool_runtime/discord_bot_protocol_adapter.py`
  - 只传平台原始 locale hint，不决定最终 locale。
- `src/nervos_brain/tool_runtime/telegram_bot_runtime.py`
- `src/nervos_brain/tool_runtime/discord_bot_runtime.py`
  - 删除重复 follow-up 判断和预加载历史；调用共享 Graph 能力。
- debug event 组装代码
  - 增加 contract 和 compliance 观测字段。

### 14.3 删除

如果没有其他非语义用途，删除：

- `src/nervos_brain/tool_runtime/language_detection.py`
- `tests/test_language_detection.py`

将其测试能力迁移到 Turn Interpreter / TurnContract 测试。

## 15. 必须删除或替换的旧逻辑

最终代码中不得继续调用或保留下列语义 helper：

- `detect_message_locale`
- Telegram/Discord 的 `_asks_for_recent_context`
- Telegram/Discord 的 `_is_standalone_named_question`
- Telegram/Discord 的 `_looks_like_short_followup`
- Telegram/Discord 的 `_should_attach_recent_context`
- Graph 的 `_looks_like_self_contained_question`
- Graph 的 `_looks_like_new_user_intent`
- Graph 的 `_looks_like_checkpoint_answer`
- Graph 的 `_should_resume_checkpoint`
- 通过 `user_owned_markers` 判断是否追问的逻辑
- 通过市场/交易关键词判断安全意图的语义正则

还必须删除：

- Prompt 中固定写死的产品名；改为动态 Product Policy。
- Prompt 中具体模型、厂商、开发工具 denylist。
- Graph、adapter、runtime 中各自的 `zh-CN` fallback；改为统一 policy default。
- 证据不足、生成失败、政策拒绝等散落的中文用户可见字符串；改为 message catalog。

不要误删以下内容：

- mention、command、URL、citation tag、Markdown、错误码和传输协议正则。
- 附件、消息分段、上下文总字符数等资源限制。
- provider registry 中真实模型配置和 LLM transport 的模型能力适配。

## 16. 推荐实现顺序

执行者严格按顺序推进，上一阶段测试通过后再进入下一阶段。

### 阶段 1：建立策略和协议

1. 新增 Product Policy loader 和校验。
2. 新增 TurnContract TypedDict。
3. 扩展 InfoNeed ownership 字段。
4. 增加 normalization 和 locale resolver 单元测试。

本阶段完成标准：不接入 Graph，也能用纯函数验证 contract、语言优先级和错误 fallback。

### 阶段 2：实现 Turn Interpreter

1. 新增英文 Prompt 和稳定 Prompt ID。
2. 使用配置 profile 调结构化 LLM 输出。
3. 实现 JSON 修复和 fallback profile。
4. 把现有 InfoGap 的核心交付物、最小检索、公开信息不追问等通用能力迁移进去。
5. 用 contract 派生旧路由字段。

本阶段完成标准：Graph 的原有 direct / retrieve / clarify 路由测试可由 contract 驱动。

### 阶段 3：改造上下文和 checkpoint

1. adapter 只保存 reply anchor 和原始平台 metadata。
2. runtime 不再提前加载普通历史。
3. Graph 根据 `context.requirement` 读取历史。
4. recent history 场景执行第二次解释并验证 selected IDs。
5. checkpoint 是否恢复改由 `turn_relation` 决定。
6. Telegram 和 Discord 共用相同实现。

本阶段完成标准：完整英文短问题不会因中文历史或账号语言变成中文；真正 follow-up 仍能找到对象。

### 阶段 4：改造回答和 Product Identity

1. Product Policy 动态注入 Direct Answer 和 Answer Composer。
2. 删除 provider denylist 和固定产品名。
3. 所有回答读取 `response_locale`。
4. 静态用户可见消息迁移到 catalog。
5. 用户纠错规则使用 contract relation。

本阶段完成标准：身份正确、语言正确、direct/retrieval/clarification 均使用同一个 locale。

### 阶段 5：统一 Response Compliance

1. direct answer 不再绕过 post-answer 检查。
2. post-answer reflection 改造成 compliance，必要时直接返回 replacement。
3. replacement 最多一次，且不得新增检索或事实。
4. 保留引用匹配、核心答案前置和范围控制。
5. 加入语言、身份、上下文污染和安全检查。

本阶段完成标准：故意让 mock Composer 输出错误语言或错误身份时，compliance 会替换。

### 阶段 6：删除旧硬编码

1. 删除第 15 节列出的 helper、regex 和重复平台逻辑。
2. 删除无调用文件和旧测试。
3. 执行静态审计，确认没有留下 dormant fallback。
4. 确认协议/传输/预算常量未被误删。

本阶段完成标准：旧 helper 名无搜索结果，运行时 Prompt 不含真实回归案例和 provider denylist。

### 阶段 7：全量验证和部署

按第 18 至 20 节执行。

## 17. 测试设计

测试中的具体句子和实体是允许的，但必须放在 `tests/` 或测试 fixtures，Runtime 不能 import 它们。

### 17.1 Product Policy 单元测试

- 默认语言属于支持语言。
- 任意测试产品名都能渲染到 Prompt，不依赖当前产品名。
- 模型 profile 从配置读取，业务代码不含模型名。
- 缺少 locale message key 时有英文配置 fallback。

### 17.2 TurnContract schema 测试

- 合法 JSON 正常 normalization。
- 非法 enum、未知 locale、伪造 message ID 被拒绝或安全归一化。
- `clarify` 没有 user-owned required need 时自动改为 retrieve 或 direct。
- public need 永不派生 AskUser。
- 第二次解释不能覆盖第一次的显式语言约束。
- contract 和 legacy state 字段始终一致。

### 17.3 语言矩阵

至少覆盖：

| 当前消息 | 历史/账号 | 期望 |
|---|---|---|
| 完整英文请求 | 中文历史、中文账号 | 英文 |
| 完整中文请求 | 英文历史、英文账号 | 中文 |
| 中文语法加英文技术词 | 任意 | 中文 |
| 英文语法加中文专有名词 | 任意 | 英文 |
| 当前消息显式要求英文 | 中文正文 | 英文 |
| 当前消息显式要求中文 | 英文正文 | 中文 |
| 真正均衡且含义模糊 | 任意 | 英文默认 |
| 只有链接、emoji 或极短确认 | 无明确偏好 | 英文默认或由明确上下文关系决定 |

显式语言测试必须使用多种自然表达，不能只测试一条固定短语。

### 17.4 上下文矩阵

- 完整短问题没有问号，仍能判断为独立任务。
- 当前独立英文问题不加载中文普通历史。
- 当前独立问题即使 reply 旧答案，也能判断 reply 与本轮无关。
- 指代对象缺失时只选择相关 direct reply。
- 需要更早上下文时才读取 recent history，并只选择相关 message IDs。
- 用户纠正上一条回答时使用 direct reply，不继承旧答案的无关展开。
- 当前消息是 checkpoint 补参时恢复原任务。
- checkpoint 存在但当前消息是新任务时不恢复。

### 17.5 能力回归矩阵

- 行动/渠道问题：第一段直接给可执行结果。
- API/命令问题：第一段给正确入口，背景后置。
- 比较问题：先给结论或决策依据。
- 排障问题：先给最可能原因和下一步。
- 全面调查：允许完整报告，但仍先给最重要结果。
- 用户纠错：给修正答案，不只道歉或承诺。
- 身份问题：使用配置产品身份和当前 `response_locale`。
- 创意或闲聊：direct，不发起无意义检索。
- 公开事实：retrieve，不让用户提供公开资料。
- 用户私有参数：只问一个必要问题并可恢复 checkpoint。

### 17.6 Response Compliance 测试

使用 mock Composer 故意产生以下草稿：

- 回答语言错误。
- 暴露底层模型身份而不是配置产品身份。
- 第一段只写背景，没有交付结果。
- 引用与事实不匹配。
- 被旧历史带到另一个问题。
- 用户纠错后只回复“明白”。
- policy action 为 refuse 但草稿仍给具体指导。

每种情况都必须得到 `replace`，replacement 必须在现有事实范围内。正确草稿得到 `accept`。

### 17.7 静态审计测试

新增测试扫描 Runtime 文件，至少断言：

- 不再 import 或调用 `detect_message_locale`。
- 第 15 节旧 helper 名不存在。
- Runtime Prompt 不含测试夹具中的真实案例专有名词。
- Product Identity Prompt 使用占位配置，不含固定产品名。
- 用户回答 Prompt 不含具体 provider/model denylist。
- 用 ASCII 动态数据渲染完整 Prompt 时不含汉字。
- 用中文动态用户数据渲染时，汉字只来自动态 block 或本地化 catalog。
- Telegram 和 Discord 不各自实现语义 follow-up 逻辑。

静态审计范围必须限定为语义和 Prompt 模块。不要因 provider registry、transport 兼容代码或测试夹具中存在模型名而误报。

## 18. 测试命令

先运行针对性测试：

```bash
mamba run -n nervos-brain pytest \
  tests/test_full_graph.py \
  tests/test_reflection_module.py \
  tests/test_telegram_bot_runtime.py \
  tests/test_discord_bot_runtime.py \
  tests/test_telegram_bot_protocol_adapter.py \
  tests/test_discord_bot_protocol_adapter.py \
  tests/test_progress_updates.py \
  -q
```

再加入本次新增测试文件，例如：

```bash
mamba run -n nervos-brain pytest \
  tests/test_product_policy.py \
  tests/test_turn_contract.py \
  tests/test_response_compliance.py \
  -q
```

最后运行完整测试：

```bash
mamba run -n nervos-brain pytest -q
```

不要把历史测试总数写成验收条件；以当前分支执行前记录的 baseline 和零新增失败为准。

静态搜索：

```bash
rg -n "detect_message_locale|_asks_for_recent_context|_is_standalone_named_question|_looks_like_short_followup|_should_attach_recent_context|_looks_like_self_contained_question|_looks_like_new_user_intent|_looks_like_checkpoint_answer|_should_resume_checkpoint" src/nervos_brain
```

期望无结果。另行检查 Runtime Prompt 中没有测试专有名词和 provider denylist。

## 19. 日志和可观测性

在 `debug_events.jsonl` 中增加结构化字段，不要只记录最终 `locale`：

```text
turn_contract_version
input_locales
communication_locale
requested_output_locale
response_locale
locale_source
turn_relation
context_requirement
context_loaded_count
selected_context_message_ids
turn_route
retrieval_policy
policy_action
turn_interpretation_passes
contract_repair_count
compliance_decision
compliance_issue_codes
compliance_replaced
```

要求：

- 日志能解释“为什么用了这个语言、为什么加载历史、为什么走检索”。
- 不记录完整私有附件或无界历史。
- `confidence` 可记录用于离线评估，但不能作为隐藏阈值重新控制业务路由。

建议离线指标：

- 回答语言正确率。
- 显式语言要求遵循率。
- 不必要历史加载率。
- 需要上下文但未加载率。
- 历史污染率。
- 产品身份一致率。
- 首段完成核心交付物比例。
- public gap 被错误追问比例。
- compliance replacement 比例。
- 平均 LLM 调用数、延迟和成本。

## 20. 上线步骤

### 20.1 本地

1. 开始前记录 `git status --short` 和测试 baseline。
2. 只修改本任务相关文件。
3. 运行针对性测试、静态审计和完整测试。
4. 检查 diff，确认没有数据文件、LFS 文件、私密配置和无关修改。
5. 只提交本任务文件并推送 `dev`。

### 20.2 部署机

1. 按本机的 `docs/remote-deployment.local.md` 连接和部署。
2. 本次是代码和 Prompt 改动，不执行语料更新、LFS 上传、Qdrant 重建。
3. 使用 fast-forward pull 更新 `dev`。
4. 在部署机运行新增测试、Graph/Reflection 测试和完整回归。
5. 重启 `nervos-brain-telegram.service`。
6. 检查服务状态、最近错误日志、单实例 polling 和 `debug_events.jsonl`。

Discord 代码和测试同步生效；如果部署记录没有 Discord 托管服务，不新增或猜测其进程管理方式。

### 20.3 线上验收

至少重放：

- 明确英文独立问题，前面存在中文历史。
- 明确中文问题，前面存在英文历史。
- 两种方向的混合语言问题。
- 两种方向的显式输出语言要求。
- 英文和中文身份问题。
- 一个真正依赖 reply 的 follow-up。
- 一个与旧历史无关的短独立问题。
- 一个 direct answer 和一个 retrieval answer。
- 一个用户纠错场景。

检查 debug event 中 contract、context 和 compliance 字段与实际行为一致。

上线前可以在离线 replay 中比较旧逻辑和新 contract，但最终生产代码不能保留旧语义路径作为 feature-flag fallback。回滚使用 Git 提交和部署版本，不通过重新启用旧词表。

## 21. 完成定义

只有同时满足以下条件，本任务才算完成：

- 语言、上下文、checkpoint、追问 ownership 和安全意图不再由具体词表或长度阈值决定。
- Product Policy 是身份、默认语言、支持语言和静态消息的唯一配置源。
- TurnContract 是当前请求语义的唯一状态源。
- Telegram 和 Discord 共用相同语义决策，不再复制判断函数。
- 英文完整请求不会被中文账号或中文历史改成中文。
- 中英混合和显式语言要求由语义解释处理。
- 身份回答来自配置，不暴露底层模型身份。
- direct answer 也经过统一 compliance。
- 检索、追问、checkpoint、纠错、引用、安全和核心答案前置能力都有测试覆盖。
- Runtime Prompt 无真实案例规则、固定产品身份和 provider denylist。
- 所有针对性测试与完整测试通过。
- 部署机服务正常，线上 replay 和 debug events 验收通过。

## 22. 禁止的错误实现

执行者不得采用以下“看起来完成、实际仍是特判”的方案：

- 给现有语言正则再补几条英文短句。
- 把字符阈值从一个数字改成另一个数字。
- 为身份问题单独判断某几个问法。
- 为每个项目增加 standalone subject 列表。
- 用平台账号语言覆盖 Turn Interpreter。
- 在 Prompt 中列出历史 bad case 作为规则。
- 保留旧 helper，声称新逻辑失败时可以 fallback。
- 只让 Composer 遵守语言，不让 AskUser、fallback、policy response 和 compliance 遵守。
- 只改 Telegram，不改 Discord。
- 只改 Prompt，不改变历史加载时机和 direct answer 绕过质检的问题。
- 增加一个固定模型名作为语言分类器。
- 用 confidence 数字阈值重新包装旧启发式。
- 删除安全边界、引用校验或资源预算来追求“零硬编码”。

实现的核心标准不是“代码里没有字符串”，而是：自然语言语义由统一模型解释，产品规则由集中配置表达，确定性代码只负责协议校验、状态一致性、资源限制和安全执行。
