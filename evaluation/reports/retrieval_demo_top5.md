# 三条典型查询的前 5 条（A 现状 / B 关向量路 / C 真实向量）

由 `scripts/eval_retrieval.py` 的同一套配置跑出，三个库合并后取 top 5。`paths` 是该结果由哪几路召回（融合前）。仅 `vector` 表示它只靠向量路进来。


## How to open a Fiber channel  —  配置 A

| # | 库 | 标题 | paths |
|---|---|---|---|
| 1 | github_code | nervosnetwork/fiber/crates/fiber-lib/src/rpc/channel.rs | bm25, fuzzy |
| 2 | docs | nervosnetwork/ncp-poc/forks/ckb-vm/SECURITY.md | vector |
| 3 | docs | nervosnetwork/fiber-docs/content/docs/how-it-works.mdx | bm25 |
| 4 | forum_talk | [DIS] Spore Protocol Mainnet Launch Sponsorship Proposal \| Spore Protocol 上主网赞助提案 — reply  | vector |
| 5 | forum_talk | [Idea Hunt] Help AI Agent find cool CKB + Fiber + Web5 use cases! / [集思广益] 帮 AI Agent 寻找有趣 | bm25 |

## get_live_cell  —  配置 A

| # | 库 | 标题 | paths |
|---|---|---|---|
| 1 | forum_talk | How to get the average occupied bytes per live cell in CKB? | bm25, exact |
| 2 | github_code | nervosnetwork/ckb-py-integration-test/test_cases/feature/test_get_live_cell.py | bm25, exact |
| 3 | forum_talk | How to get the average occupied bytes per live cell in CKB? — reply #2 | bm25, exact |
| 4 | github_code | nervosnetwork/ckb-rust-integration-test/feature/tests/mockrpc/get_live_cell.rs | bm25, exact |
| 5 | docs | nervosnetwork/ckb/util/launcher/README.md | vector |

## CKB 学习资料推荐  —  配置 A

| # | 库 | 标题 | paths |
|---|---|---|---|
| 1 | docs | nervosnetwork/ckb/util/fixed-hash/README.md | vector |
| 2 | docs | appfi5/SupeRISE-for-agent/design/README.md | bm25 |
| 3 | forum_talk | CKB 中文名字征集专用帖 — reply #80 | vector |
| 4 | forum_talk | 不太懂比特币的pow是啥，但全球电脑都计算无意义的数学题是决不能容忍的。 — reply #3 | bm25 |
| 5 | github_code | appfi5/ckb-labels/information/udt/testnet/0xd415a3fc4e7e88b7e520fd08dd16282c223fb0fae8001f | vector |

## How to open a Fiber channel  —  配置 B

| # | 库 | 标题 | paths |
|---|---|---|---|
| 1 | github_code | nervosnetwork/fiber/crates/fiber-lib/src/rpc/channel.rs | bm25, fuzzy |
| 2 | docs | nervosnetwork/fiber-docs/content/docs/how-it-works.mdx | bm25 |
| 3 | forum_talk | [Idea Hunt] Help AI Agent find cool CKB + Fiber + Web5 use cases! / [集思广益] 帮 AI Agent 寻找有趣 | bm25 |
| 4 | github_code | nervosnetwork/ckb-explorer/app/models/fiber_channel.rb | fuzzy |
| 5 | docs | nervosnetwork/fiber/crates/fiber-lib/src/rpc/README.md | bm25 |

## get_live_cell  —  配置 B

| # | 库 | 标题 | paths |
|---|---|---|---|
| 1 | forum_talk | How to get the average occupied bytes per live cell in CKB? | bm25, exact |
| 2 | github_code | nervosnetwork/ckb-py-integration-test/test_cases/feature/test_get_live_cell.py | bm25, exact |
| 3 | forum_talk | How to get the average occupied bytes per live cell in CKB? — reply #2 | bm25, exact |
| 4 | github_code | nervosnetwork/ckb-rust-integration-test/feature/tests/mockrpc/get_live_cell.rs | bm25, exact |
| 5 | docs | nervosnetwork/ckb-cli/docs/API-Server.md | bm25 |

## CKB 学习资料推荐  —  配置 B

| # | 库 | 标题 | paths |
|---|---|---|---|
| 1 | docs | appfi5/SupeRISE-for-agent/design/README.md | bm25 |
| 2 | forum_talk | 不太懂比特币的pow是啥，但全球电脑都计算无意义的数学题是决不能容忍的。 — reply #3 | bm25 |
| 3 | github_code | appfi5/nervos-official-website/public/locales/zh/learn.json | bm25 |
| 4 | docs | nervosnetwork/force-bridge/docs/zh/user-guide.md | bm25 |
| 5 | forum_talk | 一份区块链活动推荐清单（月更） | bm25 |

## How to open a Fiber channel  —  配置 C

| # | 库 | 标题 | paths |
|---|---|---|---|
| 1 | github_code | nervosnetwork/fiber/crates/fiber-lib/src/rpc/channel.rs | bm25, fuzzy, vector |
| 2 | docs | nervosnetwork/fiber/tests/bruno/e2e/open-use-close-a-channel/README.md | bm25, vector |
| 3 | docs | nervosnetwork/fiber-docs/content/docs/how-it-works.mdx | bm25, vector |
| 4 | forum_talk | [Idea Hunt] Help AI Agent find cool CKB + Fiber + Web5 use cases! / [集思广益] 帮 AI Agent 寻找有趣 | bm25, vector |
| 5 | docs | nervosnetwork/fiber/docs/network-nodes.md | bm25, vector |

## get_live_cell  —  配置 C

| # | 库 | 标题 | paths |
|---|---|---|---|
| 1 | github_code | nervosnetwork/ckb-py-integration-test/test_cases/feature/test_get_live_cell.py | bm25, exact, vector |
| 2 | github_code | nervosnetwork/ckb-rust-integration-test/feature/tests/mockrpc/get_live_cell.rs | bm25, exact, vector |
| 3 | forum_talk | How to get the average occupied bytes per live cell in CKB? | bm25, exact |
| 4 | forum_talk | How to get the average occupied bytes per live cell in CKB? — reply #2 | bm25, exact |
| 5 | github_code | nervosnetwork/neuron/packages/neuron-ui/src/services/remote/cellManage.ts | bm25, vector |

## CKB 学习资料推荐  —  配置 C

| # | 库 | 标题 | paths |
|---|---|---|---|
| 1 | docs | ckb-devrel/CKB-Developer-Resource/README.md | vector |
| 2 | docs | appfi5/SupeRISE-for-agent/design/README.md | bm25 |
| 3 | forum_talk | CKB 开发和技术讨论索引贴 — reply #2 | vector |
| 4 | forum_talk | 不太懂比特币的pow是啥，但全球电脑都计算无意义的数学题是决不能容忍的。 — reply #3 | bm25 |
| 5 | github_code | nervosnetwork/ckb/test/template/ckb.toml | vector |
