# Nervos Brain systemd 用户定时任务

这些文件是本地或服务器部署模板，不是运行测试所必需的文件。安装前需要按实际机器路径修改 service 里的环境变量。

## Talk forum 增量更新

Talk forum 数据库已经完成全量爬取；这个 timer 只负责每天抓取最新页面，并跳过已经存在的帖子锚点。

默认频率：每 24 小时一次。默认命令：

```bash
conda run -n nervos-brain python scripts/run_talk_forum_ingest.py --latest-pages 3 --incremental
```

为当前 Linux 用户安装：

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/nervos-talk-forum-ingest.service ~/.config/systemd/user/
cp deploy/systemd/nervos-talk-forum-ingest.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now nervos-talk-forum-ingest.timer
```

如果仓库路径、环境名或 Conda/Mamba 路径不同，安装前需要编辑这些 service 变量：

```ini
Environment=PROJECT_ROOT=%h/path/to/Nervos-Brain
Environment=ENV_RUNNER_BIN=%h/miniconda3/bin/conda
Environment=ENV_NAME=nervos-brain
Environment=TALK_LATEST_PAGES=3
```

查看状态和日志：

```bash
systemctl --user status nervos-talk-forum-ingest.timer
systemctl --user list-timers nervos-talk-forum-ingest.timer
journalctl --user -u nervos-talk-forum-ingest.service -n 100
```


## GitHub docs/code 增量更新

GitHub docs/code 语料默认每周刷新一次。运行 state 保存在 `data/ingest_state/`，这是本机私有游标，不提交。公开 manifest 写入 `data/manifests/`，用于记录公开语料覆盖的 repo、branch、commit 和写入数量。

默认命令：

```bash
conda run -n nervos-brain python scripts/run_github_docs_ingest.py --incremental
conda run -n nervos-brain python scripts/run_github_code_ingest.py --incremental
```

安装用户 timer：

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/nervos-github-ingest.service ~/.config/systemd/user/
cp deploy/systemd/nervos-github-ingest.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now nervos-github-ingest.timer
```

可选：配置 GitHub token 文件以降低 rate limit 风险：

```bash
mkdir -p ~/.config/nervos-brain
cat > ~/.config/nervos-brain/github-ingest.env <<'ENV'
GITHUB_TOKEN=<GITHUB_TOKEN>
ENV
chmod 600 ~/.config/nervos-brain/github-ingest.env
```

查看状态和日志：

```bash
systemctl --user status nervos-github-ingest.timer
systemctl --user list-timers nervos-github-ingest.timer
journalctl --user -u nervos-github-ingest.service -n 100
```
