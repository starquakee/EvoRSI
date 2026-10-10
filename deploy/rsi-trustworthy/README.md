# rsi-trustworthy 独立隔离部署 (US-006)

独立于旧 Windows 挂载栈（6580/10150）的可信评估栈。旧栈保持原样作为回退；
本目录只部署新栈，所有命令只影响 `rsi-trustworthy` compose 项目。

## 布局

- 网关：`http://127.0.0.1:6581`（仅回环，只代理 `/api/v1/`）。
- 网络：
  - `control`（internal）：nginx、api、dispatcher、postgres、redis。
  - `execution`（internal + `gateway_mode_ipv4=isolated`）：dispatcher、worker。
    internal 按 Docker 文档仍保留宿主机网桥网关地址；isolated 模式将其移除
    （本引擎 28.2.2 用同一 worker 镜像实测通过），worker 无任何宿主机侧网关。
  - `edge`（非 internal）：仅 nginx。本机 Docker Desktop 对只挂 internal
    网络的容器不创建宿主机端口监听（2026-10-08 用探针容器验证），因此
    网关额外挂到 edge；worker 不在 edge/control 上。
- worker（`rsi_worker`）：
  - 复用 `ghcr.io/lifeissosolong/openmle-sandbox-worker:1.0.0`，entrypoint
    换成 `worker/minimal_worker.py`（python3 标准库实现 dispatcher 的
    `/v1/shell/*` 协议；替代需要 seccomp=unconfined 并启动浏览器/VNC 的
    AIO `/opt/gem/run.sh`）。
  - 无发布端口、无 docker socket、默认 seccomp（非 unconfined）、
    `no-new-privileges:true`、`read_only: true` 根文件系统 +
    64MB tmpfs `/tmp`（nosuid/nodev/noexec）、`pids_limit: 128`、
    `mem_limit: 4g`、`cpus: 2.0`、GPU 通过 compose device reservation。
  - capability：`cap_drop: ALL` + 仅 `SETUID/SETGID/KILL`（root supervisor
    专用；候选进程以非 root uid 65432 运行、无 capability、环境变量洗白、
    继承容器级 no-new-privileges）。镜像兼容性用完全相同的选项实测：
    torch 2.6.0+cu124 / RTX 4060 Ti / CUDA 张量运算 OK。
  - 挂载：公共输入（jobs/uploads）只读 + 每任务 scratch
    （`/mnt/local_sandbox_workdir`）可写 + 控制密钥文件只读
    （宿主 0600，容器 root 无 CAP_DAC_OVERRIDE 也读不到，候选 uid 更读不到）。
    评测器注册表/答案树（`research/evaluator`）从不挂载进 worker。
  - 控制面认证（batch-2 修复）：`/v1/*` 全部要求 `X-RSI-Control-Key`。
    密钥由 supervisor 启动时通过 setuid 助手子进程读入内存（文件属宿主
    uid、0600）；候选进程既不能读密钥文件也不能读 supervisor 的
    /proc/1/environ（validate_stack.py 实测），候选代码无法驱动 worker
    控制 API。未认证调用一律 401。
  - 单执行槽 + 已证明清理：同一时刻最多一个会话；会话退出/被杀后，
    supervisor 用候选 uid 的 /proc 扫描 SIGKILL 全部残留（含 setsid
    脱离进程组的子孙），并等待输出管道排空，确认零候选进程后才释放
    执行槽并上报完成；清理无法证明时 worker 自我熔断（healthz 503、
    新 exec 503，fail closed）。启动守卫校验 PID 1=本脚本、容器标记
    env、root euid、必需 capability、候选 uid 合法，任一失败即退出，
    因此 UID 扫描绝不可能在宿主机或错误命名空间运行。
  - 候选文件系统隔离（US-008）：候选会话一律经
    `worker/exec_trampoline.py`（uid 降落之后、候选命令之前）执行——
    Landlock ABI>=3（系统库只读执行、公共输入/数据只读、**仅当前任务
    scratch 根可写**，不授予共享 /tmp、/dev/shm 或 scratch 父目录），
    叠加窄 seccomp 过滤器对 Landlock ABI3 不覆盖的元数据变更族
    （chmod/chown/utime/xattr 共 18 个调用）返回 EPERM，x32 ABI 号与
    非 x86_64 架构直接 KILL；exec 前关闭继承描述符，HOME/TMPDIR 改为
    任务本地目录。trampoline 不可用（文件缺失/ABI 过低/路径逃逸）时
    在 exec 之前失败关闭；worker 启动守卫同样校验 trampoline 存在且
    ABI 达标。dispatcher 生成的可信 staging 会话（exec_class=staging，
    控制密钥认证）不走 trampoline；候选代码无法访问控制 API 伪造类别。
    跨任务读/写/chmod/utime/符号链接别名实测全部被拒
    （probe_us008.py；修复 .runtime/cross-job-boundary-review.json
    记录的实证漏洞）。DELETE 会话只在清理已证明时返回 200
    `cleanup_verified=true`，否则 503；dispatcher 的 kill/cleanup 消费
    该确认，超时/取消结果只在已证明时写 `completed_at`；会话失踪
    （404）不算清理证据。API 对仍在队列的取消用真实 Redis LREM 结果
    作为 `execution_never_started` 证据，运行中的任务绝不标记未开始。
  - 公共合成数据（US-008）：`tasks/hello_synth/data/public`（仅公开
    train/test/sample，私有答案与可信 metric 不挂载）以只读挂载到
    `/mnt/rsi_data/hello_synth/data/public`；任务提交
    `data_dir=/mnt/rsi_data/hello_synth`。
  - worker 无法解析/访问 postgres、redis、网关，无 DNS、无外网出口
    （`validate_stack.py` 实测探针证明）。
  - 注意：worker `cap_drop ALL` 后容器 root 没有 CAP_DAC_OVERRIDE，
    scratch 宿主目录必须 0777（init_auth.sh 已处理；validate 有真实写探针）。
- 评分：dispatcher 侧外部可信评测器（US-005），候选进程停止后从
  不可变 prediction_snapshot.csv 评分；worker 命令流不接触答案/评测代码。
- 源码身份绑定（batch-2 修复）：dispatcher 在 staging 之后、执行之前校验
  准入闸证据（`source_gate`：policy_version/policy_sha256/source 或
  entrypoint sha256），与当前加载策略和实际 staged 字节比对；缺失/不一致
  一律 `source_identity_failed`，不发出候选执行命令。
  回归测试：`research/tests/test_dispatcher_worker_control.py`。
- 统一租约：同一 worker 同时注册进 cpu/gpu 两个队列；dispatcher
  （WSL 副本）用 `worker_lease:<endpoint>` Redis SET NX 跨队列互斥，
  `_run_worker_job` 的所有出口（含 quarantine）都释放，dispatcher 启动时
  清理陈旧租约。回归测试：`research/tests/test_dispatcher_lease.py`。
- 执行标记（stage/exec/eval done markers）在 local-scratch 模式下落在
  scratch 树内，因此 worker 的 jobs 输入树可以保持只读。

## 凭据（不入 Git，不打印）

```bash
bash deploy/rsi-trustworthy/init_auth.sh   # 生成 auth.env 与 worker_control_key (0600)
```

`auth.env` 含 `SANDBOX_API_KEYS` / `POSTGRES_PASSWORD` / `DB_PASSWORD`，
只注入 api/postgres/dispatcher；`worker_control_key` 只挂载进 dispatcher
（直读）和 worker（supervisor 经 setuid 助手读取一次）。候选进程无任何
凭据。`.runtime/` 已 gitignore。

## 启动 / 停止（只影响新栈）

```bash
cd deploy/rsi-trustworthy
docker compose config --quiet
docker compose up -d
docker compose ps
docker compose down        # 只停新栈；不要加 -v，postgres 数据卷保留
```

## 验证

```bash
.venv-research/bin/python deploy/rsi-trustworthy/validate_stack.py   # 结构+隔离+认证+清理+旧栈健康
.venv-research/bin/python deploy/rsi-trustworthy/smoke_e2e.py        # 真实端到端冒烟（hello_synth）
.venv-research/bin/python deploy/rsi-trustworthy/probe_single_slot.py # cpu+gpu 并发提交证明单槽串行
.venv-research/bin/python deploy/rsi-trustworthy/probe_us008.py      # US-008 真实安全闭环探针（证据 reports/us008-security-loop.json）
```

## 限制

- edge 网络有出网能力，但只有 nginx 在其上；worker/DB/Redis 全在 internal 网络。
- US-009 第二轮真实模型小额验收已完成，证据见
  `reports/us009-round2-acceptance.json`；这不证明算法优越性。worker 无网络，
  requirements 安装会失败（fail closed，符合预期）。
- worker 健康检查端点 `/healthz` 无需认证（只返回 ok/cleanup_unproven，
  不暴露任何控制面）。

## 监督复核后的验收

证据见 `reports/trustworthy-stack-acceptance.json` 与 `reports/worker-gpu-validation.json`。
候选 UID65432/caps0 的真实 GPU 作业完成评分；CPU/GPU 作业执行区间不重叠；
源码入口字节与入队证据一致，结果含 `source_identity_verified` 和 `worker_cleanup_verified`。
新 scratch 使用 `jobs-candidate-v1` 子树，保留之前生成的目录。清理失败不报告完成，
也不释放执行槽；回收仅针对候选 UID 的被收养子进程，不使用全局 waitpid(-1)。

```bash
.venv-research/bin/python deploy/rsi-trustworthy/probe_gpu.py
```
