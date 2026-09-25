# 因果 v3 改动审查

> 收尾复核通过：200 项测试通过，本轮列出的 P1/P2/P3 均已关闭；双进程互斥及标记丢失后的诊断恢复均已验证。此结论仅说明本轮工程问题修复完成，当前两次研究仍未通过，没有合格候选。

审查对象：基于 `97ed4d5` 的未提交改动，包括 `v3_data.py`、`v3_execution.py`、`v3_search.py`、`v3_pipeline.py`、`v3_artifact.py`、AlphaGPT 可选参数及新增测试。执行 `.venv/bin/python -m pytest -q`：**174 passed**，另有依赖弃用和训练标签除零告警。

结论：离线流程已经接通，但还不能视为修复完成。以下问题需要处理，其中 P1 会改变训练奖励与退出路径。当前“没有合格候选、不能据此上线”的判断有必要保留；修复后需重新评估，不应继续引用本次收益数字作为新实现的结果。本审查未修改实现、交易开关、策略文件或服务。

## 1. [P1] 训练标签使用 USD，组合回测与实盘使用 SOL

位置：`model_core/v3_execution.py:195-196`、`:242`、`:255-256`。

`trade_labels()` 根据美元价格计算持币数量、收益和止损/止盈条件，完全没有 SOL/USD 变化。`simulate_portfolio()` 在第 393、414、439 行则按当时 SOL 汇率计算。这会同时改变收益和退出时间，训练实际优化了另一个目标。

已复现：币价美元计价不变，SOL 从 100 USD 升至 106 USD。同一入场的训练标签为 **0% 收益、第 71 分钟 TimeExit**；组合模拟为 **-5.6604%、第 21 分钟 StopLoss**。现有一致性测试主动将 SOL/USD 固定，所以没有覆盖这个问题。

建议：标签复用与组合模拟一致的入场、监控、卖出 SOL 时点换算和缺失/过期判断；加入 SOL 上涨、下跌、报价缺失时标签与单笔模拟一致的测试。随后重新训练和筛选。

## 2. [P2] 冻结数据指纹遗漏实际特征输入

位置：`model_core/v3_data.py:42-50`；最终检查在 `model_core/v3_pipeline.py:367-368`。

指纹包括 `prices.open/close`，却遗漏因子使用的 `raw`，尤其 high、low、volume；raw.open/close 与 prices 的一致性也未验证。因此特征数据改变后，最终测试仍会认定它是原来冻结的数据。

已复现：把最后 120 列 volume 乘 1000，数据指纹不变，但 `[5]` 公式的原始输出最大变化 **4.664**。

建议：哈希全部实际输入及其结构，并校验观测格子的 raw 与成交价格一致性；测试任一 OHLCV 字段改变都能使最终测试拒绝运行。

## 3. [P2] 独立 CLI 可能忽略 `.env` 中的数据库配置

位置：`model_core/v3_data.py:169-172`；提前导入路径为 `v3_pipeline → v3_search → alphagpt → ModelConfig`。

`ModelConfig.DB_URL` 在导入时固定，`load_dotenv()` 执行得更晚。如果 DB_* 只存在于 `.env` 而没有预先导出到 shell，文档中的独立 search 命令可能使用默认连接而非用户配置。仅调换 loader 内两行代码不能修复更早的模块导入。

已用无网络、无秘密的子进程 mock 复现：dotenv 设置指定主机后，`create_engine` 仍收到 localhost。

建议：加载 dotenv 后按当前环境动态构建数据库 URL；新增干净子进程测试，不依赖整套 pytest 中其他模块提前加载 dotenv。

## 4. [P2] 期末平仓绕过 SOL 汇率时效限制

位置：`model_core/v3_execution.py:514-516`。

这里获得 `(sol, age)` 后不检查 age。普通入场、监控和退出拒绝过期汇率，期末结算却直接使用它，将无法可靠估值的仓位记成成功平仓。

已复现：SOL 行情在第 20 分钟停更，持仓保留至第 120 分钟；过程记录 **70 次 monitor_skipped_no_sol_price**，最后仍用 **6000 秒前**的汇率记为 PeriodEndClose、回收 1 SOL，而上限是 1800 秒。

建议：期末应用相同的汇率时效规则，明确无法估值时的保守结算/未结算状态，并覆盖对应测试。

## 5. [P2] 最终测试未使用冻结的验收规则与基线

位置：`model_core/v3_pipeline.py:328-341`、`:394`。

`selection.json` 保存了 acceptance_criteria 和 baseline_formulas，但最终验收读取的是当前模块全局 ACCEPTANCE、BASELINE_FORMULAS；压力参数也未一起冻结校验。搜索完成后修改这些常量，配置摘要和版本检查不会发现，却会改变最终通过条件。

已用现有结果只在内存复现：冻结要求至少 30 次入场、实际 16 次；将当前全局要求改为 1，该项从失败变为通过，runtime_versions 不变。

建议：最终阶段读取冻结副本，或校验包含规则、基线、压力参数的完整摘要。代码规则变化应要求新的研究记录，不得静默沿用旧的“预注册”声明。

## 6. [P2] 失败重跑可能残留此前的合格候选

位置：`model_core/v3_pipeline.py:403-425`。

代码将 `--acknowledge-rerun` 的结果强制设为 passed=False，但只处理成功时写候选，没有处理旧候选的作废。若首次通过后重跑，`candidate_causal_v3.json` 仍留在目录中，新的报告却写“未生成”，后续工具或人工可能误用旧产物。

建议：在已消费测试的重跑或失败结果中明确作废/隔离旧产物，保留审计记录。补充“首次通过生成产物 → 重跑失败 → 旧产物不可再当作当前合格候选加载”的测试。

## 7. [P2] 币种排名子查询没有过滤无效及测试 K 线

位置：`model_core/v3_data.py:195-198`。

启用 `--max-tokens` 后，子查询用所有行数排名，外层才过滤无效 K 线及 `source='test'`。测试数据或无效行很多的币可占满名额，随后被外层剔除，导致明明存在有效行情却加载出更少币甚至空数据。

建议：排名子查询和最终加载使用相同的有效性及来源条件，注意将 SQL 别名从 o 换成 o2；补充受污染币行数更多的边界测试。

## 报告文案也需同步

- `docs/model-v3-evaluation-2026-09-25.md:7` 称 4332 个公式没有一个在验证段盈利；实际仅对训练排名前 20 个执行了验证，结论应限定为这 20 个。
- 第 110 行仍称测试段是未见数据，与第 73 行披露已重跑矛盾。
- 成本样本前文为 46 条，限制部分却为 41 条，应采用冻结 selection 的统计。

建议先处理 1、3，保证训练目标和独立执行正确；随后处理 2、4、5、6、7，再重跑有限预算研究。已看过的最终测试段不能因为换目录或修改代码就重新声称是未见数据。

## 处理结果（2026-09-25 续）

上述 7 项与报告文案问题均已处理。每项都有对应的回归测试，全量 `pytest` 结果见 `docs/model-v3-evaluation-2026-09-25.md`。仍未修改交易开关、实盘策略文件或服务。

| # | 处理 | 测试 |
| --- | --- | --- |
| 1 | `trade_labels()` 改用与 `simulate_portfolio()` 相同的 SOL 记账：成交 K 线开盘时点换算买入和卖出，每根收盘时点做监控换算，汇率时效使用同一 `sol_price_max_age_seconds`。缺少新鲜汇率时，入场不成交，监控跳过，待成交退出顺延到下一根，与组合模拟一致。持仓步数上限放宽到 2×最长持有；仍未了结的仓位剔除，不作猜测。 | `test_labels_use_sol_accounting_like_the_portfolio`（复现 SOL 100→106：两者均为第 21 分钟 StopLoss、−5.66%；另覆盖 SOL 下跌和持平）、`test_labels_follow_missing_sol_rates_like_the_portfolio`（覆盖入场无汇率、监控跳过、退出顺延），逐笔一致性测试新增 SOL 变动和 SOL 缺口两种情形 |
| 2 | 指纹纳入全部 `raw`（OHLCV）、`prices`、SOL 序列，以及每个数组的名称、dtype 和 shape。新增 `check_consistency()`：观测格子的 raw.open/close 必须等于成交价格的 float32 映像，未观测格子必须为 0，字段必须齐全。载入数据和最终测试都会执行该检查。 | `test_fingerprint_covers_every_ohlcv_input`、`test_consistency_check_rejects_raw_price_mismatch`、`test_final_test_refuses_changed_feature_inputs` |
| 3 | 改为在 `load_dotenv()` 之后由 `database_url()` 按当前环境构建连接串，不再读取导入时就固定下来的 `ModelConfig.DB_URL`。 | `test_standalone_cli_uses_database_settings_from_dotenv`：在干净子进程中先导入 `v3_pipeline`，并剔除 DB_* 环境变量 |
| 4 | 期末估值应用同一汇率时效规则。超时或缺失时记为 `unvalued_no_sol_price`（`PeriodEndUnvalued`），保守按 0 计，不算成功平仓。 | `test_period_end_close_refuses_stale_sol_rate` |
| 5 | `selection.json` 冻结 `rules`，包括验证门槛、验收标准、基线公式和压力参数，并保存 `rules_digest`。最终测试只读取冻结副本；若代码中的规则摘要已变化，则拒绝运行并要求开启新的研究记录。 | `test_final_test_refuses_rules_changed_after_selection`（分别修改验收、基线、压力参数）、`test_acceptance_uses_frozen_copy` |
| 6 | 每次最终测试开始时，若目录中已有 `candidate_causal_v3.json`，就将其移入 `revoked/`，打上 `revocation` 标记（原文件 sha256），并追加 `artifact_revocations.jsonl` 审计记录。`load_v3_candidate` 拒绝已作废产物，只有本次通过时才重新生成候选。 | `test_failed_rerun_revokes_earlier_candidate` |
| 7 | 排名子查询使用与外层相同的有效性和来源条件，别名为 `o2`，由 `valid_candle_sql(alias)` 和 `candle_query_sql()` 生成。 | `test_token_ranking_ignores_test_and_invalid_rows`：用 sqlite 执行同一 SQL，受污染币的行数更多 |

另外针对“换目录或改代码后又声称测试段未见”的问题，新增跨目录账本 `logs/research-v3-consumed-test-windows.jsonl`。每次最终测试都会追加自己的窗口；之后同一数据源、测试窗口重叠的运行，搜索阶段会发出警告，最终测试强制不通过，报告也会注明。账本已补记 `research-v3-01` 的测试窗口（09-22 11:16 → 09-24 19:16 UTC）。测试：`test_new_directory_cannot_reclaim_a_consumed_test_window`。

## 二次复核（2026-09-25）

已确认原 7 项修复有效，相关回归测试覆盖了原问题。全量测试 **186 passed**，先前训练标签除零告警也已消失。本轮未修改实现或实盘状态。

`research-v3-02` 顶层结果仍为 `passed=false`，其窗口被判定与此前测试重叠，没有生成合格候选。以下 4 项应继续修复；其中第 8 项已出现在实际结果文件，第 9–11 项在指定边界场景复现。

### 8. [P2] 验收布尔值序列化为字符串，重新读取会改变含义

位置：`model_core/v3_artifact.py:111`；布尔值来源包括 `model_core/v3_pipeline.py:430-435`。

`write_json_atomic()` 的 `json.dump(..., default=str)` 会把 NumPy 布尔值写成字符串。实际 `logs/research-v3-02/test_results.json` 中，四项负收益检查均为 `"False"`，回撤检查为 `"True"`。JSON 加载后 `bool("False")` 为 True，按布尔值统计只能识别 3 项失败，实际应为 7 项。顶层 passed 是原生布尔值，当前仍正确为 false，不能据此声称发生了自动上线。

最小复现：`write_json_atomic(path, {"check": np.bool_(False)})` 后加载得到 `{"check": "False"}`。

建议：递归将 NumPy 标量转换为原生类型，或在生成 checks 时显式 `bool()`；不支持的类型应报错，不应统一转字符串。补充 JSON 往返测试，断言所有检查项加载后仍是 bool 且真假不变；修复现存结果文件应注明为序列化修正，不能伪装成新的未见测试。

### 9. [P2] 更换输出目录的父目录会绕开已消费窗口账本

位置：`model_core/v3_pipeline.py:155-158`。

默认账本位于 `config.out.parent`，因此它只跨同一父目录的运行共享。已复现：`runs/first` 消费窗口后，`runs/second` 能找到 1 项重叠；改为 `runs/group/third` 就找到 0 项，因为账本路径变了。这与文档中“不能靠换目录绕过”的保证不符，整理运行目录就可能意外把旧测试当作新测试。

建议：固定项目/研究数据源级账本位置，与输出目录解耦；将账本身份纳入冻结配置或显式迁移，并覆盖不同父目录与移动目录的测试。

### 10. [P2] 并行最终测试的检查与登记不是原子操作

位置：`model_core/v3_pipeline.py:486-490`，通过条件在第 526 行。

`consumed_overlaps()` 和 `record_consumed()` 分开执行且没有锁。两个不同运行目录并行评估同一窗口时，可能都先读到空历史，再分别追加，双方都得到 `earlier=[]`，失去阻止重复使用测试集的作用。

已用临时目录和两个线程同步复现：两边重叠计数均为 0，未消费门槛均放行。没有修改实际研究账本或运行训练。

建议：使用文件锁或 SQLite 事务，将“检查冲突 + 认领窗口”做成一个原子操作；必须在任何测试评估前完成认领。增加并发回归测试，确保重叠窗口至多一个运行能取得未消费资格。

### 11. [P2] 限币选择截止点与最终训练截止点可能不一致

位置：`model_core/v3_data.py:294-297`、`model_core/v3_pipeline.py:138-142`。

`--max-tokens` 按请求的完整范围计算 selection_end，按该时间以前的行数选币；返回选中币后，`compute_splits()` 又按这些币实际的最早/最晚时间重新计算 train_end。若选中币较早停更，实际训练结束会提前，选币排名就使用了实际训练结束以后的数据。

已用 SQLite 和实际切分函数复现：请求 10000 分钟，选择截止为第 6000 分钟。A 在前 5000 分钟有 4999 根，B 在前 3000 分钟有 3000 根；限选一个币会选 A。加载后的数据到第 5000 分钟结束，实际训练截止变成第 3000 分钟；在这个真正训练段内 A 只有 2999 根、B 有 3000 根，选 A 的依据来自训练段之后。

建议：选币前冻结一组时间边界，并让数据加载、切分、标签、报告统一使用；或确保排名只读取最终冻结 train_end 以前的数据。补充选中币提前停更/上市较晚的测试。本次 v3-02 使用 max_tokens=None，因此这项不直接影响其已报告结果。

## 二次复核处理结果（2026-09-25）

第 8–11 项均已修复，并有对应的回归测试。全量 `pytest` **193 passed**，`git diff --check` 无问题。本轮未修改交易开关、实盘策略文件或服务，也没有重新运行训练或最终测试。

| # | 处理 | 测试 |
| --- | --- | --- |
| 8 | `write_json_atomic` 不再使用 `default=str`。改为严格转换：NumPy bool/int/float/数组和 Path 转为原生 JSON 类型，其他类型直接报 `TypeError`。`_criteria_results` 另外显式 `bool()`。已有的 `research-v3-01/02` 的 `test_results.json` 中，5 个字符串检查项已改回布尔值，文件内附 `serialization_correction`（修正字段、说明、原文件 sha256）。这只是序列化修正，没有重新评估，也不是新的测试运行。修正后两次运行都是 7 项失败，`passed=false` 不变。 | `test_numpy_values_round_trip_as_native_json`、`test_final_test_checks_reload_as_booleans` |
| 9 | 账本固定为项目级 `<repo>/logs/research-v3-consumed-test-windows.jsonl`（`DEFAULT_LEDGER`），与 `--out` 无关；可用 `--consumed-ledger` 显式指定。解析后的账本路径和随机 `run_id` 会冻结进 `selection.json`，最终测试使用冻结的账本路径，因此移动或重组运行目录不会换到新账本。同一次运行的重跑按 `run_id` 识别；旧 selection 没有 `run_id` 时按路径识别。 | `test_other_parent_directories_cannot_reclaim_a_consumed_test_window`（`runs/first` → `runs/group/second`、`elsewhere/deeper/third`）、`test_moved_run_directory_keeps_its_frozen_ledger` |
| 10 | 新增 `claim_test_window()`：在 `<ledger>.lock` 的 `fcntl.flock` 排他锁内完成“读取重叠 + 追加认领”，并 fsync。认领发生在任何测试段模拟之前。 | `test_concurrent_claims_admit_at_most_one_unconsumed_run`（双线程 + 放大竞争窗口；已验证去掉锁时该测试失败）、`test_own_rerun_is_not_counted_as_another_run` |
| 11 | 请求的时间范围（SOL 截断后）在选币前就冻结为 `meta["split_range"]`，选币截止时间由同一个函数 `train_end_for_range()` 计算，写入 `meta["selection_end_epoch"]`。`compute_splits` 按冻结范围切分，不再按已加载币的首尾 K 线切分；`Context` 会校验选币截止时间等于训练截止时间，不一致则拒绝运行。 | `test_token_selection_cutoff_is_the_training_cutoff`（复现审查场景：训练截止保持在第 6000 分钟）、`test_context_refuses_mismatched_selection_cutoff` |

另外，第 9 项修复后，两个旧测试曾向项目真实账本写入 7 条合成数据记录（`source=synthetic`，2025 年窗口）。这些记录已删除，账本恢复为 v3-01、v3-02 两条真实记录。测试模块现在由 `setUpModule` 把默认账本重定向到临时目录，每个测试另有独立账本。

## 三次复核（2026-09-25）

已独立确认全量 **193 passed**、`git diff --check` 通过。测试前后真实账本 SHA-256 相同，账本仅含 v3-01、v3-02 两条 Postgres 记录。两个 `test_results.json` 的全部 checks 均为原生布尔值，各 7 项失败，顶层 passed=false；序列化修正说明及原文件哈希齐全，两个目录均无候选产物。

第 8 项可关闭；第 9 项固定项目账本及冻结路径的修复有效。第 10 项不同 run_id 并发登记、第 11 项正常数据库加载路径也已修复，但以下两个边界仍需处理。本轮没有修改实现、真实研究结果或实盘。

### 10a. [P2] 同一运行并发最终测试仍可重复获得未消费资格

位置：`model_core/v3_pipeline.py:504`、`:212`、`:574`。

`rerun = marker.exists()` 在锁外读取；`claim_test_window()` 又排除了相同 run_id 的既有登记。因此同一输出目录的两个并行 `final-test` 可以都先得到 rerun=False，然后在锁内先后登记时仍都得到 earlier=[]。若其他验收项通过，两个进程都能生成通过结果。

已在临时目录中完整调用实际 `command_final_test` 复现：使用模拟数据/Context 和模拟通过的验收结果（未训练），以双进程屏障确保两边 marker 预检查完成。两进程均返回 `passed=True, test_rerun=False, earlier=0`，账本两行，候选文件存在。该复现只证明门槛漏洞，不代表有策略通过实际收益验证。

建议：对同一运行加覆盖整个最终测试的互斥锁，并在锁内重读 marker；或让原子 claim 返回“本 run 已消费”状态并拒绝未确认重复执行。测试需覆盖相同 run_id/相同输出目录的双进程竞争，不能只覆盖两个不同 run_id。

### 11a. [P2] 新的切分元数据没有纳入冻结校验

位置：`model_core/v3_pipeline.py:311-315`、`model_core/v3_data.py:58-67`；最终校验入口 `model_core/v3_pipeline.py:519-526`。

`Context` 现在依赖 `meta.split_range` 和 `meta.selection_end_epoch` 决定训练/验证/测试边界，但数据指纹仍仅覆盖数组，不覆盖这两个字段。同步修改两项后，它们彼此一致，Context 检查会通过；指纹和配置摘要也完全不变。最终测试没有独立比较原先冻结的切分定义。

已在内存快照复现：仅将 split_range 结束时间提前 36000 秒，并同步更新 selection_end_epoch；数据一致性检查和指纹都通过。测试范围随之提前，与原验证段重叠 2544 秒。

建议：将所有会影响切分的元数据纳入指纹，或在 selection 中冻结完整切分并于最终阶段逐项核对；同时覆盖元数据同步变化的回归测试。缺少这些字段的旧限币快照应根据原请求范围迁移或拒绝，不能无提示回落到已加载币的首尾范围。

## 三次复核处理结果（2026-09-25）

10a、11a 均已修复。全量 `pytest` **199 passed**，`git diff --check` 通过；测试前后真实账本 SHA-256 不变（`35e2095f…`，仍为 v3-01、v3-02 两条记录）。本轮未修改真实研究结果或实盘状态，也没有重新运行训练或最终测试。

| # | 处理 | 测试 |
| --- | --- | --- |
| 10a | 两层防护。① `command_final_test` 在整个最终测试期间持有运行级排他锁（`<out>/.final_test.lock`，`fcntl.flock` 非阻塞），`test_consumed.json` 标记在锁内读取；已有最终测试在运行时，第二个立即以 “another final test is running” 退出，不会等待后变成重跑。② `claim_test_window()` 在账本锁内检查本运行（按 `run_id`，旧 selection 按目录）是否已认领重叠窗口；若已认领且未确认重跑，就拒绝认领。因此即使标记丢失或被删除，也不会出现第二次“首次测试”。 | `test_concurrent_final_tests_on_one_run_admit_only_one`：同一输出目录双线程并发执行真实的 `command_final_test`，且模拟验收全部通过；结果一个完成、一个被拒，账本 1 行，`previous_runs=0`。`test_lost_marker_cannot_make_a_second_first_test`、`test_own_rerun_is_not_counted_as_another_run` |
| 11a | ① 指纹纳入 `split_range`、`selection_end_epoch`、`max_tokens`。只要其中任一字段存在就参与哈希；所有当前加载器都会写入 `split_range`，所以同步修改或删除都会改变指纹。从未有这些字段的旧快照和合成数据，指纹保持不变。② `selection.json` 冻结完整的 train/validation/test 边界（`splits`），最终测试逐项比对，不一致即拒绝。③ 限币数据（`max_tokens` 非空）若缺少 `split_range`，`Context` 直接拒绝，不再回落到已加载币的首尾范围。 | `test_fingerprint_covers_split_metadata`（同步修改、逐项删除）、`test_final_test_refuses_synchronised_split_metadata_edit`（复现审查场景：结束时间提前 36000 秒并同步更新截止时间 → 指纹拒绝，且未登记账本）、`test_final_test_refuses_changed_split_boundaries`、`test_token_limited_dataset_without_split_range_is_refused` |

对已有运行的影响：v3-01、v3-02 的 selection 没有 `run_id` 和 `splits`，快照也没有切分元数据，因此指纹不变。若再对其执行 final-test，账本会按目录识别出它已消费过，必须加 `--acknowledge-rerun`，且结果不能通过。

## 四次复核（2026-09-25）

原 10a、11a 可关闭。独立执行全量测试 **199 passed**，`git diff --check` 通过。测试前后真实账本 SHA-256 一致，仍仅有 v3-01、v3-02 两条；两个结果文件均为 passed=false、7 项失败，无候选产物。

补充双进程验证：临时目录中调用实际 `command_final_test`，使用模拟数据和模拟验收结果。第一进程持锁，第二进程立即收到 `another final test is running`；第一进程完成，账本仅 1 条登记。切分元数据指纹、冻结边界对比及旧限币快照拒绝路径也已核对。本轮未改变真实研究结果或实盘状态。

### [P3] 标记丢失后，显式确认仍无法进行诊断重跑

位置：`model_core/v3_pipeline.py:543`、`:577-578`。

账本已有本运行的记录，但 `test_consumed.json` 丢失时，即使调用 `command_final_test(..., acknowledge_rerun=True)`，局部 rerun 仍因 marker 不存在而为 False；传给 claim 的也是这个 False。结果再次拒绝，并提示添加已经提供的 `--acknowledge-rerun`。已在临时目录复现：先运行一次，仅删除临时 marker，再带确认参数调用，仍被拒绝，账本保持 1 行。

该行为继续阻止重复取得首次测试资格，不影响当前真实结果，也不是并发放行漏洞；但文档承诺的诊断恢复路径无法使用。

建议：原子登记接口同时返回“本运行此前已消费”状态，将它与 marker 状态合并为最终 rerun；确认参数只授权诊断重跑。补充“marker 丢失 + 已有账本 + 明确确认”测试，断言可以完成诊断但 passed 必须为 False，旧候选必须作废。不能只允许 claim 忽略 own 记录而仍令最终 rerun=False。

## 四次复核处理结果（2026-09-25）

P3 已修复。全量 `pytest` **200 passed**，`git diff --check` 通过；测试前后真实账本 SHA-256 不变（`35e2095f…`）。未修改真实研究结果或实盘状态。

- `claim_test_window()` 在账本锁内返回 `(earlier, own)`：`own` 是本运行此前对重叠窗口的认领（按 `run_id` 判断，旧 selection 按目录）。参数改名为 `acknowledge_rerun`，只表示用户确认过，不再冒充 marker 状态。
- `_final_test_locked` 的最终 `rerun = marker 存在 or own 非空`，以账本为准。`--acknowledge-rerun` 只允许诊断重跑：结果中 `test_rerun=True`，`passed` 必为 False，旧候选照常作废，报告注明测试段不再是未见数据，`previous_runs` 按 marker 或账本中本运行的认领次数计算。自身的旧认领不计入 `test_window_previously_consumed`，那里只列其他运行。
- 未确认时，marker 丢失仍会被拒绝，行为不变。
- 测试：`test_lost_marker_allows_acknowledged_diagnostic_rerun_that_cannot_pass`。流程为模拟首测通过、生成候选 → 删除 marker → 带确认参数重跑，断言诊断完成、`passed=False`、候选移入 `revoked/`、`previous_runs=1`、账本 2 行。`test_own_rerun_is_not_counted_as_another_run` 同步改为校验 `(earlier, own)`。

独立收尾复核：已检查上述代码与完整回归流程，再次执行全量测试得到 **200 passed**；`git diff --check` 通过。测试前后真实账本 SHA-256 相同，仍仅有 v3-01、v3-02 两条记录。两个真实结果均为 passed=false、7 项失败，目录中无候选产物。该 P3 可关闭，本次修复范围内未发现新的遗留问题。未重新评估真实研究数据、操作实盘或提交代码。
