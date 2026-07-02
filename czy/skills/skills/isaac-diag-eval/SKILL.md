---
name: "isaac-diag-eval"
description: "评估 Isaac 仿真诊断 CSV（isaac_diag_*.csv）某一轮训练相对上一轮是否改进。Exp1 标准数据分析工具，替代 czy/exp1/analyze_5_*.py。聚焦稳定性、关节高频抖动、限位 bang-bang、航向漂移、左右对称性、关节力矩、速度跟踪、落地冲击，以及 exp1 口径的 pos 范围/偏移/Welch HF%。当用户给出 isaac_diag 数据并问是否改进/分析结果/步态诊断时使用。"
---

# Isaac 诊断 CSV 改进评估

## Exp1 标准数据分析工具

**`czy/exp1` 后续实验（7.0 起）统一使用本 skill 分析 `isaac_diag_*.csv`，不再新建 `analyze_5_*.py` 一次性脚本。**

工作流：

1. 训练/回放后导出 CSV 到 `czy/data/isaac_diag_<时间戳>.csv`
2. 运行 `eval_isaac_diag.py`（见下方命令）
3. Agent 按本文件「分析报告模板」汇总结论
4. 将关键数值写回 `czy/exp1/实验记录/exp1.md` 对应实验的「实验结果」

与 `walk-diagnostics` 的分工：

| skill | 关注点 |
| --- | --- |
| `walk-diagnostics` | 对称性、跟踪误差、步态周期（稳态段统计） |
| **`isaac-diag-eval`（本 skill）** | **改进判定 + exp1 笔记指标**：覆盖度、稳定性、抖动、bang-bang、航向、速度/接触力、pos 范围与偏移 |

## 关键前提：先确认数据能验证什么

Isaac 诊断 CSV **新版已包含** base 线速度和接触力字段：
- `base_lin_vel_x/y/z`：机器人线速度（可验证速度跟踪）
- `left_foot_contact_force_z/mag`、`right_foot_contact_force_z/mag`：接触力（可验证落地冲击）
- `feet_contact_force_penalty`：接触力惩罚项（训练奖励值）

但**旧版本可能缺失**，且 `imu_accel_*` / `tau_des_*` 可能整列 NaN。因此：

| 实验目标 | 能否判定 |
| --- | --- |
| 稳定性 / 是否摔倒 | ✅ 能（base_euler + ang_vel） |
| 关节活动范围 / hip_roll 偏移 | ✅ 能（E1/E2，pos） |
| 位置域高频抖动（exp1 历史口径） | ✅ 能（E3，Welch pos >5Hz%） |
| 关节高频抖动（速度域，踝重点） | ✅ 能（第 3 节，vel FFT） |
| 限位 bang-bang 根因 | ✅ 能（pos_des_lpf/raw） |
| 航向漂移 | ✅ 能（base_euler_z） |
| 左右对称性 | ✅ 能（E1 L/R + 第 5 节 action） |
| **速度跟踪** | ✅ **能**（有 base_lin_vel_x） |
| **落地冲击** | ✅ **能**（有 contact_force） |

> ⚠️ 若 CSV 缺少线速度/接触力列，脚本第 0 节会自动报告缺失项。

## 分析脚本

脚本：`eval_isaac_diag.py`（与本文件同目录）。

```bash
# 从仓库根目录执行（Windows 建议先设 UTF-8）
$env:PYTHONIOENCODING='utf-8'
python czy/skills/skills/isaac-diag-eval/eval_isaac_diag.py czy/data/isaac_diag_<时间戳>.csv

# 与上一轮 CSV 对比（替代 analyze_5_8.py 第 9 节）
python czy/skills/skills/isaac-diag-eval/eval_isaac_diag.py czy/data/isaac_diag_new.csv `
  --compare czy/data/isaac_diag_old.csv --compare-label "5.8"

# 可调参数
python czy/skills/skills/isaac-diag-eval/eval_isaac_diag.py <csv> --win 1.0 --hf 5.0
```

### 脚本输出章节

| 节 | 内容 | 写回 exp1 笔记 |
| --- | --- | --- |
| 0 | 数据覆盖度 | 可选 |
| cmd | 命令速度/角速度 | 可选 |
| **E1** | 关节 pos 实际范围 + L/R 比 | **必引**（膝/髋范围） |
| **E2** | pos 均值对称性 + hip_roll/yaw 详情 | **必引**（hip_roll 偏移） |
| 1 | 稳定性分窗 + 摔倒判定 | 推荐 |
| 2 | 航向漂移 | 推荐 |
| 3 | 速度域高频抖动（vel FFT） | 辅助（与 E3 不可横比） |
| **E3** | 位置域 Welch HF%（exp1 口径） | **必引**（>5Hz 能量%） |
| E4 | PD 跟踪率 | 可选 |
| E5 | pos_des_raw 范围 | 可选 |
| 4 | 限位 bang-bang | 根因分析时引用 |
| 5 | action 对称性 L/R 比 | 推荐 |
| 6 | 关节力矩 | 可选 |
| 7 | 速度跟踪 | **必引** |
| 8 | 落地冲击 | **必引** |
| E6 | `--compare` 对比表 | 有上一轮 CSV 时使用 |

### 两套抖动指标（勿混用）

| 指标 | 章节 | 信号 | 阈值（经验） | 用途 |
| --- | --- | --- | --- | --- |
| HF% | **E3** | pos + Welch | < 5% OK | **与 exp1 5.7/5.8 历史记录对比** |
| HF 占比 | 第 3 节 | vel + rfft | 踝 < 0.5 OK | 改进判定、踝关节根因 |

### 可配置项（脚本顶部常量）

- `HARD_LIMITS`：踝关节硬限位，用于 bang-bang 判定
- `LEG_JOINTS` / `ANKLE_JOINTS` / `SYM_BASES`：关节命名

## 评估标准

### E1/E2 关节范围与对称性（exp1 口径）
- 膝/髋 pitch range：双侧 > 0.5 rad 为达标参考
- 左右 range 比 L/R：✅ 0.85~1.18 / ⚠️ 0.7~1.4 / ❌ 超出
- hip_roll/yaw `|mean|`：✅ < 0.05 rad / ⚠️ < 0.15 / ❌ 更大

### E3 位置域抖动（Welch pos >5Hz%）
- ✅ < 5%
- ⚠️ 5%~10%
- ❌ > 10%

### 1. 稳定性（摔倒判定）
- ✅ `max|roll|` 且 `max|pitch|` < ~0.35 rad
- ❌ 超 0.6 rad → 疑似摔倒

### 2. 航向漂移（直行时）
- ✅ `|yaw 漂移|` < 0.15 rad
- ❌ > 0.4 rad → 明显跑偏

### 3. 速度域抖动（踝关节为重点）
- ✅ 踝 HF 占比 < 0.5
- ❌ > 0.7 → 严重

### 4. 限位 bang-bang
- ❌ des 顶限位 > 30% → 抖动常见根因

### 5. action 对称性 L/R 比
- ✅ 0.85~1.18

### 7. 速度跟踪
- ✅ ≥ 90% / ⚠️ 75%~90% / ❌ < 75%

### 8. 落地冲击
- ✅ 峰值 < 150N / ⚠️ 150~300N / ❌ > 300N

## 改进判定逻辑（给 Agent）

1. **先跑脚本，读第 0 节**：明确可验证项。
2. **写 exp1 笔记时优先引用 E1/E2/E3、第 7/8 节**；与历史实验对比用 E3 HF% 和 E1 range，不用第 3 节 vel HF。
3. **有上一轮 CSV 时加 `--compare`**，直接出 E6 对比表。
4. **改进 = 目标项变好且无新退化**；结论分级：✅改进 / ⚠️部分改进 / ❌未改进。
5. **结论写回** `czy/exp1/实验记录/exp1.md` 对应实验 **§7 实验结果**（结构见 `lab-notebook` skill）。

## 分析报告模板

```markdown
## Isaac 诊断分析报告：实验 <编号>

> 数据：`isaac_diag_<时间戳>.csv`（<时长>s，<采样>Hz，cmd=<速度> m/s）
> 工具：`isaac-diag-eval` / `eval_isaac_diag.py`
> 本轮目标：<目标1> / <目标2>

### 关键指标（exp1 口径）
| 指标 | 数值 | 判定 |
| --- | --- | --- |
| 左/右膝 range | … / … | … |
| 膝 L/R 比 | … | … |
| hip_roll 偏移 | … | … |
| 左膝 HF% (E3) | …% | … |
| 速度跟踪 | …% | … |
| 落地冲击峰值 | …N | … |

### 逐目标结论
| 目标 | 判定 | 依据 |
| --- | --- | --- |
| 对称性 | ✅/⚠️/❌ | E1/E2/E5 … |
| 抖动 | ✅/⚠️/❌ | E3=…%, 踝 vel HF=… |
| 速度/冲击 | ✅/⚠️/❌ | 第 7/8 节 … |

### 与上一轮对比（如有 --compare）
| 指标 | 上轮 | 本轮 |
| --- | --- | --- |
| … | … | … |

### 总体判断
✅改进 / ⚠️部分改进 / ❌未改进 —— 一句话理由

### 下一轮建议
1. …
```

## 注意事项

1. **不臆测缺失数据**；第 0 节会列出不可验证项。
2. **E3 与第 3 节抖动不可横比**；历史笔记对照只用 E3。
3. **Windows**：`$env:PYTHONIOENCODING='utf-8'` 避免 emoji 编码错误。
4. **勿再新建 analyze_5_*.py**；新需求沉淀进 `eval_isaac_diag.py`。
