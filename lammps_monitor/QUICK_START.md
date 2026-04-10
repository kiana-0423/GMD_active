# 快速使用指南

## 项目已生成

完整的"纯 LAMMPS 在线检测与自动停机系统"已在以下位置创建：

```
/home/guozy/workspace/active_allegro/lammps_monitor/
```

## 文件清单

### 核心文件

| 文件 | 说明 |
|------|------|
| `README.md` | 完整项目文档（含详细配置指南、故障排除等） |
| `config.yaml` | 配置文件（阈值、路径、参数） |
| `inputs/in.monitor.lmp` | LAMMPS 输入脚本（完整注释） |

### Python 脚本模块

| 文件 | 说明 |
|------|------|
| `scripts/run_monitor.py` | 主控制器（**启动入口**） |
| `scripts/parse_log.py` | 日志解析器 |
| `scripts/utils.py` | 工具函数和数据类 |
| `scripts/extract_last_safe_frame.py` | 帧提取脚本 |
| `scripts/save_event_metadata.py` | 事件元数据管理 |

### 示例和其他

| 文件 | 说明 |
|------|------|
| `examples/monitor.log.example` | LAMMPS 输出日志示例 |
| `outputs/.gitkeep` | 输出目录占位符 |

## 🚀 快速开始 3 步

### 1️⃣ 修改 `config.yaml`

**必须修改的项**：

```yaml
lammps:
  executable: "lmp"  # 改成你的 LAMMPS 路径
  data_file: null    # 改成你的数据文件！

thresholds:
  displacement:
    soft: 2.0   # 根据体系调整
    hard: 5.0
  # ... 其他阈值 ...
```

### 2️⃣ 修改 `inputs/in.monitor.lmp`

**必须修改的部分**：

```lammps
# 【第一部分】基础设置
units real          # 改成你的单位制
atom_style full     # 改成你的原子风格

# 【第二部分】势函数 - 选择一个取消注释并修改！
pair_style lj/cut 10.0
pair_coeff 1 1 0.1 3.0

# 【第四部分】积分器 - 根据集合选择（NVE/NVT/NPT）
variable Tstart equal 300.0  # 改成你的温度
```

更多细节见 **README.md → 配置 LAMMPS 输入脚本** 章节。

### 3️⃣ 运行

```bash
cd /home/guozy/workspace/active_allegro/lammps_monitor

# 可选：先验证安装
python3 scripts/parse_log.py examples/monitor.log.example

# 运行监控系统（需先配置好 LAMMPS）
python3 scripts/run_monitor.py config.yaml
```

## 📊 系统功能总结

✅ **单步位移保险丝**（`fix dt/reset`）
- 自动调整 dt，防止数值爆炸
- 参数：dtmax = 0.002 ps，xmax = 0.1 Å（可调）

✅ **四层在线检测**
1. 位移异常（compute displace/atom）
2. 单原子势能异常（compute pe/atom）
3. 单原子动能异常（compute ke/atom）
4. 配位数变化（compute coord/atom，可选）

✅ **自动停机**（`fix halt`）
- 基于多个条件同时监控
- 硬停止 / 软警告两个阈值

✅ **完整日志记录**
- CSV 事件日志
- JSON 元数据
- JSONL 序列化
- 应急帧提取

✅ **Python 控制器**
- 自动调度 LAMMPS
- 实时日志解析
- 决策逻辑（normal / warning / abort）
- 结果保存和后续处理

## 🔧 关键参数解释

### config.yaml

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `dtreset.xmax` | 每步最大位移 | 0.05 ~ 0.2 Å |
| `dtreset.dtmax` | 最大时间步长 | 0.001 ~ 0.005 ps |
| `thresholds.displacement.hard` | 位移硬停止 | 3.0 ~ 10.0 Å |
| `thresholds.per_atom_pe.hard` | 能量硬停止 | 100 ~ 500 kcal/mol |
| `thresholds.per_atom_ke.hard` | 动能硬停止 | 20 ~ 100 kcal/mol |

> ⚠️ **注意**：所有阈值都应该根据你的体系用小规模测试先估算！建议从宽松阈值开始逐步调整。

## 🎯 典型应用场景

### 反应体系（如 ReaxFF）

```yaml
thresholds:
  displacement: {soft: 1.0, hard: 3.0}   # 易于突变，更严
  per_atom_pe: {soft: 50.0, hard: 150.0}
  per_atom_ke: {soft: 10.0, hard: 40.0}
```

### 高温金属体系

```yaml
thresholds:
  displacement: {soft: 3.0, hard: 8.0}   # 高温运动快，较宽松
  per_atom_pe: {soft: 100.0, hard: 300.0}
  per_atom_ke: {soft: 30.0, hard: 80.0}
```

### 蛋白质等生物分子

```yaml
thresholds:
  displacement: {soft: 2.0, hard: 5.0}
  coordination: {soft: 1.0, hard: 2.0}   # 监控二级结构
  per_atom_ke: {soft: 5.0, hard: 20.0}
```

## 📝 输出解释

运行完成后在 `outputs/` 中会生成：

1. **log.lammps** - LAMMPS 完整日志
2. **monitor_events.csv** - 汇总表（step, status, reason, 监控指标）
3. **event_metadata.json** - 最近事件的元数据
4. **dump.monitor.atom** - 完整轨迹（用于事后分析）
5. **last_safe_frame_run*.atom** - 异常时的最后帧（供人工检查或恢复运行）

## ⚡️ 进阶用法

### 提取指定帧

```bash
python3 scripts/extract_last_safe_frame.py outputs/dump.monitor.atom outputs/my_frame.atom
```

### 解析已有日志

```bash
python3 scripts/parse_log.py outputs/log.lammps
```

### 管理事件元数据

```bash
# 保存自定义事件
python3 scripts/save_event_metadata.py create 1 abort "pe_exceeded" 12345

# 查看事件
python3 scripts/save_event_metadata.py print outputs/event_metadata.json
```

### 干运行模式（仅验证不执行）

修改 `config.yaml`：

```yaml
run:
  dry_run: true
```

## ❓ 常见问题

### "找不到 LAMMPS"

→ 在 `config.yaml` 中指定绝对路径，如：`executable: /opt/lammps/src/lmp_serial`

### "数据文件找不到"

→ 确保路径在 `config.yaml` 和 `in.monitor.lmp` 中相对一致，或使用绝对路径

### 日志解析失败

→ 检查 `thermo_style custom` 是否包含了所有期望的列名（step, temp, press, ... v_maxdisp, v_maxpe, v_maxke, v_maxcn）

### 势函数配置不对

→ LAMMPS 会报错。必须在 `in.monitor.lmp` **【第二部分】** 中取消注释适合你的势函数并修改参数

## 📚 详细文档

详见 **README.md** 查看：
- 完整配置指南
- LAMMPS 输入脚本各部分详解
- Python API 文档
- 故障排除
- 扩展方法

## 💡 设计理念

- ✅ **纯 LAMMPS 原生命令**：无任何 ML / Deep Learning 依赖
- ✅ **模块化 Python**：易于扩展和集成到自动工作流
- ✅ **多层防护**：位移保险丝 + 多条件检测 + 硬停止
- ✅ **完整可追踪性**：每次异常都留下可分析的日志和帧

---

**项目位置**：`/home/guozy/workspace/active_allegro/lammps_monitor/`

**下一步**：修改 `config.yaml` 和 `in.monitor.lmp`，然后运行 `python3 scripts/run_monitor.py` 🚀
