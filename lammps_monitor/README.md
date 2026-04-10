# LAMMPS 在线检测与自动停机系统

完全基于 LAMMPS 原生命令的在线监控和自动停机系统，适用于反应型体系、高温模拟和可能发生结构突变的分子动力学。

## 概述

本系统在 LAMMPS 运行过程中实现以下功能：

1. **单步位移限制**（`fix dt/reset`）：防止时间步长过大导致数值爆炸
2. **位移异常检测**（`compute displace/atom`）：监控原子累计位移
3. **单原子势能 / 动能异常检测**（`compute pe/atom`, `compute ke/atom`）
4. **配位数变化检测**（`compute coord/atom`）：捕捉局部结构变化
5. **在线停机**（`fix halt`）：基于多个阈值条件自动停止
6. **完整日志记录和转储输出**
7. **Python 外层控制器**：自动调度、分析、决策和保存

## 特点

- ✅ **纯 LAMMPS 原生命令**：不依赖 DeePMD、ML 不确定性模块或其他外部力场
- ✅ **模块化 Python 代码**：易于扩展和定制
- ✅ **灵活配置**：所有参数都在 `config.yaml` 中，易于调整
- ✅ **多层位移保险丝**：时间步长自适应 + 硬停止条件
- ✅ **事件日志**：CSV、JSONL、JSON 格式，便于后续分析
- ✅ **应急帧提取**：异常时自动保存最后安全帧
- ✅ **完整错误处理和报警机制**

## 项目结构

```
project/
├── README.md                          # 本文件
├── config.yaml                        # 配置文件（用户需自行修改）
├── inputs/
│   └── in.monitor.lmp                 # LAMMPS 输入脚本
├── scripts/
│   ├── run_monitor.py                 # 主控制脚本（启动入口）
│   ├── parse_log.py                   # 日志解析器
│   ├── extract_last_safe_frame.py     # 帧提取脚本
│   ├── save_event_metadata.py         # 事件元数据脚本
│   └── utils.py                       # 工具函数
├── outputs/                           # 运行输出目录（自动创建）
│   ├── log.lammps                     # LAMMPS 标准日志
│   ├── dump.monitor.atom              # 原子转储
│   ├── monitor_events.csv             # 事件日志（CSV）
│   ├── monitor_events.jsonl           # 事件日志（JSONL）
│   ├── event_metadata.json            # 最近事件的完整元数据
│   ├── monitor_trend.csv              # 监控趋势表
│   └── last_safe_frame_run*.atom      # 异常时保存的最后安全帧
└── examples/
    └── monitor.log.example            # 示例日志
```

## 快速开始

### 1. 准备

确保已安装：
- LAMMPS（可从 PATH 访问或指定绝对路径）
- Python 3.7+
- PyYAML：`pip install pyyaml`

### 2. 配置体系

编辑 `config.yaml`：

```yaml
# LAMMPS 执行和文件路径
lammps:
  executable: "lmp"  # 或 "/path/to/lmp"
  input_file: "inputs/in.monitor.lmp"
  log_file: "outputs/log.lammps"
  dump_file: "outputs/dump.monitor.atom"
  data_file: "data/your_system.data"  # 指定你的数据文件

# 调整阈值（每个体系不同）
thresholds:
  displacement:
    soft: 2.0   # 警告
    hard: 5.0   # 停止
  per_atom_pe:
    soft: 50.0
    hard: 200.0
  per_atom_ke:
    soft: 10.0
    hard: 50.0
  # ... 其他参数 ...
```

### 3. 配置 LAMMPS 输入脚本

编辑 `inputs/in.monitor.lmp`：

#### 3.1 替换势函数

在 **【第二部分】** 中取消注释并修改适合你体系的势函数。示例：

```lammps
# LJ 流体
pair_style lj/cut 10.0
pair_coeff 1 1 0.1 3.0

# 或金属 EAM
# pair_style eam/alloy
# pair_coeff * * Al.eam.alloy Al

# 或 ReaxFF（反应体系）
# pair_style reax/c NULL
# pair_coeff * * ffield.reax.CH C H O N
```

#### 3.2 替换数据文件

在脚本顶部修改：

```lammps
read_data data/your_system.data  # 改成你的数据文件路径
```

#### 3.3 调整控温器和积分器

根据集合选择（NVE / NVT / NPT），修改：

```lammps
# 示例：NVT（等温）
variable Tstart equal 300.0  # 你的初始温度
variable Tend equal 300.0    # 你的目标温度
fix thermostat all nvt temp ${Tstart} ${Tend} 100.0
```

#### 3.4 调整时间步长和阈值

修改以下变量（与 `config.yaml` 中 `dtreset` 配置保持一致）：

```lammps
variable dtmax equal 0.002   # 最大 dt (ps)
variable dtmin equal 0.0001  # 最小 dt (ps)
variable xmax equal 0.1      # 每步最大位移 (Å)
```

#### 3.5 调整停机阈值

修改 `fix halt` 中的条件（与 `config.yaml` 中 `thresholds` 保持一致）：

```lammps
fix halt_disp all halt 1 v_maxdisp > 5.0 error hard message yes
fix halt_pe all halt 1 v_maxpe > 200.0 error hard message yes
fix halt_ke all halt 1 v_maxke > 50.0 error hard message yes
```

#### 3.6 可选：配位数和键监控

若要监控配位数，取消注释 **【第七部分】**：

```lammps
variable coord_cutoff equal 3.5  # 调整截断距离
compute coord_atom all coord/atom cutoff ${coord_cutoff}
# ... rest of the code ...
```

若体系有显式键且要监控，取消注释 **【第八部分】**：

```lammps
# compute bonds all property/local btype batom1 batom2 blen
# compute max_bond all reduce max c_bonds[4]
# variable v_maxbond equal c_max_bond
```

并在 `fix halt` 中添加：

```lammps
# fix halt_bond all halt 1 v_maxbond > 2.0 error hard message yes
```

### 4. 运行

```bash
cd /home/guozy/workspace/active_allegro/lammps_monitor

# 基本运行
python3 scripts/run_monitor.py

# 或指定配置文件
python3 scripts/run_monitor.py config.yaml

# 干运行（仅验证，不实际运行）
# （修改 config.yaml 中 run.dry_run: true）
```

当运行完成时，会看到类似输出：

```
============================================================
事件元数据摘要
============================================================
运行 ID: 1
时间戳: 2024-04-10T12:34:56
状态: normal
原因: （无）
最后步数: 500000

最后监控指标：
  温度: 300.00 K
  压力: 1.00 atm
  总能量: -1234.5678 kcal/mol
  最大位移: 1.5342 Å
  最大单原子PE: 45.2341 kcal/mol
  最大单原子KE: 8.3456 kcal/mol
  最大配位数: 11.5
============================================================
```

## 输出文件说明

### `log.lammps`

LAMMPS 标准输出日志，包含每步的热力学数据：

```
step        temp        press pe        ke        etotal      vol         dt   v_maxdisp v_maxpe   v_maxke   v_maxcn
1           300.0       1.0   -1234.56  45.67     -1188.89    12345.0     0.002 0.0001   0.001     0.0001    8.0
2           300.1       1.05  -1234.52  45.72     -1188.80    12345.1     0.002 0.0002   0.003     0.0002    8.0
...
```

### `monitor_events.csv`

每次运行的汇总事件日志：

```csv
run_id,timestamp,status,reason,final_step,temp,press,pe,ke,etotal,vol,dt,maxdisp,vmaxpe,vmaxke,vmaxcn
1,2024-04-10T12:34:56,normal,,500000,300.00,1.00,-1234.56,45.67,-1188.89,12345.0,0.002,1.5342,45.2341,8.3456,11.5
```

### `event_metadata.json`

最近一次运行的完整元数据（用于自动处理）：

```json
{
  "run_id": 1,
  "timestamp": "2024-04-10T12:34:56",
  "status": "normal",
  "reason": "",
  "final_step": 500000,
  "final_metrics": {
    "step": 500000,
    "temp": 300.0,
    ...
  }
}
```

### `dump.monitor.atom`

原子轨迹转储，包含每 100 步的原子坐标、速度、力及计算量（`c_pe_atom`, `c_ke_atom`, `c_coord_atom`）

### `last_safe_frame_run*.atom`

在异常停机时自动提取的最后安全帧，可用于：
- 人工检查停机前的结构
- 作为新运行的初始条件（进行恢复运行）
- 进一步分析异常原因

## 决策逻辑

系统根据最后监控指标判断运行状态：

```python
# 定义在 utils.py 中的 make_decision() 函数

if maxdisp > disp_hard or vmaxpe > pe_hard or vmaxke > ke_hard:
    decision = "abort"      # 立即停止
    reason = specific_reason

elif maxdisp > disp_soft or vmaxpe > pe_soft or vmaxke > ke_soft:
    decision = "warning"    # 标记警告但不停止
    reason = specific_reason

else:
    decision = "normal"     # 运行正常
    reason = ""
```

可能的 `reason` 值：

- `""` - 正常运行
- `"displacement_exceeded"` - 最大位移超过硬限
- `"per_atom_pe_exceeded"` - 单原子势能超过硬限
- `"per_atom_ke_exceeded"` - 单原子动能超过硬限
- `"coord_change_abnormal"` - 配位数异常
- `"bondmax_exceeded"` - 最大键长异常
- `"log_parse_failed"` - 日志解析失败
- `"analysis_error: ..."` - 分析异常

## 手动操作

### 提取任意帧

```bash
python3 scripts/extract_last_safe_frame.py outputs/dump.monitor.atom outputs/custom_frame.atom
```

### 解析日志文件

```bash
python3 scripts/parse_log.py outputs/log.lammps
```

输出：

```
============================================================
LAMMPS 日志解析摘要
============================================================
最后一步：500000
温度: 300.00 K
压力: 1.00 atm
总能量: -1188.89 kcal/mol
最大位移: 1.53 Å
最大单原子PE: 45.23 kcal/mol
最大单原子KE: 8.35 kcal/mol
最大配位数: 11.5

【正常完成】
============================================================
```

### 保存自定义事件元数据

```bash
# 创建并保存事件
python3 scripts/save_event_metadata.py create 2 abort "displacement_exceeded" 125000 outputs/custom_event.json

# 加载和展示事件
python3 scripts/save_event_metadata.py load outputs/custom_event.json

# 仅打印事件
python3 scripts/save_event_metadata.py print outputs/event_metadata.json
```

## 高级用法

### 多 Chunk 运行（分段执行）

修改 `config.yaml`：

```yaml
run:
  chunk_size: 100000     # 每 chunk 100000 步
  max_chunks: 10         # 最多 10 个 chunk（总共最多 1M 步）
  restart_mode: false    # 若支持 restart，可设为 true
```

然后循环调用 `run_monitor.py`（可用 shell 脚本包装）。

### 扩展监控量

若需监控额外的物理量（如应力、RDF、二面角等），在 `in.monitor.lmp` 中添加对应的 `compute` 和 `thermo_style` 命令，然后在 `utils.py` 中的 `MonitorMetrics` 和 `parse_final_metrics()` 中更新字段。

### 自动故障恢复

可在 `run_monitor.py` 中添加自动恢复逻辑：

```python
if status == "abort":
    # 自动降低时间步长，读取最后帧，继续模拟
    # 1. 从 last_safe_frame_run*.atom 读取坐标
    # 2. 修改 config.yaml 中的 dtmax，降低 20%
    # 3. 调用 read_data / read_restart，继续 run
    pass
```

## 故障排除

### LAMMPS 命令找不到

**问题**：`[ERROR] LAMMPS 可执行文件 'lmp' 不在 PATH 中`

**解决**：
1. 确保 LAMMPS 已安装在系统上
2. 在 `config.yaml` 中指定绝对路径：`executable: "/usr/local/bin/lmp"`

### 数据文件找不到

**问题**：`read_data: file not found` 错误

**解决**：
1. 在 `in.monitor.lmp` 中使用相对于 LAMMPS 工作目录的路径
2. 或使用绝对路径
3. 确保 `config.yaml` 中的 `data_file` 路径正确

### 日志解析失败

**问题**：`[ERROR] 未找到 thermo 输出表头`

**解决**：
1. 检查 `log.lammps` 是否正确生成
2. 确保 `in.monitor.lmp` 中有 `thermo` 和 `thermo_style` 命令
3. 检查 `log_file` 路径是否正确，权限是否足够

### 进程无法终止

**问题**：`LAMMPS 未响应 SIGTERM，强制杀死`

**解决**：
1. 这通常说明 LAMMPS 在做重计算，很正常
2. 若频繁发生，可增加超时时间（`wait_lammps()` 中的 `timeout`）

## 参考文献

- LAMMPS 文档：https://docs.lammps.org/
- 在线停机命令：https://docs.lammps.org/fix_halt.html
- compute 指令集合：https://docs.lammps.org/Commands_compute.html

## 许可

本项目不限制使用。任意修改、扩展。

## 联系与反馈

如有问题或建议，请检查：
1. 是否正确修改了 `config.yaml` 和 `in.monitor.lmp`
2. 势函数是否与体系匹配
3. 阈值是否合理（建议从宽松阈值开始，逐步调整）
4. 日志输出是否完整

---

**最后更新**：2024-04-10  
**版本**：v1.0

