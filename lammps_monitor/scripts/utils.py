"""
LAMMPS 在线监控系统 - 工具函数模块
==================================================
包含日志解析、配置管理、决策逻辑等辅助函数
"""

import re
import json
import csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict


@dataclass
class MonitorMetrics:
    """单次监控指标"""
    step: int
    temp: float
    press: float
    pe: float
    ke: float
    etotal: float
    vol: float
    dt: float
    maxdisp: float  # 最大位移
    vmaxpe: float  # 最大单原子势能
    vmaxke: float  # 最大单原子动能
    vmaxcn: float  # 最大配位数


@dataclass
class MonitorEvent:
    """单次运行事件信息"""
    run_id: int  # 运行 ID
    timestamp: str  # ISO 格式时间戳
    status: str  # "normal", "warning", "abort"
    reason: str  # 触发原因
    final_step: int  # 最终步数
    final_metrics: Optional[MonitorMetrics] = None


@dataclass
class RunConfig:
    """运行配置"""
    lammps_executable: str
    input_file: Path
    log_file: Path
    dump_file: Path
    data_file: Optional[Path]
    output_dir: Path
    dtreset: Dict[str, float]
    thresholds: Dict[str, Dict[str, float]]
    enable_coordination: bool
    enable_bondmax: bool
    thermo_freq: int
    dump_freq: int
    verbose: bool
    dry_run: bool


def parse_config(config_dict: Dict[str, Any]) -> RunConfig:
    """将 YAML 字典转换为 RunConfig 对象"""
    return RunConfig(
        lammps_executable=config_dict["lammps"]["executable"],
        input_file=Path(config_dict["lammps"]["input_file"]),
        log_file=Path(config_dict["lammps"]["log_file"]),
        dump_file=Path(config_dict["lammps"]["dump_file"]),
        data_file=Path(config_dict["lammps"]["data_file"]) 
                   if config_dict["lammps"].get("data_file") else None,
        output_dir=Path(config_dict["output"]["output_dir"]),
        dtreset=config_dict["dtreset"],
        thresholds=config_dict["thresholds"],
        enable_coordination=config_dict["monitor"]["enable_coordination_monitoring"],
        enable_bondmax=config_dict["monitor"]["enable_bondmax_monitoring"],
        thermo_freq=config_dict["monitor"]["thermo_frequency"],
        dump_freq=config_dict["monitor"]["dump_frequency"],
        verbose=config_dict["monitor"]["verbose"],
        dry_run=config_dict["run"]["dry_run"],
    )


def make_decision(
    maxdisp: float,
    vmaxpe: float,
    vmaxke: float,
    vmaxcn: Optional[float],
    thresholds: Dict[str, Dict[str, float]]
) -> Tuple[str, str]:
    """
    根据监控指标和阈值判断运行状态
    
    返回：(decision, reason)
    - decision: "normal", "warning", "abort"
    - reason: 触发原因，e.g. "displacement_exceeded", "" (正常无异常)
    """
    
    disp_soft = thresholds["displacement"]["soft"]
    disp_hard = thresholds["displacement"]["hard"]
    pe_soft = thresholds["per_atom_pe"]["soft"]
    pe_hard = thresholds["per_atom_pe"]["hard"]
    ke_soft = thresholds["per_atom_ke"]["soft"]
    ke_hard = thresholds["per_atom_ke"]["hard"]
    
    # 检查硬停止条件
    if maxdisp > disp_hard:
        return "abort", "displacement_exceeded"
    
    if vmaxpe > pe_hard:
        return "abort", "per_atom_pe_exceeded"
    
    if vmaxke > ke_hard:
        return "abort", "per_atom_ke_exceeded"
    
    # 检查配位数（若启用且有数据）
    if vmaxcn is not None:
        coord_hard = thresholds["coordination"].get("hard")
        if coord_hard is not None and vmaxcn > coord_hard:
            return "abort", "coord_change_abnormal"
    
    # 检查键长（若启用且有数据）
    bondmax = thresholds["bondmax"].get("hard")
    if bondmax is not None and vmaxcn is not None:  # 这里用 vmaxcn 作占位符，实际应该有 vmaxbond 变量
        # 注：下面以 vmaxcn 作例子，实际应改为对应的 bondmax 变量
        # if vmaxbond > bondmax:
        #     return "abort", "bondmax_exceeded"
        pass
    
    # 检查软警告条件
    if maxdisp > disp_soft or vmaxpe > pe_soft or vmaxke > ke_soft:
        reason_list = []
        if maxdisp > disp_soft:
            reason_list.append("displacement_warning")
        if vmaxpe > pe_soft:
            reason_list.append("per_atom_pe_warning")
        if vmaxke > ke_soft:
            reason_list.append("per_atom_ke_warning")
        return "warning", "; ".join(reason_list)
    
    # 正常
    return "normal", ""


def parse_thermo_header_line(line: str) -> Optional[List[str]]:
    """解析 thermo 表头行，若不是表头则返回 None"""

    if re.match(r"^\s*step\s+", line, flags=re.IGNORECASE):
        return line.split()

    return None


def parse_thermo_data_line(
    line: str,
    columns: Optional[List[str]]
) -> Optional[Dict[str, str]]:
    """根据已有列名解析单行 thermo 数据"""

    if not columns:
        return None

    stripped = line.strip()
    if not stripped or not re.match(r"^\d+\s+", stripped):
        return None

    values = stripped.split()
    if len(values) != len(columns):
        return None

    return dict(zip(columns, values))


def parse_thermo_metrics_line(
    line: str,
    columns: Optional[List[str]]
) -> Optional[MonitorMetrics]:
    """将单行 thermo 数据解析为结构化指标"""

    row = parse_thermo_data_line(line, columns)
    if row is None:
        return None

    return parse_final_metrics([row], columns)


def extract_thermo_data(log_content: str) -> List[Dict[str, str]]:
    """
    从 LAMMPS 日志中提取 thermo 数据行
    
    处理多段 run 导致的重复表头，只返回最后一个有效段的数据
    
    返回：[(step, temp, press, ...), ...]
    """
    
    lines = log_content.split("\n")
    
    # 查找所有 thermo 表头位置
    header_indices = []
    for i, line in enumerate(lines):
        # thermo 表头通常以 "step" 开头
        if re.match(r"^\s*step\s+", line):
            header_indices.append(i)
    
    if not header_indices:
        raise ValueError("未找到 thermo 输出表头，请检查日志文件或 LAMMPS 配置")
    
    # 取最后一个表头之后的所有数据行
    last_header_idx = header_indices[-1]
    
    # 提取表头
    header_line = lines[last_header_idx]
    columns = header_line.split()
    
    # 提取数据行（跳过分割线）
    data_lines = []
    for i in range(last_header_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line or re.match(r"^\s*[-=]+", line):
            continue
        if re.match(r"^\d+\s+", line):  # 以数字开头的行
            data_lines.append(line)
    
    # 解析为字典列表
    result = []
    for data_line in data_lines:
        values = data_line.split()
        if len(values) == len(columns):
            result.append(dict(zip(columns, values)))
    
    return result, columns


def parse_final_metrics(
    data_rows: List[Dict[str, str]],
    columns: List[str]
) -> Optional[MonitorMetrics]:
    """
    抽取最后一行数据的监控指标
    
    返回：MonitorMetrics 或 None
    """
    
    if not data_rows:
        return None
    
    last_row = data_rows[-1]
    
    try:
        metrics = MonitorMetrics(
            step=int(last_row.get("step", 0)),
            temp=float(last_row.get("temp", 0)),
            press=float(last_row.get("press", 0)),
            pe=float(last_row.get("pe", 0)),
            ke=float(last_row.get("ke", 0)),
            etotal=float(last_row.get("etotal", 0)),
            vol=float(last_row.get("vol", 0)),
            dt=float(last_row.get("dt", 0)),
            maxdisp=float(last_row.get("v_maxdisp", 0)),
            vmaxpe=float(last_row.get("v_maxpe", 0)),
            vmaxke=float(last_row.get("v_maxke", 0)),
            vmaxcn=float(last_row.get("v_maxcn", 0)),
        )
        return metrics
    except (ValueError, KeyError) as e:
        raise ValueError(f"解析监控指标失败: {e}")


def detect_abort_reason(log_content: str) -> Optional[str]:
    """
    从日志中识别 LAMMPS 的停止原因
    
    返回：停止原因字符串，或 None 表示正常完成
    """
    
    for line in log_content.split("\n"):
        reason = detect_abort_reason_in_line(line)
        if reason:
            return reason

    return None


def detect_abort_reason_in_line(line: str) -> Optional[str]:
    """从单行输出中识别是否出现致命错误"""

    stripped = line.strip()
    if not stripped:
        return None

    lower_line = stripped.lower()

    if "error:" in lower_line:
        return stripped

    fatal_keywords = [
        "segmentation fault",
        "nan",
        "inf",
        "diverged",
        "lost atoms",
        "out of bounds",
        "halting",
    ]

    for keyword in fatal_keywords:
        if keyword in lower_line:
            return stripped

    return None


def write_event_csv(
    event: MonitorEvent,
    csv_file: Path,
    append: bool = True
):
    """
    将单个事件写入 CSV 文件
    
    若文件不存在或 append=False，则创建新文件
    """
    
    fieldnames = ["run_id", "timestamp", "status", "reason", "final_step",
                  "temp", "press", "pe", "ke", "etotal", "vol", "dt",
                  "maxdisp", "vmaxpe", "vmaxke", "vmaxcn"]
    
    file_exists = csv_file.exists()
    
    with open(csv_file, "a" if (append and file_exists) else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists or not append:
            writer.writeheader()
        
        row = {
            "run_id": event.run_id,
            "timestamp": event.timestamp,
            "status": event.status,
            "reason": event.reason,
            "final_step": event.final_step,
        }
        
        if event.final_metrics:
            m = event.final_metrics
            row.update({
                "temp": m.temp,
                "press": m.press,
                "pe": m.pe,
                "ke": m.ke,
                "etotal": m.etotal,
                "vol": m.vol,
                "dt": m.dt,
                "maxdisp": m.maxdisp,
                "vmaxpe": m.vmaxpe,
                "vmaxke": m.vmaxke,
                "vmaxcn": m.vmaxcn,
            })
        
        writer.writerow(row)


def write_event_jsonl(
    event: MonitorEvent,
    jsonl_file: Path
):
    """
    将单个事件追加到 JSONL 文件（一行一个 JSON 对象）
    """
    
    event_dict = {
        "run_id": event.run_id,
        "timestamp": event.timestamp,
        "status": event.status,
        "reason": event.reason,
        "final_step": event.final_step,
    }
    
    if event.final_metrics:
        event_dict["final_metrics"] = asdict(event.final_metrics)
    
    with open(jsonl_file, "a") as f:
        f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")


def write_metadata_json(
    event: MonitorEvent,
    metadata_file: Path
):
    """
    保存最近一次停机事件的完整元数据（JSON 格式）
    """
    
    metadata = {
        "run_id": event.run_id,
        "timestamp": event.timestamp,
        "status": event.status,
        "reason": event.reason,
        "final_step": event.final_step,
    }
    
    if event.final_metrics:
        metadata["final_metrics"] = asdict(event.final_metrics)
    
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def write_trend_csv(
    metrics: MonitorMetrics,
    csv_file: Path,
    append: bool = True,
):
    """追加写入单条实时监控指标"""

    fieldnames = [
        "step", "temp", "press", "pe", "ke", "etotal", "vol", "dt",
        "maxdisp", "vmaxpe", "vmaxke", "vmaxcn",
    ]

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_file.exists()

    with open(csv_file, "a" if (append and file_exists) else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists or not append:
            writer.writeheader()

        writer.writerow({
            "step": metrics.step,
            "temp": metrics.temp,
            "press": metrics.press,
            "pe": metrics.pe,
            "ke": metrics.ke,
            "etotal": metrics.etotal,
            "vol": metrics.vol,
            "dt": metrics.dt,
            "maxdisp": metrics.maxdisp,
            "vmaxpe": metrics.vmaxpe,
            "vmaxke": metrics.vmaxke,
            "vmaxcn": metrics.vmaxcn,
        })


def get_last_safe_timestep(
    data_rows: List[Dict[str, str]]
) -> int:
    """
    获取最后一个步数（作为"最后安全帧"的参考）
    """
    
    if not data_rows:
        return 0
    
    try:
        return int(data_rows[-1].get("step", 0))
    except ValueError:
        return 0


def alert_user(message: str, level: str = "INFO"):
    """
    简单的摘要输出函数，可扩展为接\网络报警
    
    level: "INFO", "WARNING", "ERROR"
    """
    
    prefix_map = {
        "INFO": "[INFO]",
        "WARNING": "[WARN]",
        "ERROR": "[ERROR]",
    }
    prefix = prefix_map.get(level, "[LOG]")
    print(f"{prefix} {message}")
