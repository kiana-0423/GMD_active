"""
保存事件元数据脚本
==================================================
将停机事件信息保存为 JSON，用于后续分析和追踪
"""

import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    from utils import MonitorEvent, write_metadata_json, alert_user
except ImportError:
    print("错误：无法导入 utils.py")
    sys.exit(1)


def create_event_metadata(
    run_id: int,
    status: str,
    reason: str,
    final_step: int,
    final_metrics_dict: Optional[dict] = None,
) -> MonitorEvent:
    """
    创建事件元数据对象
    
    参数：
    - run_id: 运行 ID
    - status: "normal", "warning", "abort"
    - reason: 触发原因或空字符串
    - final_step: 最后的步数
    - final_metrics_dict: 最后的监控指标（字典格式）
    """
    
    timestamp = datetime.now().isoformat(timespec='seconds')
    
    # 若提供了指标字典，需要重新构建 MonitorMetrics
    from utils import MonitorMetrics
    
    final_metrics = None
    if final_metrics_dict:
        try:
            final_metrics = MonitorMetrics(**final_metrics_dict)
        except TypeError:
            alert_user("指标字典格式不正确", level="WARNING")
    
    event = MonitorEvent(
        run_id=run_id,
        timestamp=timestamp,
        status=status,
        reason=reason,
        final_step=final_step,
        final_metrics=final_metrics,
    )
    
    return event


def save_event(
    event: MonitorEvent,
    metadata_file: Path,
):
    """保存事件到 JSON 文件"""
    
    write_metadata_json(event, metadata_file)
    alert_user(f"已保存事件元数据到 {metadata_file}", level="INFO")


def load_event(metadata_file: Path) -> Optional[MonitorEvent]:
    """从 JSON 文件加载事件"""
    
    if not metadata_file.exists():
        return None
    
    with open(metadata_file, "r") as f:
        data = json.load(f)
    
    # 重建 MonitorEvent
    from utils import MonitorMetrics
    
    final_metrics = None
    if "final_metrics" in data:
        try:
            final_metrics = MonitorMetrics(**data["final_metrics"])
        except TypeError:
            pass
    
    event = MonitorEvent(
        run_id=data.get("run_id", 0),
        timestamp=data.get("timestamp", ""),
        status=data.get("status", ""),
        reason=data.get("reason", ""),
        final_step=data.get("final_step", 0),
        final_metrics=final_metrics,
    )
    
    return event


def print_event_summary(event: MonitorEvent):
    """打印事件摘要"""
    
    print("\n" + "=" * 60)
    print("事件元数据摘要")
    print("=" * 60)
    print(f"运行 ID: {event.run_id}")
    print(f"时间戳: {event.timestamp}")
    print(f"状态: {event.status}")
    print(f"原因: {event.reason or '（无）'}")
    print(f"最后步数: {event.final_step}")
    
    if event.final_metrics:
        m = event.final_metrics
        print(f"\n最后监控指标：")
        print(f"  温度: {m.temp:.2f} K")
        print(f"  压力: {m.press:.2f} atm")
        print(f"  总能量: {m.etotal:.4f} kcal/mol")
        print(f"  最大位移: {m.maxdisp:.4f} Å")
        print(f"  最大单原子PE: {m.vmaxpe:.4f} kcal/mol")
        print(f"  最大单原子KE: {m.vmaxke:.4f} kcal/mol")
        print(f"  最大配位数: {m.vmaxcn:.1f}")
    
    print("=" * 60)


def main():
    """命令行接口"""
    
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <action> [args...]")
        print(f"动作：")
        print(f"  create <run_id> <status> <reason> <final_step> [output_file]")
        print(f"  load <metadata_file>")
        print(f"  print <metadata_file>")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "create":
        if len(sys.argv) < 5:
            print("用法: create <run_id> <status> <reason> <final_step> [output_file]")
            sys.exit(1)
        
        run_id = int(sys.argv[2])
        status = sys.argv[3]
        reason = sys.argv[4]
        final_step = int(sys.argv[5])
        output_file = Path(sys.argv[6]) if len(sys.argv) > 6 else Path("outputs/event_metadata.json")
        
        event = create_event_metadata(run_id, status, reason, final_step)
        save_event(event, output_file)
        print_event_summary(event)
    
    elif action == "load":
        if len(sys.argv) < 3:
            print("用法: load <metadata_file>")
            sys.exit(1)
        
        metadata_file = Path(sys.argv[2])
        event = load_event(metadata_file)
        
        if event:
            print_event_summary(event)
        else:
            print(f"[ERROR] 无法加载元数据文件: {metadata_file}")
            sys.exit(1)
    
    elif action == "print":
        if len(sys.argv) < 3:
            print("用法: print <metadata_file>")
            sys.exit(1)
        
        metadata_file = Path(sys.argv[2])
        event = load_event(metadata_file)
        
        if event:
            print_event_summary(event)
        else:
            print(f"[ERROR] 无法加载元数据文件: {metadata_file}")
            sys.exit(1)
    
    else:
        print(f"[ERROR] 未知动作: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
