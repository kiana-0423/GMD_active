#!/usr/bin/env python3
"""
LAMMPS 在线监控系统 - 主控制脚本
==================================================
调度 LAMMPS 运行，实时监控，决策停机，保存日志
"""

import sys
import subprocess
import signal
import os
from pathlib import Path
from datetime import datetime
import yaml
import json
from typing import Optional, Tuple

# 导入本地模块
try:
    from utils import (
        MonitorEvent,
        MonitorMetrics,
        parse_config,
        make_decision,
        write_event_csv,
        write_event_jsonl,
        write_metadata_json,
        alert_user,
        RunConfig,
    )
    from parse_log import LAMMPSLogParser
    from extract_last_safe_frame import extract_frame
    from save_event_metadata import (
        create_event_metadata,
        save_event,
        print_event_summary,
    )
except ImportError as e:
    print(f"[ERROR] 导入本地模块失败: {e}")
    sys.exit(1)


class LAMMPSMonitor:
    """LAMMPS 在线监控器"""
    
    def __init__(self, config_file: Path):
        """初始化监控器"""
        self.config_file = Path(config_file)
        self.config_dict = None
        self.config = None
        self.run_id = 0
        self.process = None
    
    def load_config(self) -> bool:
        """加载 YAML 配置文件"""
        
        if not self.config_file.exists():
            alert_user(f"配置文件不存在: {self.config_file}", level="ERROR")
            return False
        
        try:
            with open(self.config_file, "r") as f:
                self.config_dict = yaml.safe_load(f)
            
            self.config = parse_config(self.config_dict)
            
            alert_user(f"已加载配置文件: {self.config_file}", level="INFO")
            return True
        
        except Exception as e:
            alert_user(f"加载配置文件失败: {e}", level="ERROR")
            return False
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        
        if not self.config:
            alert_user("配置未加载", level="ERROR")
            return False
        
        # 检查 LAMMPS 可执行文件是否存在（若非绝对路径则检查 PATH）
        lmp = self.config.lammps_executable
        if not Path(lmp).is_absolute():
            # 交给 shell 检查
            result = subprocess.run(
                f"which {lmp}",
                shell=True,
                capture_output=True,
            )
            if result.returncode != 0:
                alert_user(f"LAMMPS 可执行文件 '{lmp}' 不在 PATH 中", level="WARNING")
        
        # 检查 LAMMPS 输入文件
        if not self.config.input_file.exists():
            alert_user(f"LAMMPS 输入文件不存在: {self.config.input_file}", level="ERROR")
            return False
        
        # 检查输出目录
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        alert_user("配置验证成功", level="INFO")
        return True
    
    def run_lammps(self) -> bool:
        """
        启动 LAMMPS 子进程
        
        若启用 dry-run，则仅验证不运行
        """
        
        if self.config.dry_run:
            alert_user("干运行模式：跳过实际 LAMMPS 执行", level="INFO")
            return True
        
        cmd = [
            self.config.lammps_executable,
            "-in", str(self.config.input_file),
            "-log", str(self.config.log_file),
        ]
        
        alert_user(f"启动 LAMMPS: {' '.join(cmd)}", level="INFO")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # 行缓模式
            )
            
            return True
        
        except Exception as e:
            alert_user(f"启动 LAMMPS 失败: {e}", level="ERROR")
            return False
    
    def wait_lammps(self, timeout: Optional[float] = None) -> int:
        """
        等待 LAMMPS 进程完成
        
        返回：退出码（0=正常，非0=异常）
        """
        
        if not self.process:
            return 1
        
        try:
            rc = self.process.wait(timeout=timeout)
            return rc
        
        except subprocess.TimeoutExpired:
            alert_user("LAMMPS 运行超时，尝试终止进程", level="ERROR")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            return -1
    
    def terminate_lammps(self) -> bool:
        """优雅地终止 LAMMPS 进程"""
        
        if self.process and self.process.poll() is None:
            alert_user("正在终止 LAMMPS 进程...", level="INFO")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                alert_user("LAMMPS 未响应 SIGTERM，强制杀死", level="WARNING")
                self.process.kill()
                self.process.wait()
            return True
        
        return False
    
    def analyze_results(self) -> Tuple[str, str, MonitorMetrics]:
        """
        分析 LAMMPS 运行结果
        
        返回：(status, reason, metrics)
        - status: "normal", "warning", "abort"
        - reason: 触发原因或空字符串
        - metrics: 最后的监控指标
        """
        
        # 解析日志
        parser = LAMMPSLogParser(self.config.log_file)
        
        try:
            parser.read_log()
            
            if not parser.parse():
                alert_user("日志解析失败", level="ERROR")
                return "abort", "log_parse_failed", None
            
            # 获取最后指标
            metrics = parser.get_final_metrics()
            if not metrics:
                alert_user("无法提取最后监控指标", level="ERROR")
                return "abort", "no_metrics", None
            
            # 判断状态
            status, reason = make_decision(
                maxdisp=metrics.maxdisp,
                vmaxpe=metrics.vmaxpe,
                vmaxke=metrics.vmaxke,
                vmaxcn=metrics.vmaxcn if self.config.enable_coordination else None,
                thresholds=self.config.thresholds,
            )
            
            # 若解析出 abort reason，覆盖判决
            if parser.is_abnormal_exit():
                status = "abort"
                reason = parser.get_abort_reason()
            
            return status, reason, metrics
        
        except Exception as e:
            alert_user(f"结果分析异常: {e}", level="ERROR")
            return "abort", f"analysis_error: {e}", None
    
    def save_results(self, status: str, reason: str, metrics: Optional[MonitorMetrics]):
        """保存运行结果到 CSV / JSONL / JSON"""
        
        self.run_id += 1
        
        event = MonitorEvent(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(timespec='seconds'),
            status=status,
            reason=reason,
            final_step=metrics.step if metrics else 0,
            final_metrics=metrics,
        )
        
        # 保存到 CSV
        events_csv = Path(self.config_dict["output"]["events_csv"])
        events_csv.parent.mkdir(parents=True, exist_ok=True)
        write_event_csv(event, events_csv, append=True)
        alert_user(f"已记录事件到 CSV: {events_csv}", level="INFO")
        
        # 保存到 JSONL（可选）
        events_jsonl = Path(self.config_dict["output"]["events_jsonl"])
        if events_jsonl:
            write_event_jsonl(event, events_jsonl)
            alert_user(f"已记录事件到 JSONL: {events_jsonl}", level="INFO")
        
        # 保存最近事件的完整元数据
        metadata_json = Path(self.config_dict["output"]["metadata_json"])
        write_metadata_json(event, metadata_json)
        alert_user(f"已保存事件元数据: {metadata_json}", level="INFO")
        
        # 若异常，尝试提取最后安全帧
        if status != "normal":
            self._extract_emergency_frames(event)
        
        return event
    
    def _extract_emergency_frames(self, event: MonitorEvent):
        """在异常时提取最后安全帧和上下文"""
        
        dump_file = self.config.dump_file
        if not dump_file.exists():
            alert_user(f"转储文件不存在，跳过帧提取: {dump_file}", level="WARNING")
            return
        
        try:
            safe_frame_file = self.config.output_dir / f"last_safe_frame_run{event.run_id}.atom"
            
            if extract_frame(dump_file, safe_frame_file):
                alert_user(f"已提取最后安全帧: {safe_frame_file}", level="INFO")
            else:
                alert_user("帧提取失败", level="WARNING")
        
        except Exception as e:
            alert_user(f"帧提取异常: {e}", level="WARNING")
    
    def run(self) -> bool:
        """执行完整的监控流程"""
        
        alert_user("=" * 60, level="INFO")
        alert_user("LAMMPS 在线监控系统启动", level="INFO")
        alert_user("=" * 60, level="INFO")
        
        # 第 1 步：加载和验证配置
        if not self.load_config():
            return False
        
        if not self.validate_config():
            return False
        
        # 第 2 步：启动 LAMMPS
        if not self.run_lammps():
            return False
        
        # 第 3 步：等待完成
        alert_user("等待 LAMMPS 运行...", level="INFO")
        rc = self.wait_lammps()
        
        if rc == 0:
            alert_user("LAMMPS 进程正常退出", level="INFO")
        else:
            alert_user(f"LAMMPS 进程异常退出，退出码: {rc}", level="WARNING")
        
        # 第 4 步：分析结果
        alert_user("分析运行结果...", level="INFO")
        status, reason, metrics = self.analyze_results()
        
        # 第 5 步：保存结果
        alert_user("保存运行结果...", level="INFO")
        event = self.save_results(status, reason, metrics)
        
        # 第 6 步：摘要和决策
        alert_user("=" * 60, level="INFO")
        print_event_summary(event)
        alert_user("=" * 60, level="INFO")
        
        # 根据状态决定是否继续迭代运行（可选）
        if status == "abort":
            alert_user("监控系统检测到异常，停止迭代", level="WARNING")
            return False
        
        return True


def main():
    """主函数"""
    
    # 默认配置文件
    config_file = Path("config.yaml")
    
    # 若命令行指定了配置文件
    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
    
    # 创建监控器并运行
    monitor = LAMMPSMonitor(config_file)
    
    try:
        success = monitor.run()
        
        if success:
            alert_user("监控系统正常完成", level="INFO")
            sys.exit(0)
        else:
            alert_user("监控系统异常退出", level="ERROR")
            sys.exit(1)
    
    except KeyboardInterrupt:
        alert_user("收到中断信号，正在清理...", level="WARNING")
        monitor.terminate_lammps()
        sys.exit(1)
    
    except Exception as e:
        alert_user(f"监控系统崩溃: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
