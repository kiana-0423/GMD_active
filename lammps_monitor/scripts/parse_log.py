"""
LAMMPS 日志解析器
==================================================
解析 LAMMPS output log.lammps 并提取监控指标
支持多段 run 和不规则格式
"""

import sys
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# 假设 utils 在同一目录
try:
    from utils import (
        MonitorMetrics,
        extract_thermo_data,
        parse_final_metrics,
        detect_abort_reason,
        alert_user,
    )
except ImportError:
    print("错误：无法导入 utils.py，请确保 utils.py 在同一目录")
    sys.exit(1)


class LAMMPSLogParser:
    """LAMMPS 日志解析器"""
    
    def __init__(self, log_file: Path):
        """初始化解析器"""
        self.log_file = Path(log_file)
        self.content = None
        self.thermo_data = None
        self.thermo_columns = None
        self.final_metrics = None
        self.abort_reason = None
    
    def read_log(self) -> str:
        """读取日志文件"""
        if not self.log_file.exists():
            raise FileNotFoundError(f"日志文件不存在: {self.log_file}")
        
        with open(self.log_file, "r") as f:
            self.content = f.read()
        
        return self.content
    
    def parse(self) -> bool:
        """
        解析日志
        
        返回：True 表示解析成功，False 表示解析失败或日志为空
        """
        
        if self.content is None:
            self.read_log()
        
        if not self.content.strip():
            alert_user("日志文件为空", level="ERROR")
            return False
        
        try:
            # 提取 thermo 数据
            self.thermo_data, self.thermo_columns = extract_thermo_data(self.content)
            
            # 检查是否有数据
            if not self.thermo_data:
                alert_user("未找到有效的 thermo 数据行", level="WARNING")
                return False
            
            # 解析最后一行的监控指标
            self.final_metrics = parse_final_metrics(self.thermo_data, self.thermo_columns)
            
            # 检测终止原因
            self.abort_reason = detect_abort_reason(self.content)
            
            return True
        
        except Exception as e:
            alert_user(f"日志解析异常: {e}", level="ERROR")
            return False
    
    def get_final_metrics(self) -> Optional[MonitorMetrics]:
        """获取最后一行的监控指标"""
        return self.final_metrics
    
    def get_all_data(self) -> List[Dict[str, str]]:
        """获取所有 thermo 数据行"""
        return self.thermo_data or []
    
    def get_columns(self) -> List[str]:
        """获取列名"""
        return self.thermo_columns or []
    
    def get_abort_reason(self) -> Optional[str]:
        """获取停止原因（若有）"""
        return self.abort_reason
    
    def is_abnormal_exit(self) -> bool:
        """判断是否异常退出（有 abort reason）"""
        return self.abort_reason is not None
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("LAMMPS 日志解析摘要")
        print("=" * 60)
        
        if self.final_metrics:
            m = self.final_metrics
            print(f"最后一步：{m.step}")
            print(f"温度: {m.temp:.2f} K")
            print(f"压力: {m.press:.2f} atm")
            print(f"总能量: {m.etotal:.4f} kcal/mol")
            print(f"最大位移: {m.maxdisp:.4f} Å")
            print(f"最大单原子PE: {m.vmaxpe:.4f} kcal/mol")
            print(f"最大单原子KE: {m.vmaxke:.4f} kcal/mol")
            print(f"最大配位数: {m.vmaxcn:.1f}")
        
        if self.abort_reason:
            print(f"\n【异常退出】")
            print(f"原因: {self.abort_reason}")
        else:
            print(f"\n【正常完成】")
        
        print("=" * 60 + "\n")


def main():
    """命令行接口"""
    
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <log_file>")
        print(f"示例: {sys.argv[0]} outputs/log.lammps")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    parser = LAMMPSLogParser(log_file)
    
    try:
        parser.read_log()
        if parser.parse():
            parser.print_summary()
            
            # 若需要 JSON 输出，可添加以下代码
            # metrics = parser.get_final_metrics()
            # if metrics:
            #     print(json.dumps(asdict(metrics), indent=2))
        else:
            alert_user("日志解析失败", level="ERROR")
            sys.exit(1)
    
    except Exception as e:
        alert_user(f"解析过程异常: {e}", level="ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
