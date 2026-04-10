"""
提取最后安全帧脚本
==================================================
从 dump 文件中提取最后一个有效帧，保存为独立文件
用于人工检查或作为新运行的起点
"""

import sys
import re
from pathlib import Path
from typing import Optional, Tuple


def find_last_frame(dump_file: Path) -> Tuple[Optional[int], int, int]:
    """
    找出 dump 文件中最后一个完整帧的行号范围
    
    返回：(frame_timestep, start_line, end_line) 或 (None, 0, 0)
    
    LAMMPS dump custom 格式：
    ITEM: TIMESTEP
    <timestep>
    ITEM: NUMBER OF ATOMS
    <natoms>
    ITEM: BOX BOUNDS ...
    ...
    ITEM: ATOMS ...
    <atom rows>
    """
    
    with open(dump_file, "r") as f:
        lines = f.readlines()
    
    frame_starts = []  # (timestep, line_idx)
    
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            if i + 1 < len(lines):
                try:
                    timestep = int(lines[i + 1].strip())
                    frame_starts.append((timestep, i))
                except ValueError:
                    pass
            i += 1
        else:
            i += 1
    
    if not frame_starts:
        return None, 0, 0
    
    # 取最后一个 frame 的起始行
    last_timestep, last_start = frame_starts[-1]
    
    # 查找下一个帧的开始（或文件结尾）
    next_frame_start = len(lines)
    if len(frame_starts) > 1:
        next_frame_start = frame_starts[-2][1] if len(frame_starts) > 1 else len(lines)
    
    # 实际上应该寻找最后一个帧之后的下一个帧开始
    # 简化策略：从最后帧开始找下一个 "ITEM: TIMESTEP"
    for j in range(last_start + 1, len(lines)):
        if lines[j].startswith("ITEM: TIMESTEP"):
            next_frame_start = j
            break
    
    return last_timestep, last_start, next_frame_start


def extract_frame(dump_file: Path, output_file: Path):
    """
    提取最后一个帧到输出文件
    """
    
    timestep, start, end = find_last_frame(dump_file)
    
    if timestep is None:
        print(f"[ERROR] 无法从 {dump_file} 中提取有效帧")
        return False
    
    with open(dump_file, "r") as f:
        lines = f.readlines()
    
    # 将最后一帧写出
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        f.writelines(lines[start:end])
    
    print(f"[INFO] 已提取帧 timestep={timestep}")
    print(f"[INFO] 保存到 {output_file}")
    
    return True


def extract_frame_context(
    dump_file: Path,
    output_prefix: Path,
    num_frames: int = 3
):
    """
    提取最后 N 个帧到分别的文件
    
    用于分析停机前的 context（如边界条件改变的过程）
    """
    
    with open(dump_file, "r") as f:
        lines = f.readlines()
    
    frame_starts = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            if i + 1 < len(lines):
                try:
                    timestep = int(lines[i + 1].strip())
                    frame_starts.append((timestep, i))
                except ValueError:
                    pass
            i += 1
        else:
            i += 1
    
    if not frame_starts:
        print(f"[ERROR] 无法找到任何有效帧")
        return False
    
    # 取最后 num_frames 个帧
    frames_to_extract = frame_starts[-num_frames:]
    
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    for idx, (timestep, start) in enumerate(frames_to_extract):
        # 找下一帧或文件结尾
        next_idx = frame_starts.index((timestep, start)) + 1
        end = frame_starts[next_idx][1] if next_idx < len(frame_starts) else len(lines)
        
        output_file = output_prefix.parent / f"{output_prefix.stem}_frame{idx}_{timestep}{output_prefix.suffix}"
        
        with open(output_file, "w") as f:
            f.writelines(lines[start:end])
        
        print(f"[INFO] 已提取帧 {idx}: timestep={timestep}")
        print(f"[INFO] 保存到 {output_file}")
    
    return True


def main():
    """命令行接口"""
    
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <dump_file> [output_file]")
        print(f"示例: {sys.argv[0]} outputs/dump.monitor.atom outputs/last_safe.atom")
        sys.exit(1)
    
    dump_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/last_safe.atom")
    
    if not dump_file.exists():
        print(f"[ERROR] dump 文件不存在: {dump_file}")
        sys.exit(1)
    
    if extract_frame(dump_file, output_file):
        print("[SUCCESS] 帧提取成功")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
