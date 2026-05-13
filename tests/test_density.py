#!/usr/bin/env python
"""Test information density computation for cold-start trigger."""

from memory_graph_v2 import _compute_information_density

# Test case 1: High-density turn (fire emergency with lots of details)
high_density_text = "Q: 三楼化学品仓库起火，火势很大，有浓烟和爆炸声，可能有人受困\nA: 已启动应急预案，消防队前往灭火，疏散二楼和四楼人员，关闭电梯和排烟系统"
high_density_q = "三楼化学品仓库起火，火势很大，有浓烟和爆炸声，可能有人受困"
high_density_a = "已启动应急预案，消防队前往灭火，疏散二楼和四楼人员，关闭电梯和排烟系统"

score1 = _compute_information_density(high_density_text, high_density_q, high_density_a)
print(f"High-density turn: {score1:.4f} (should be >= 0.65)")

# Test case 2: Medium-density turn (some details)
medium_density_text = "Q: 二楼有烟雾，需要检查\nA: 派人前往查看，关闭相关设备"
medium_density_q = "二楼有烟雾，需要检查"
medium_density_a = "派人前往查看，关闭相关设备"

score2 = _compute_information_density(medium_density_text, medium_density_q, medium_density_a)
print(f"Medium-density turn: {score2:.4f} (should be 0.3-0.6)")

# Test case 3: Low-density turn (minimal information)
low_density_text = "Q: 情况如何\nA: 正常"
low_density_q = "情况如何"
low_density_a = "正常"

score3 = _compute_information_density(low_density_text, low_density_q, low_density_a)
print(f"Low-density turn: {score3:.4f} (should be < 0.3)")

# Test case 4: Empty turn
empty_text = ""
score4 = _compute_information_density(empty_text)
print(f"Empty turn: {score4:.4f} (should be 0.0)")

# Summary
print("\n--- Summary ---")
print(f"High density triggers immediate cold-start: {score1 >= 0.65}")
print(f"Medium density needs 2+ turns: {score2 >= 0.65 * 0.7}")
print(f"Low density needs 6+ turns (fallback): {score3 < 0.65 * 0.7}")
