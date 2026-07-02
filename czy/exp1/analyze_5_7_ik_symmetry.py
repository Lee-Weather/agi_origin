import numpy as np
import pandas as pd

df = pd.read_csv(r'e:\X1\real_test\origin_exp1\agibot_x1_train-main\czy\data\isaac_diag_20260618_152415.csv')

# 关键问题：pos_des_raw 的 hip_yaw 和 hip_roll 不对称远超默认值差异
# 需要验证 IK 函数是否正确输出了对称的参考轨迹

print("=" * 70)
print("【IK 参考轨迹对称性验证】")
print("=" * 70)

# 1. 检查 IK 输出（pos_des_raw）是否应该完全对称
# 根据 IK 代码，hip_roll 和 hip_yaw 应该保持默认值，不做调整
# 左腿 hip_roll = default[1], hip_yaw = default[2]
# 右腿 hip_roll = default[7], hip_yaw = default[8]

print("\n─── 配置的默认角度 ───")
print("  left_hip_pitch  =  0.4")
print("  left_hip_roll   =  0.05")
print("  left_hip_yaw    = -0.31")
print("  right_hip_pitch = -0.4")
print("  right_hip_roll  = -0.05")
print("  right_hip_yaw   =  0.31")

print("\n─── pos_des_raw hip_roll/yaw 分析（应该 = 默认值）───")
lh_roll = df['pos_des_raw_left_hip_roll_joint'].values
lh_yaw = df['pos_des_raw_left_hip_yaw_joint'].values
rh_roll = df['pos_des_raw_right_hip_roll_joint'].values
rh_yaw = df['pos_des_raw_right_hip_yaw_joint'].values

print(f"左髋 roll : 均值={np.mean(lh_roll):.4f}  std={np.std(lh_roll):.4f}  期望≈0.05")
print(f"左髋 yaw  : 均值={np.mean(lh_yaw):.4f}  std={np.std(lh_yaw):.4f}  期望≈-0.31")
print(f"右髋 roll : 均值={np.mean(rh_roll):.4f}  std={np.std(rh_roll):.4f}  期望≈-0.05")
print(f"右髋 yaw  : 均值={np.mean(rh_yaw):.4f}  std={np.std(rh_yaw):.4f}  期望≈0.31")

print("\n─── 问题：IK 输出的 hip_roll/yaw 是否被策略修改？───")
# pos_des_raw = action * 0.5 + default
# 如果 IK 正确输出 default，那么 action 应该 ≈ 0
lh_roll_action = df['action_left_hip_roll_joint'].values
lh_yaw_action = df['action_left_hip_yaw_joint'].values
rh_roll_action = df['action_right_hip_roll_joint'].values
rh_yaw_action = df['action_right_hip_yaw_joint'].values

print(f"左髋 roll action : 均值={np.mean(lh_roll_action):.4f}  std={np.std(lh_roll_action):.4f}")
print(f"左髋 yaw action  : 均值={np.mean(lh_yaw_action):.4f}  std={np.std(lh_yaw_action):.4f}")
print(f"右髋 roll action : 均值={np.mean(rh_roll_action):.4f}  std={np.std(rh_roll_action):.4f}")
print(f"右髋 yaw action  : 均值={np.mean(rh_yaw_action):.4f}  std={np.std(rh_yaw_action):.4f}")

print("\n─── 推论 ───")
print("如果 action ≠ 0，说明策略在主动修改 hip_roll/yaw")
print("如果 pos_des_raw ≠ default，可能原因：")
print("  1. IK 代码没有正确固定 hip_roll/yaw 到默认值")
print("  2. ref_action 计算错误，导致策略学到了不对称行为")

# 2. 检查 ref_action（IK 输出 - default）是否对称
print("\n─── ref_action 推算（IK输出 - default）───")
# ref_action = (ref_dof_pos - default_dof_pos) / action_scale
# 如果 IK 正确，ref_action 对 hip_roll/yaw 应该 ≈ 0

# 从数据推算 ref_dof_pos = pos_des_raw（如果策略完全跟踪参考）
# 但训练中策略会偏离参考，所以这个推算不可靠
# 需要直接检查代码

print("\n─── 关键诊断 ───")
print("数据显示：")
print(f"  pos_des_raw 左髋yaw 均值 = {np.mean(lh_yaw):.4f}（期望 -0.31）")
print(f"  pos_des_raw 右髋yaw 均值 = {np.mean(rh_yaw):.4f}（期望  0.31）")
print(f"  差值 = {np.mean(lh_yaw) - np.mean(rh_yaw):.4f}（期望 -0.62）")
print()
print("如果 IK 正确固定 hip_yaw 到默认值，pos_des_raw 应该 ≈ default")
print("但数据显示均值差达到 -0.74，远超配置的 -0.62")
print()
print("可能原因：")
print("  1. IK 函数正确，但策略学习到了额外的不对称补偿")
print("  2. ref_action 计算有 Bug，导致参考轨迹本身不对称")
print("  3. hip_yaw_roll_default 奖励权重（0.5）不足以约束策略")

# 3. 检查 hip_yaw_roll_default 奖励是否生效
print("\n─── hip_yaw_roll_default 奖励验证 ───")
print("该奖励应该约束 hip_yaw/roll 接近默认值")
print("如果奖励生效，actual hip_yaw/roll 应该接近默认值")
lh_yaw_actual = df['pos_left_hip_yaw_joint'].values
lh_roll_actual = df['pos_left_hip_roll_joint'].values
rh_yaw_actual = df['pos_right_hip_yaw_joint'].values
rh_roll_actual = df['pos_right_hip_roll_joint'].values

print(f"实际左髋 yaw  : 均值={np.mean(lh_yaw_actual):.4f}  默认=-0.31")
print(f"实际右髋 yaw  : 均值={np.mean(rh_yaw_actual):.4f}  默认=0.31")
print(f"实际左髋 roll : 均值={np.mean(lh_roll_actual):.4f}  默认=0.05")
print(f"实际右髋 roll : 均值={np.mean(rh_roll_actual):.4f}  默认=-0.05")