import numpy as np
import pandas as pd

df = pd.read_csv(r'e:\X1\real_test\origin_exp1\agibot_x1_train-main\czy\data\isaac_diag_20260623_191213.csv')
print(f"Shape: {df.shape}")

def joint_range(arr):
    return arr.max() - arr.min()

joints = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll']

print("=" * 70)
print("【实验 6.2 诊断】isaac_diag_20260623_191213.csv")
print("=" * 70)

# 1. 速度
print("\n─── 1. 速度跟踪 ───")
cmd = df['cmd_linear_x'].values
blv = df['base_lin_vel_x'].values
print(f"  cmd_linear_x 均值: {np.mean(cmd):.3f}")
print(f"  base_lin_vel_x 均值: {np.mean(blv):.3f}")
print(f"  base_lin_vel_x std: {np.std(blv):.3f}")

# 2. 关节范围
print("\n─── 2. 关节实际范围 ───")
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    print(f"  {j:15s}: 左={joint_range(l):.4f}  右={joint_range(r):.4f}")

# 3. 接触状态
print("\n─── 3. 接触状态 ───")
lc = df['left_contact'].values
rc = df['right_contact'].values
print(f"  左脚接触率: {np.mean(lc>0)*100:.1f}%")
print(f"  右脚接触率: {np.mean(rc>0)*100:.1f}%")
print(f"  双脚同时接触率: {np.mean((lc>0)&(rc>0))*100:.1f}%")

# 4. action 输出
print("\n─── 4. Action 输出范围 ───")
for j in joints:
    l = df[f'action_left_{j}_joint'].values
    r = df[f'action_right_{j}_joint'].values
    print(f"  {j:15s}: 左={joint_range(l):.4f}  右={joint_range(r):.4f}")

# 5. 躯干稳定性
print("\n─── 5. 躯干稳定性 ───")
print(f"  base_euler_x 范围: {joint_range(df['base_euler_x'].values):.4f}")
print(f"  base_euler_y 范围: {joint_range(df['base_euler_y'].values):.4f}")
print(f"  base_ang_vel_x std: {np.std(df['base_ang_vel_x'].values):.4f}")

# 6. 对比 6.1
print("\n─── 6. 与 6.1 对比 ───")
print(f"{'指标':<25s} {'6.1':>10s} {'6.2':>10s}")
print("-" * 50)
print(f"{'速度vx均值':<25s} {'0.258':>10s} {np.mean(blv):>10.3f}")
print(f"{'左膝范围':<25s} {'0.460':>10s} {joint_range(df['pos_left_knee_pitch_joint'].values):>10.3f}")
print(f"{'右膝范围':<25s} {'0.282':>10s} {joint_range(df['pos_right_knee_pitch_joint'].values):>10.3f}")
print(f"{'双脚接触率':<25s} {'22%':>10s} {np.mean((lc>0)&(rc>0))*100:>10.1f}%")

# 7. 检查策略是否输出
print("\n─── 7. 策略输出检查 ───")
l_hip_act = df['action_left_hip_pitch_joint'].values
r_hip_act = df['action_right_hip_pitch_joint'].values
print(f"  left_hip_pitch action: 均值={np.mean(l_hip_act):.4f} 范围={joint_range(l_hip_act):.4f}")
print(f"  right_hip_pitch action: 均值={np.mean(r_hip_act):.4f} 范围={joint_range(r_hip_act):.4f}")

# 8. 检查参考轨迹
print("\n─── 8. 参考轨迹检查 ───")
try:
    l_ref = df['ref_dof_pos_left_hip_pitch_joint'].values
    r_ref = df['ref_dof_pos_right_hip_pitch_joint'].values
    print(f"  left_hip_pitch ref: 均值={np.mean(l_ref):.4f} 范围={joint_range(l_ref):.4f}")
    print(f"  right_hip_pitch ref: 均值={np.mean(r_ref):.4f} 范围={joint_range(r_ref):.4f}")
except:
    print("  ref_dof_pos 列不存在")
