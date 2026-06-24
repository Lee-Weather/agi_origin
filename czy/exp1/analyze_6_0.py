import numpy as np
import pandas as pd
from scipy import signal

df = pd.read_csv(r'e:\X1\real_test\origin_exp1\agibot_x1_train-main\czy\data\isaac_diag_20260623_154917.csv')
print(f"Shape: {df.shape}")
print()

def joint_range(arr):
    return arr.max() - arr.min()

joints = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll']

print("=" * 70)
print("【实验 6.0 诊断】isaac_diag_20260623_154917.csv")
print("问题：机器人完全不迈步前进")
print("=" * 70)

# 1. 速度
print("\n─── 1. 速度跟踪 ───")
cmd = df['cmd_linear_x'].values
blv = df['base_lin_vel_x'].values
print(f"  cmd_linear_x 均值: {np.mean(cmd):.3f}")
print(f"  base_lin_vel_x 均值: {np.mean(blv):.3f}")
print(f"  base_lin_vel_x std: {np.std(blv):.3f}")
print(f"  base_lin_vel_x max: {np.max(blv):.3f}")

# 2. 关节范围
print("\n─── 2. 关节实际范围 ───")
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    print(f"  {j:15s}: 左={joint_range(l):.4f}  右={joint_range(r):.4f}")

# 3. 关节均值
print("\n─── 3. 关节均值（对比默认值）───")
defaults = {
    'hip_pitch': (0.4, -0.4),
    'hip_roll': (0.05, -0.05),
    'hip_yaw': (-0.31, 0.31),
    'knee_pitch': (0.49, 0.49),
    'ankle_pitch': (-0.21, -0.21),
    'ankle_roll': (0.0, 0.0),
}
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    dl, dr = defaults[j]
    print(f"  {j:15s}: 左={np.mean(l):+.4f}(默认{dl:+.2f})  右={np.mean(r):+.4f}(默认{dr:+.2f})")

# 4. action 均值和范围
print("\n─── 4. action 均值和范围 ───")
for j in ['hip_pitch', 'knee_pitch', 'ankle_pitch']:
    la = df[f'action_left_{j}_joint'].values
    ra = df[f'action_right_{j}_joint'].values
    print(f"  {j:15s}: 左均值={np.mean(la):+.4f} 范围={joint_range(la):.4f}  右均值={np.mean(ra):+.4f} 范围={joint_range(ra):.4f}")

# 5. pos_des_raw vs 实际
print("\n─── 5. pos_des_raw 范围（策略输出）vs 实际范围 ───")
for j in ['hip_pitch', 'knee_pitch', 'ankle_pitch']:
    l_act = df[f'pos_left_{j}_joint'].values
    l_des = df[f'pos_des_raw_left_{j}_joint'].values
    r_act = df[f'pos_right_{j}_joint'].values
    r_des = df[f'pos_des_raw_right_{j}_joint'].values
    print(f"  {j:15s}: 左 des范围={joint_range(l_des):.4f} 实际范围={joint_range(l_act):.4f}  右 des范围={joint_range(r_des):.4f} 实际范围={joint_range(r_act):.4f}")

# 6. 跟踪率
print("\n─── 6. 跟踪率 (|actual - des_raw| < 0.1) ───")
for j in ['hip_pitch', 'knee_pitch', 'ankle_pitch']:
    l_act = df[f'pos_left_{j}_joint'].values
    l_des = df[f'pos_des_raw_left_{j}_joint'].values
    r_act = df[f'pos_right_{j}_joint'].values
    r_des = df[f'pos_des_raw_right_{j}_joint'].values
    lt = np.mean(np.abs(l_act - l_des) < 0.1) * 100
    rt = np.mean(np.abs(r_act - r_des) < 0.1) * 100
    print(f"  {j:15s}: 左={lt:.1f}%  右={rt:.1f}%")

# 7. 躯干
print("\n─── 7. 躯干稳定性 ───")
print(f"  base_ang_vel_x std: {np.std(df['base_ang_vel_x'].values):.4f}")
print(f"  base_ang_vel_y std: {np.std(df['base_ang_vel_y'].values):.4f}")
print(f"  base_euler_x std: {np.std(df['base_euler_x'].values):.4f}")
print(f"  base_euler_y std: {np.std(df['base_euler_y'].values):.4f}")

# 8. 接触状态
print("\n─── 8. 接触状态 ───")
lc = df['left_contact'].values
rc = df['right_contact'].values
print(f"  左脚接触率: {np.mean(lc)*100:.1f}%")
print(f"  右脚接触率: {np.mean(rc)*100:.1f}%")
print(f"  双脚同时接触率: {np.mean((lc > 0) & (rc > 0))*100:.1f}%")

# 9. 与 5.8 对比
print("\n─── 9. 与 5.8 对比 ───")
print(f"{'指标':<25s} {'5.8':>10s} {'6.0':>10s}")
print("-" * 45)
print(f"{'base_lin_vel_x 均值':<25s} {'0.32':>10s} {np.mean(blv):>10.3f}")
print(f"{'左膝范围':<25s} {'0.604':>10s} {joint_range(df['pos_left_knee_pitch_joint'].values):>10.3f}")
print(f"{'右膝范围':<25s} {'0.311':>10s} {joint_range(df['pos_right_knee_pitch_joint'].values):>10.3f}")
print(f"{'左踝pitch范围':<25s} {'—':>10s} {joint_range(df['pos_left_ankle_pitch_joint'].values):>10.3f}")
print(f"{'右踝pitch范围':<25s} {'—':>10s} {joint_range(df['pos_right_ankle_pitch_joint'].values):>10.3f}")
