import numpy as np
import pandas as pd
from scipy import signal

df = pd.read_csv(r'e:\X1\real_test\origin_exp1\agibot_x1_train-main\czy\data\isaac_diag_20260623_173512.csv')
print(f"Shape: {df.shape}")

def joint_range(arr):
    return arr.max() - arr.min()

joints = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll']

print("=" * 70)
print("【实验 6.1 诊断】isaac_diag_20260623_173512.csv")
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

# 4. 左右对称性
print("\n─── 4. 左右腿对称性（运动幅度）───")
for j in ['hip_pitch', 'knee_pitch', 'ankle_pitch']:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    dl, dr = defaults[j]
    l_off = np.abs(l - dl)
    r_off = np.abs(r - dr)
    ratio = np.mean(l_off) / (np.mean(r_off) + 1e-6)
    print(f"  {j:15s}: 左偏移均值={np.mean(l_off):.4f}  右偏移均值={np.mean(r_off):.4f}  比值={ratio:.2f}")

# 5. hip_roll/yaw 偏移
print("\n─── 5. hip_roll/yaw 偏移 ───")
for j in ['hip_roll', 'hip_yaw']:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    dl, dr = defaults[j]
    print(f"  {j}: 左偏移={np.abs(np.mean(l)-dl):.4f}  右偏移={np.abs(np.mean(r)-dr):.4f}")

# 6. 踝关节滚动检查
print("\n─── 6. 踝关节主动滚动（ankle_pitch）───")
l_ap = df['pos_left_ankle_pitch_joint'].values
r_ap = df['pos_right_ankle_pitch_joint'].values
print(f"  左踝pitch: 均值={np.mean(l_ap):+.4f}  范围={joint_range(l_ap):.4f}  默认={defaults['ankle_pitch'][0]:+.2f}")
print(f"  右踝pitch: 均值={np.mean(r_ap):+.4f}  范围={joint_range(r_ap):.4f}  默认={defaults['ankle_pitch'][1]:+.2f}")
print(f"  左踝偏移: {np.mean(l_ap) - defaults['ankle_pitch'][0]:+.4f}")
print(f"  右踝偏移: {np.mean(r_ap) - defaults['ankle_pitch'][1]:+.4f}")

# 7. 抖动指标
print("\n─── 7. 抖动指标 ───")
for j in ['knee_pitch', 'hip_pitch']:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    acc_l = np.diff(l)
    acc_r = np.diff(r)
    fs = 100; n = len(l)
    freqs, psd_l = signal.welch(l, fs=fs, nperseg=min(2000, n))
    _, psd_r = signal.welch(r, fs=fs, nperseg=min(2000, n))
    hf_l = np.sum(psd_l[freqs>5])/np.sum(psd_l)*100
    hf_r = np.sum(psd_r[freqs>5])/np.sum(psd_r)*100
    print(f"  {j}: 左HF={hf_l:.1f}% 右HF={hf_r:.1f}%  左acc峰值={np.max(np.abs(acc_l)):.3f}")

# 8. 接触状态
print("\n─── 8. 接触状态 ───")
lc = df['left_contact'].values
rc = df['right_contact'].values
print(f"  左脚接触率: {np.mean(lc>0)*100:.1f}%")
print(f"  右脚接触率: {np.mean(rc>0)*100:.1f}%")
print(f"  双脚同时接触率: {np.mean((lc>0)&(rc>0))*100:.1f}%")

# 9. 躯干稳定性
print("\n─── 9. 躯干稳定性 ───")
print(f"  base_ang_vel_x std: {np.std(df['base_ang_vel_x'].values):.4f}")
print(f"  base_ang_vel_y std: {np.std(df['base_ang_vel_y'].values):.4f}")
print(f"  base_euler_x std: {np.std(df['base_euler_x'].values):.4f}")

# 10. 跟踪率
print("\n─── 10. 跟踪率 (|actual - des_raw| < 0.1) ───")
for j in ['hip_pitch', 'knee_pitch', 'ankle_pitch']:
    l_act = df[f'pos_left_{j}_joint'].values
    l_des = df[f'pos_des_raw_left_{j}_joint'].values
    r_act = df[f'pos_right_{j}_joint'].values
    r_des = df[f'pos_des_raw_right_{j}_joint'].values
    lt = np.mean(np.abs(l_act - l_des) < 0.1) * 100
    rt = np.mean(np.abs(r_act - r_des) < 0.1) * 100
    print(f"  {j:15s}: 左={lt:.1f}%  右={rt:.1f}%")

# 11. 对比
print("\n─── 11. 历史对比 ───")
print(f"{'指标':<25s} {'5.8':>10s} {'6.0(Bug)':>10s} {'6.1':>10s}")
print("-" * 55)
print(f"{'速度vx均值':<25s} {'0.32':>10s} {'0.048':>10s} {np.mean(blv):>10.3f}")
print(f"{'左膝范围':<25s} {'0.604':>10s} {'0.364':>10s} {joint_range(df['pos_left_knee_pitch_joint'].values):>10.3f}")
print(f"{'右膝范围':<25s} {'0.311':>10s} {'0.205':>10s} {joint_range(df['pos_right_knee_pitch_joint'].values):>10.3f}")
print(f"{'左踝pitch范围':<25s} {'—':>10s} {'0.269':>10s} {joint_range(df['pos_left_ankle_pitch_joint'].values):>10.3f}")
print(f"{'右踝pitch范围':<25s} {'—':>10s} {'0.145':>10s} {joint_range(df['pos_right_ankle_pitch_joint'].values):>10.3f}")
lr = joint_range(df['pos_left_knee_pitch_joint'].values)
rr = joint_range(df['pos_right_knee_pitch_joint'].values)
print(f"{'左右膝范围比':<25s} {'1.9:1':>10s} {'1.8:1':>10s} {lr/max(rr,0.001):>10.1f}:1")
