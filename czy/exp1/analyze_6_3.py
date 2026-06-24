import numpy as np
import pandas as pd
from scipy import signal

df = pd.read_csv(r'e:\X1\real_test\origin_exp1\agibot_x1_train-main\czy\data\isaac_diag_20260624_094900.csv')
print(f"Shape: {df.shape}")

def joint_range(arr):
    return arr.max() - arr.min()

joints = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll']

print("=" * 70)
print("【实验 6.3 诊断】isaac_diag_20260624_094900.csv")
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

# 5. 抖动指标
print("\n─── 5. 抖动指标 ───")
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

# 6. 接触状态
print("\n─── 6. 接触状态 ───")
lc = df['left_contact'].values
rc = df['right_contact'].values
print(f"  左脚接触率: {np.mean(lc>0)*100:.1f}%")
print(f"  右脚接触率: {np.mean(rc>0)*100:.1f}%")
print(f"  双脚同时接触率: {np.mean((lc>0)&(rc>0))*100:.1f}%")

# 7. 躯干稳定性
print("\n─── 7. 躯干稳定性 ───")
print(f"  base_ang_vel_x std: {np.std(df['base_ang_vel_x'].values):.4f}")
print(f"  base_ang_vel_y std: {np.std(df['base_ang_vel_y'].values):.4f}")
print(f"  base_euler_x 范围: {joint_range(df['base_euler_x'].values):.4f}")

# 8. 步态周期分析
print("\n─── 8. 步态周期分析 ───")
phase = np.arctan2(df['phase_sin'].values, df['phase_cos'].values)
phase_unwrapped = np.unwrap(phase)
phase_diff = np.diff(phase_unwrapped)
period_frames = 2 * np.pi / np.mean(phase_diff[phase_diff > 0])
period_seconds = period_frames / 100  # 100 Hz
print(f"  平均步态周期: {period_seconds:.3f} s")
print(f"  步频: {1/period_seconds:.2f} Hz")

# 9. Action 输出
print("\n─── 9. Action 输出范围 ───")
for j in ['hip_pitch', 'knee_pitch']:
    l = df[f'action_left_{j}_joint'].values
    r = df[f'action_right_{j}_joint'].values
    print(f"  {j:15s}: 左={joint_range(l):.4f}  右={joint_range(r):.4f}")

# 10. 对比历史
print("\n─── 10. 历史对比 ───")
print(f"{'指标':<25s} {'6.1':>10s} {'6.3':>10s}")
print("-" * 50)
print(f"{'速度vx均值':<25s} {'0.258':>10s} {np.mean(blv):>10.3f}")
print(f"{'左膝范围':<25s} {'0.460':>10s} {joint_range(df['pos_left_knee_pitch_joint'].values):>10.3f}")
print(f"{'右膝范围':<25s} {'0.282':>10s} {joint_range(df['pos_right_knee_pitch_joint'].values):>10.3f}")
print(f"{'步态周期':<25s} {'—':>10s} {period_seconds:>10.3f}s")
