"""
已废弃：请改用 isaac-diag-eval skill。
  python czy/skills/skills/isaac-diag-eval/eval_isaac_diag.py czy/data/isaac_diag_20260623_104848.csv
保留本文件仅作 5.8 历史复现参考。
"""
import numpy as np
import pandas as pd
from scipy import signal

df = pd.read_csv(r'e:\X1\real_test\origin_exp1\agibot_x1_train-main\czy\data\isaac_diag_20260623_104848.csv')
print(f"Shape: {df.shape}")
print()

def joint_range(arr):
    return arr.max() - arr.min()

# ========== 1. 关节实际范围 ==========
print("=" * 70)
print("【实验 5.8 诊断】isaac_diag_20260623_104848.csv")
print("=" * 70)

joints = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll']
print("\n─── 1. 关节实际范围 ───")
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    print(f"  {j:15s}: 左={joint_range(l):.4f}  右={joint_range(r):.4f}")

# ========== 2. 对称性 ==========
print("\n─── 2. 左右腿对称性（实际角度均值）───")
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    print(f"  {j:15s}: 左={np.mean(l):+.4f}  右={np.mean(r):+.4f}  差={np.mean(l)-np.mean(r):+.4f}")

# ========== 3. hip_roll/yaw 偏移 ==========
print("\n─── 3. hip_roll/yaw 偏移（关键不对称指标）───")
for j in ['hip_roll', 'hip_yaw']:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    l_action = df[f'action_left_{j}_joint'].values
    r_action = df[f'action_right_{j}_joint'].values
    l_des = df[f'pos_des_raw_left_{j}_joint'].values
    r_des = df[f'pos_des_raw_right_{j}_joint'].values
    print(f"  {j}:")
    print(f"    实际: 左={np.mean(l):+.4f}  右={np.mean(r):+.4f}")
    print(f"    action: 左={np.mean(l_action):+.4f}  右={np.mean(r_action):+.4f}")
    print(f"    pos_des_raw: 左={np.mean(l_des):+.4f}  右={np.mean(r_des):+.4f}")

# ========== 4. 抖动指标 ==========
print("\n─── 4. 抖动指标 ───")
for j in ['knee_pitch', 'hip_pitch']:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    # 加速度
    acc_l = np.diff(l)
    acc_r = np.diff(r)
    # FFT
    fs = 100
    n = len(l)
    freqs, psd_l = signal.welch(l, fs=fs, nperseg=min(2000, n))
    _, psd_r = signal.welch(r, fs=fs, nperseg=min(2000, n))
    high_mask = freqs > 5
    hf_l = np.sum(psd_l[high_mask]) / np.sum(psd_l) * 100
    hf_r = np.sum(psd_r[high_mask]) / np.sum(psd_r) * 100
    print(f"  {j}:")
    print(f"    左: acc均值={np.mean(np.abs(acc_l)):.3f}  acc峰值={np.max(np.abs(acc_l)):.3f}  HF能量={hf_l:.1f}%")
    print(f"    右: acc均值={np.mean(np.abs(acc_r)):.3f}  acc峰值={np.max(np.abs(acc_r)):.3f}  HF能量={hf_r:.1f}%")

# ========== 5. 躯干稳定性 ==========
print("\n─── 5. 躯干稳定性 ───")
bav_x = df['base_ang_vel_x'].values
bav_y = df['base_ang_vel_y'].values
bav_z = df['base_ang_vel_z'].values
print(f"  base_ang_vel_x std: {np.std(bav_x):.4f}")
print(f"  base_ang_vel_y std: {np.std(bav_y):.4f}")
print(f"  base_ang_vel_z std: {np.std(bav_z):.4f}")

# ========== 6. 跟踪率 ==========
print("\n─── 6. 跟踪率 (|actual - des_raw| < 0.1) ───")
for j in ['hip_pitch', 'knee_pitch', 'ankle_pitch']:
    l_act = df[f'pos_left_{j}_joint'].values
    l_des = df[f'pos_des_raw_left_{j}_joint'].values
    r_act = df[f'pos_right_{j}_joint'].values
    r_des = df[f'pos_des_raw_right_{j}_joint'].values
    lt = np.mean(np.abs(l_act - l_des) < 0.1) * 100
    rt = np.mean(np.abs(r_act - r_des) < 0.1) * 100
    print(f"  {j:15s}: 左={lt:.1f}%  右={rt:.1f}%")

# ========== 7. pos_des_raw 范围 ==========
print("\n─── 7. pos_des_raw 范围（策略输出目标）───")
for j in joints:
    l = df[f'pos_des_raw_left_{j}_joint'].values
    r = df[f'pos_des_raw_right_{j}_joint'].values
    print(f"  {j:15s}: 左=[{l.min():.3f}, {l.max():.3f}]  右=[{r.min():.3f}, {r.max():.3f}]")

# ========== 8. 速度跟踪 ==========
print("\n─── 8. 速度跟踪 ───")
cmd = df['cmd_linear_x'].values
blv = df['base_lin_vel_x'].values
print(f"  cmd_linear_x 均值: {np.mean(cmd):.3f}")
print(f"  base_lin_vel_x 均值: {np.mean(blv):.3f}")

# ========== 9. 与 5.7 对比 ==========
print("\n" + "=" * 70)
print("【5.7 vs 5.8 对比】")
print("=" * 70)
print(f"{'指标':<25s} {'5.7':>10s} {'5.8':>10s}")
print("-" * 45)
print(f"{'左膝范围':<25s} {'0.402':>10s} {joint_range(df['pos_left_knee_pitch_joint'].values):>10.3f}")
print(f"{'右膝范围':<25s} {'0.335':>10s} {joint_range(df['pos_right_knee_pitch_joint'].values):>10.3f}")
print(f"{'左髋pitch范围':<25s} {'0.445':>10s} {joint_range(df['pos_left_hip_pitch_joint'].values):>10.3f}")
print(f"{'右髋pitch范围':<25s} {'0.467':>10s} {joint_range(df['pos_right_hip_pitch_joint'].values):>10.3f}")
print(f"{'左髋roll偏移':<25s} {'0.26':>10s} {abs(np.mean(df['pos_left_hip_roll_joint'].values)-0.05):>10.3f}")
print(f"{'右髋roll偏移':<25s} {'0.25':>10s} {abs(np.mean(df['pos_right_hip_roll_joint'].values)+0.05):>10.3f}")
print(f"{'左膝HF能量%':<25s} {'0.9':>10s} ", end="")
fs = 100; n = len(df); freqs, psd = signal.welch(df['pos_left_knee_pitch_joint'].values, fs=fs, nperseg=min(2000, n))
hf = np.sum(psd[freqs>5])/np.sum(psd)*100; print(f"{hf:>10.1f}")
print(f"{'base_ang_vel_x std':<25s} {'0.658':>10s} {np.std(df['base_ang_vel_x'].values):>10.4f}")
