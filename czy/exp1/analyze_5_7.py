import numpy as np
import pandas as pd
from scipy import signal

df = pd.read_csv(r'e:\X1\real_test\origin_exp1\agibot_x1_train-main\czy\data\isaac_diag_20260618_152415.csv')
print(f"Shape: {df.shape}")
print()

# --- Key columns ---
left_hip_pitch = df['pos_left_hip_pitch_joint'].values
left_knee_pitch = df['pos_left_knee_pitch_joint'].values
left_ankle_pitch = df['pos_left_ankle_pitch_joint'].values
right_hip_pitch = df['pos_right_hip_pitch_joint'].values
right_knee_pitch = df['pos_right_knee_pitch_joint'].values
right_ankle_pitch = df['pos_right_ankle_pitch_joint'].values

base_ang_vel_x = df['base_ang_vel_x'].values
base_euler_z = df['base_euler_z'].values
cmd_vel = df['cmd_linear_x'].values
ref_foot_x = df['ref_foot_pos'][:, 0, 0].values if 'ref_foot_pos' in df.columns else None
ref_dof = df['ref_dof_pos'][:, :, 0].values if 'ref_dof_pos' in df.columns else None

# --- Joint Range ---
def joint_range(arr):
    return arr.max() - arr.min()

print("=" * 60)
print("【实验 5.7 诊断】isaac_diag_20260618_152415.csv")
print("=" * 60)
print()
print("─── 关节实际范围 ───")
print(f"左髋 pitch  : {joint_range(left_hip_pitch):.4f} rad")
print(f"左膝 pitch  : {joint_range(left_knee_pitch):.4f} rad")
print(f"左踝 pitch  : {joint_range(left_ankle_pitch):.4f} rad")
print(f"右髋 pitch  : {joint_range(right_hip_pitch):.4f} rad")
print(f"右膝 pitch  : {joint_range(right_knee_pitch):.4f} rad")
print(f"右踝 pitch  : {joint_range(right_ankle_pitch):.4f} rad")
print()

# --- hip_yaw / hip_roll ---
if 'pos_left_hip_yaw_joint' in df.columns:
    lh_yaw = df['pos_left_hip_yaw_joint'].values
    lh_roll = df['pos_left_hip_roll_joint'].values
    rh_yaw = df['pos_right_hip_yaw_joint'].values
    rh_roll = df['pos_right_hip_roll_joint'].values
    print("─── hip_yaw / hip_roll ───")
    print(f"左髋 yaw  : {joint_range(lh_yaw):.4f} rad")
    print(f"左髋 roll : {joint_range(lh_roll):.4f} rad")
    print(f"右髋 yaw  : {joint_range(rh_yaw):.4f} rad")
    print(f"右髋 roll : {joint_range(rh_roll):.4f} rad")
print()

# --- pos_des_raw vs actual ---
lp_action = df['action_left_hip_pitch_joint'].values
lp_actual = left_hip_pitch
lp_des_raw = df['pos_des_raw_left_hip_pitch_joint'].values
lp_tracking = np.mean(np.abs(lp_actual - lp_des_raw) < 0.1) * 100

kp_action = df['action_left_knee_pitch_joint'].values
kp_actual = left_knee_pitch
kp_des_raw = df['pos_des_raw_left_knee_pitch_joint'].values
kp_tracking = np.mean(np.abs(kp_actual - kp_des_raw) < 0.1) * 100

ap_action = df['action_left_ankle_pitch_joint'].values
ap_actual = left_ankle_pitch
ap_des_raw = df['pos_des_raw_left_ankle_pitch_joint'].values
ap_tracking = np.mean(np.abs(ap_actual - ap_des_raw) < 0.1) * 100

print("─── 跟踪率 (|actual - des_raw| < 0.1) ───")
print(f"左髋 pitch  : {lp_tracking:.1f}%")
print(f"左膝 pitch  : {kp_tracking:.1f}%")
print(f"左踝 pitch  : {ap_tracking:.1f}%")
print()

# --- 速度跟踪 ---
v_track = np.mean(np.abs(cmd_vel) > 0) * 100
print(f"命令速度 > 0: {v_track:.1f}%")
print()

# --- 抖动指标 ---
# Acceleration
acc = np.diff(left_knee_pitch)
acc_mean_abs = np.mean(np.abs(acc))
acc_max = np.max(np.abs(acc))
print(f"─── 抖动指标（左膝）───")
print(f"加速度均值绝对值: {acc_mean_abs:.2f} rad/s²")
print(f"加速度峰值: {acc_max:.2f} rad/s²")

# FFT for high-frequency energy
fs = 100  # sampling rate estimate ~100Hz from phase resolution
n = len(left_knee_pitch)
freqs, psd = signal.welch(left_knee_pitch, fs=fs, nperseg=min(2000, n))
high_freq_mask = freqs > 5
high_freq_energy = np.sum(psd[high_freq_mask])
total_energy = np.sum(psd)
high_freq_pct = high_freq_energy / total_energy * 100 if total_energy > 0 else 0

print(f"高频能量占比 (>5Hz): {high_freq_pct:.1f}%")

# Base angular velocity std
bav_std = np.std(base_ang_vel_x)
print(f"base_ang_vel_x std: {bav_std:.4f}")
print()

# --- Gait cycle estimation ---
phase = df['phase_sin'].values
cycle_periods = []
for i in range(1, len(phase)):
    delta = (phase[i] - phase[i-1]) % (2*np.pi)
    if delta > 0 and delta < np.pi:
        cycle_periods.append(delta)
if cycle_periods:
    avg_cycle = np.mean(cycle_periods)
    print(f"─── 步态周期估计 ───")
    print(f"平均相位差: {avg_cycle:.4f} rad → 约 {avg_cycle/(2*np.pi)*0.7:.3f} s (假设周期~0.7s)")
else:
    print("无法估算步态周期")
print()

# --- action range (pos_des_raw) ---
print("─── pos_des_raw 范围（策略输出目标）───")
print(f"左髋 pitch  : [{lp_des_raw.min():.3f}, {lp_des_raw.max():.3f}]")
print(f"左膝 pitch  : [{kp_des_raw.min():.3f}, {kp_des_raw.max():.3f}]")
print(f"左踝 pitch  : [{ap_des_raw.min():.3f}, {ap_des_raw.max():.3f}]")
print()

# --- Compare with 5.6 ---
print("=" * 60)
print("【对比：5.6 vs 5.7】")
print("=" * 60)
compare = {
    '左膝实际范围': (0.310, 0.319),  # approx from 5.6 data
    '左膝加速度均值': (33.15, None),
    '左膝高频能量%': (22.4, None),
    'base_ang_vel_x std': (0.427, None),
    '左髋 hip_pitch des范围': (2.841, None),
    '左髋 hip_roll des范围': (2.838, None),
}
print("注：5.6的精确数值需重新计算，此处为近似对比")
