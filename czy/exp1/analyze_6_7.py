import numpy as np
import pandas as pd
from scipy import signal

CSV = r'e:\EdgeDownLoad\isaac_diag_20260626_112942.csv'
df = pd.read_csv(CSV)
print(f"Shape: {df.shape}")
duration = (df.timestamp_ns.iloc[-1] - df.timestamp_ns.iloc[0]) / 1e9
print(f"Duration: {duration:.1f}s")


def joint_range(arr):
    return arr.max() - arr.min()


joints = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll']
defaults = {
    'hip_pitch': (0.4, -0.4),
    'hip_roll': (0.05, -0.05),
    'hip_yaw': (-0.31, 0.31),
    'knee_pitch': (0.49, 0.49),
    'ankle_pitch': (-0.21, -0.21),
    'ankle_roll': (0.0, 0.0),
}

print("=" * 70)
print("[Exp 6.7] isaac_diag_20260626_112942.csv")
print("=" * 70)

# 1. velocity
print("\n--- 1. Velocity ---")
cmd = df['cmd_linear_x'].values
blv = df['base_lin_vel_x'].values
bly = df['base_lin_vel_y'].values
print(f"  cmd_linear_x mean: {np.mean(cmd):.3f}")
print(f"  base_lin_vel_x mean: {np.mean(blv):.3f}")
print(f"  base_lin_vel_x std: {np.std(blv):.3f}")
print(f"  base_lin_vel_y mean: {np.mean(bly):.3f}")
print(f"  tracking ratio: {np.mean(blv)/np.mean(cmd)*100:.1f}%")

# 2. joint range
print("\n--- 2. Joint range ---")
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    print(f"  {j:15s}: L={joint_range(l):.4f}  R={joint_range(r):.4f}")

# 3. joint mean
print("\n--- 3. Joint mean vs default ---")
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    dl, dr = defaults[j]
    print(f"  {j:15s}: L={np.mean(l):+.4f}(def{dl:+.2f})  R={np.mean(r):+.4f}(def{dr:+.2f})")

# 4. hip offset
print("\n--- 4. hip_roll/yaw offset ---")
for j in ['hip_roll', 'hip_yaw']:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    dl, dr = defaults[j]
    print(f"  {j}: L_offset={np.abs(np.mean(l) - dl):.4f}  R_offset={np.abs(np.mean(r) - dr):.4f}")

# 5. swing/stance knee
print("\n--- 5. Swing/stance knee ---")
phase = np.arctan2(df['phase_sin'].values, df['phase_cos'].values)
left_swing = (phase > 0) & (phase < np.pi)
right_swing = (phase < 0) | (phase > np.pi)
lk = df['pos_left_knee_pitch_joint'].values
rk = df['pos_right_knee_pitch_joint'].values
print(f"  L swing range: {lk[left_swing].max()-lk[left_swing].min():.4f}  stance: {lk[~left_swing].max()-lk[~left_swing].min():.4f}")
print(f"  R swing range: {rk[right_swing].max()-rk[right_swing].min():.4f}  stance: {rk[~right_swing].max()-rk[~right_swing].min():.4f}")

# 6. jitter
print("\n--- 6. Jitter (HF>5Hz PSD) ---")
for j in ['knee_pitch', 'hip_pitch', 'ankle_roll']:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    fs = 100
    n = len(l)
    freqs, psd_l = signal.welch(l, fs=fs, nperseg=min(2000, n))
    _, psd_r = signal.welch(r, fs=fs, nperseg=min(2000, n))
    hf_l = np.sum(psd_l[freqs > 5]) / np.sum(psd_l) * 100
    hf_r = np.sum(psd_r[freqs > 5]) / np.sum(psd_r) * 100
    print(f"  {j}: L_HF={hf_l:.2f}% R_HF={hf_r:.2f}%")

# 7. contact
print("\n--- 7. Contact ---")
lc = df['left_contact'].values
rc = df['right_contact'].values
dual = (lc > 0) & (rc > 0)
single = (lc > 0) ^ (rc > 0)
print(f"  left contact: {np.mean(lc > 0) * 100:.1f}%")
print(f"  right contact: {np.mean(rc > 0) * 100:.1f}%")
print(f"  dual contact: {np.mean(dual) * 100:.1f}%")
print(f"  single support: {np.mean(single) * 100:.1f}%")
print(f"  vx dual contact: {df.loc[dual, 'base_lin_vel_x'].mean():.3f}")
print(f"  vx single support: {df.loc[single, 'base_lin_vel_x'].mean():.3f}")

# 8. trunk
print("\n--- 8. Trunk stability ---")
print(f"  base_ang_vel_x std: {np.std(df['base_ang_vel_x'].values):.4f}")
print(f"  base_ang_vel_y std: {np.std(df['base_ang_vel_y'].values):.4f}")
print(f"  base_ang_vel_z std: {np.std(df['base_ang_vel_z'].values):.4f}")
print(f"  base_euler_x range: {joint_range(df['base_euler_x'].values):.4f}")
print(f"  base_euler_y range: {joint_range(df['base_euler_y'].values):.4f}")

# 9. action
print("\n--- 9. Action range ---")
for j in ['hip_pitch', 'knee_pitch', 'ankle_pitch']:
    l = df[f'action_left_{j}_joint'].values
    r = df[f'action_right_{j}_joint'].values
    print(f"  {j:15s}: L={joint_range(l):.4f}  R={joint_range(r):.4f}")
print(f"  clip_count delta: {df['clip_count'].iloc[-1] - df['clip_count'].iloc[0]}")

# 10. gait period
print("\n--- 10. Gait period ---")
phase_unwrapped = np.unwrap(phase)
phase_diff = np.diff(phase_unwrapped)
period_frames = 2 * np.pi / np.mean(phase_diff[phase_diff > 0])
period_seconds = period_frames / 100
print(f"  avg gait period: {period_seconds:.3f} s")
print(f"  step freq: {1/period_seconds:.2f} Hz")

# 11. history
print("\n--- 11. History compare ---")
lr = joint_range(lk)
rr = joint_range(rk)
fs = 100
n = len(lk)
freqs, psd = signal.welch(lk, fs=fs, nperseg=min(2000, n))
hf_l = np.sum(psd[freqs > 5]) / np.sum(psd) * 100
_, psd = signal.welch(rk, fs=fs, nperseg=min(2000, n))
hf_r = np.sum(psd[freqs > 5]) / np.sum(psd) * 100
dual_pct = np.mean(dual) * 100
hy_l = np.abs(np.mean(df['pos_left_hip_yaw_joint']) - defaults['hip_yaw'][0])
hr_r = np.abs(np.mean(df['pos_right_hip_roll_joint']) - defaults['hip_roll'][1])

print(f"{'metric':<25s} {'6.4':>8s} {'6.5':>8s} {'6.6':>8s} {'6.7':>8s}")
print("-" * 62)
print(f"{'vx mean':<25s} {'0.368':>8s} {'0.213':>8s} {'0.132':>8s} {np.mean(blv):>8.3f}")
print(f"{'L knee range':<25s} {'0.673':>8s} {'0.176':>8s} {'0.244':>8s} {lr:>8.3f}")
print(f"{'R knee range':<25s} {'0.531':>8s} {'0.261':>8s} {'0.262':>8s} {rr:>8.3f}")
print(f"{'L knee HF':<25s} {'1.3%':>8s} {'1.0%':>8s} {'0.8%':>8s} {hf_l:>7.1f}%")
print(f"{'R knee HF':<25s} {'1.3%':>8s} {'3.1%':>8s} {'1.1%':>8s} {hf_r:>7.1f}%")
print(f"{'dual contact':<25s} {'10.3%':>8s} {'22.2%':>8s} {'64.7%':>8s} {dual_pct:>7.1f}%")
print(f"{'L/R knee ratio':<25s} {'1.3:1':>8s} {'0.67:1':>8s} {'0.93:1':>8s} {lr/max(rr,0.001):>7.2f}:1")
print(f"{'L hip_yaw offset':<25s} {'0.71':>8s} {'0.34':>8s} {'0.02':>8s} {hy_l:>8.3f}")
print(f"{'R hip_roll offset':<25s} {'0.31':>8s} {'0.38':>8s} {'0.02':>8s} {hr_r:>8.3f}")
print(f"{'ang_vel_x std':<25s} {'0.993':>8s} {'--':>8s} {'0.347':>8s} {np.std(df['base_ang_vel_x'].values):>8.3f}")
