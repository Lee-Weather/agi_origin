import numpy as np
import pandas as pd
from scipy import signal

df = pd.read_csv(r'e:\X1\real_test\origin_exp1\agibot_x1_train-main\czy\data\isaac_diag_20260624_183924.csv')
print(f"Shape: {df.shape}")
duration = (df.timestamp_ns.iloc[-1] - df.timestamp_ns.iloc[0]) / 1e9
print(f"Duration: {duration:.1f}s")


def joint_range(arr):
    return arr.max() - arr.min()


joints = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll']

print("=" * 70)
print("[Exp 6.6] isaac_diag_20260624_183924.csv")
print("=" * 70)

# 1. velocity
print("\n--- 1. Velocity ---")
cmd = df['cmd_linear_x'].values
blv = df['base_lin_vel_x'].values
print(f"  cmd_linear_x mean: {np.mean(cmd):.3f}")
print(f"  base_lin_vel_x mean: {np.mean(blv):.3f}")
print(f"  base_lin_vel_x std: {np.std(blv):.3f}")
print(f"  tracking ratio: {np.mean(blv)/np.mean(cmd)*100:.1f}%")

# 2. joint range
print("\n--- 2. Joint range ---")
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    print(f"  {j:15s}: L={joint_range(l):.4f}  R={joint_range(r):.4f}")

# 3. joint mean vs default
print("\n--- 3. Joint mean vs default ---")
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
    print(f"  {j:15s}: L={np.mean(l):+.4f}(def{dl:+.2f})  R={np.mean(r):+.4f}(def{dr:+.2f})")

# 4. hip offset
print("\n--- 4. hip_roll/yaw offset ---")
for j in ['hip_roll', 'hip_yaw']:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    dl, dr = defaults[j]
    print(f"  {j}: L_offset={np.abs(np.mean(l) - dl):.4f}  R_offset={np.abs(np.mean(r) - dr):.4f}")

# 5. jitter
print("\n--- 5. Jitter (HF>5Hz PSD) ---")
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

# 6. contact
print("\n--- 6. Contact ---")
lc = df['left_contact'].values
rc = df['right_contact'].values
print(f"  left contact: {np.mean(lc > 0) * 100:.1f}%")
print(f"  right contact: {np.mean(rc > 0) * 100:.1f}%")
print(f"  dual contact: {np.mean((lc > 0) & (rc > 0)) * 100:.1f}%")
print(f"  single support: {np.mean((lc > 0) ^ (rc > 0)) * 100:.1f}%")

# 7. trunk
print("\n--- 7. Trunk stability ---")
print(f"  base_ang_vel_x std: {np.std(df['base_ang_vel_x'].values):.4f}")
print(f"  base_ang_vel_y std: {np.std(df['base_ang_vel_y'].values):.4f}")
print(f"  base_euler_x range: {joint_range(df['base_euler_x'].values):.4f}")
print(f"  base_euler_y range: {joint_range(df['base_euler_y'].values):.4f}")

# 8. action
print("\n--- 8. Action range ---")
for j in joints:
    l = df[f'action_left_{j}_joint'].values
    r = df[f'action_right_{j}_joint'].values
    print(f"  {j:15s}: L={joint_range(l):.4f}  R={joint_range(r):.4f}")

# 9. gait period
print("\n--- 9. Gait period ---")
phase = np.arctan2(df['phase_sin'].values, df['phase_cos'].values)
phase_unwrapped = np.unwrap(phase)
phase_diff = np.diff(phase_unwrapped)
period_frames = 2 * np.pi / np.mean(phase_diff[phase_diff > 0])
period_seconds = period_frames / 100
print(f"  avg gait period: {period_seconds:.3f} s")
print(f"  step freq: {1/period_seconds:.2f} Hz")

# 10. contact force
print("\n--- 10. Foot contact force ---")
print(f"  left Fz mean: {df['left_foot_contact_force_z'].mean():.1f} N")
print(f"  right Fz mean: {df['right_foot_contact_force_z'].mean():.1f} N")

# 11. history compare
print("\n--- 11. History compare ---")
lr = joint_range(df['pos_left_knee_pitch_joint'].values)
rr = joint_range(df['pos_right_knee_pitch_joint'].values)
l = df['pos_left_knee_pitch_joint'].values
r = df['pos_right_knee_pitch_joint'].values
fs = 100
n = len(l)
freqs, psd = signal.welch(l, fs=fs, nperseg=min(2000, n))
hf_l = np.sum(psd[freqs > 5]) / np.sum(psd) * 100
_, psd = signal.welch(r, fs=fs, nperseg=min(2000, n))
hf_r = np.sum(psd[freqs > 5]) / np.sum(psd) * 100
dual = np.mean((lc > 0) & (rc > 0)) * 100

header = f"{'metric':<25s} {'6.3':>10s} {'6.4':>10s} {'6.5':>10s} {'6.6':>10s}"
print(header)
print("-" * 70)
print(f"{'vx mean':<25s} {'0.430':>10s} {'0.368':>10s} {'0.213':>10s} {np.mean(blv):>10.3f}")
print(f"{'L knee range':<25s} {'0.303':>10s} {'0.673':>10s} {'0.176':>10s} {lr:>10.3f}")
print(f"{'R knee range':<25s} {'0.333':>10s} {'0.531':>10s} {'0.261':>10s} {rr:>10.3f}")
print(f"{'L knee HF':<25s} {'12.2%':>10s} {'1.3%':>10s} {'1.0%':>10s} {hf_l:>9.1f}%")
print(f"{'R knee HF':<25s} {'11.1%':>10s} {'1.3%':>10s} {'3.1%':>10s} {hf_r:>9.1f}%")
print(f"{'dual contact':<25s} {'--':>10s} {'--':>10s} {'22.2%':>10s} {dual:>9.1f}%")
print(f"{'L/R knee ratio':<25s} {'1.1:1':>10s} {'1.3:1':>10s} {'0.67:1':>10s} {lr/max(rr,0.001):>9.2f}:1")
