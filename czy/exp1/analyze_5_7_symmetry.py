import numpy as np
import pandas as pd

df = pd.read_csv(r'e:\X1\real_test\origin_exp1\agibot_x1_train-main\czy\data\isaac_diag_20260618_152415.csv')

joints = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll']

print("=" * 70)
print("【5.7 左右腿对称性分析】")
print("=" * 70)

print("\n─── 1. 关节角度均值（默认姿态差异）───")
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    print(f"  {j:15s}: 左={np.mean(l):+.4f}  右={np.mean(r):+.4f}  差={np.mean(l)-np.mean(r):+.4f}")

print("\n─── 2. 关节角度范围（运动幅度差异）───")
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    lr = l.max() - l.min()
    rr = r.max() - r.min()
    print(f"  {j:15s}: 左={lr:.4f}  右={rr:.4f}  差={lr-rr:+.4f}  比值={lr/rr:.2f}" if rr > 0.001 else f"  {j:15s}: 左={lr:.4f}  右={rr:.4f}")

print("\n─── 3. pos_des_raw 均值（策略输出不对称）───")
for j in joints:
    l = df[f'pos_des_raw_left_{j}_joint'].values
    r = df[f'pos_des_raw_right_{j}_joint'].values
    print(f"  {j:15s}: 左={np.mean(l):+.4f}  右={np.mean(r):+.4f}  差={np.mean(l)-np.mean(r):+.4f}")

print("\n─── 4. pos_des_raw 范围（参考轨迹不对称）───")
for j in joints:
    l = df[f'pos_des_raw_left_{j}_joint'].values
    r = df[f'pos_des_raw_right_{j}_joint'].values
    lr = l.max() - l.min()
    rr = r.max() - r.min()
    print(f"  {j:15s}: 左={lr:.4f}  右={rr:.4f}  差={lr-rr:+.4f}")

print("\n─── 5. action 均值（网络输出不对称）───")
for j in joints:
    l = df[f'action_left_{j}_joint'].values
    r = df[f'action_right_{j}_joint'].values
    print(f"  {j:15s}: 左={np.mean(l):+.4f}  右={np.mean(r):+.4f}  差={np.mean(l)-np.mean(r):+.4f}")

print("\n─── 6. 默认关节角度对比 ───")
# 从数据推断 default_dof_pos
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    # 取前10帧的均值作为初始位置近似
    l_init = np.mean(l[:10])
    r_init = np.mean(r[:10])
    print(f"  {j:15s}: 左≈{l_init:+.4f}  右≈{r_init:+.4f}  差={l_init-r_init:+.4f}")

print("\n─── 7. 逐帧关节角度差异统计 ───")
for j in joints:
    l = df[f'pos_left_{j}_joint'].values
    r = df[f'pos_right_{j}_joint'].values
    diff = l - r
    # 对于pitch关节，左右腿在同一时刻不在同一相，所以直接比较无意义
    # 但均值差异反映默认姿态不对称
    print(f"  {j:15s}: 均值差={np.mean(diff):+.4f}  std={np.std(diff):.4f}  max|差|={np.max(np.abs(diff)):.4f}")
