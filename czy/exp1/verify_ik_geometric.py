"""验证严格几何 IK 的正运动学一致性。

方法：
1. 给定默认足端位置，用 IK 计算关节角度
2. 用正运动学（FK）从关节角度反算足端位置
3. 比较 FK 结果与原始足端位置

正运动学公式（2D矢状面）：
  foot_x = L1 * sin(hip_pitch) + L2 * sin(hip_pitch - knee)
  foot_z = -L1 * cos(hip_pitch) - L2 * cos(hip_pitch - knee)

注意：这里的角度定义是 hip_pitch 相对垂直轴偏转，
knee 是膝关节弯曲角度（0=完全伸直）。
"""
import numpy as np

# URDF 参数
L1 = 0.2678  # 大腿
L2 = 0.3068  # 小腿

# 默认关节角度
default_hip_pitch = 0.4
default_knee = 0.49
default_ankle_pitch = -0.21

# 偏移补偿
ik_hip_pitch_offset = 0.0
ik_knee_offset = 0.49
ik_ankle_pitch_offset = -0.21

print("=" * 60)
print("IK 精度验证")
print("=" * 60)

# 测试 1：默认站姿
print("\n─── 测试 1：默认站姿 ───")
# 默认足端位置（相对于髋关节）
# FK 计算：hip_pitch=0.4, knee=0.49
foot_x_default = L1 * np.sin(default_hip_pitch) + L2 * np.sin(default_hip_pitch - default_knee)
foot_z_default = -L1 * np.cos(default_hip_pitch) - L2 * np.cos(default_hip_pitch - default_knee)
print(f"  FK 默认足端位置: x={foot_x_default:.4f}, z={foot_z_default:.4f}")

# IK 逆算：dx=0, dz=0 → r=0
dx, dz = 0.0, 0.0
r = np.sqrt(dx**2 + dz**2)
print(f"  IK 输入: dx={dx}, dz={dz}, r={r}")

# r=0 时余弦定理失效
if r < 0.001:
    print(f"  r≈0，IK 返回默认关节角度")
    hip_angle = 0.0
    knee_angle = 0.0
    ankle_angle = 0.0
    hip_pitch = default_hip_pitch + hip_angle + ik_hip_pitch_offset
    knee_pitch = default_knee + knee_angle + ik_knee_offset
    ankle_pitch = default_ankle_pitch + ankle_angle + ik_ankle_pitch_offset
    print(f"  IK 输出: hip={hip_pitch:.4f}, knee={knee_pitch:.4f}, ankle={ankle_pitch:.4f}")
    print(f"  默认角度: hip={default_hip_pitch:.4f}, knee={default_knee:.4f}, ankle={default_ankle_pitch:.4f}")
    print(f"  ⚠️ 问题：IK 输出 knee={knee_pitch:.4f}，但默认是 {default_knee:.4f}")

# 测试 2：前移 0.06m
print("\n─── 测试 2：前移 0.06m ───")
dx = 0.06
dz = 0.0
r = np.sqrt(dx**2 + dz**2)
r_clamped = r
cos_knee = (L1**2 + L2**2 - r_clamped**2) / (2 * L1 * L2)
cos_knee = np.clip(cos_knee, -1, 1)
knee_angle = np.arccos(cos_knee)
alpha = np.arctan2(dz, dx)
cos_alpha = (L1**2 + r_clamped**2 - L2**2) / (2 * L1 * r_clamped)
cos_alpha = np.clip(cos_alpha, -1, 1)
alpha_offset = np.arccos(cos_alpha)
hip_angle = alpha + alpha_offset
ankle_angle = hip_angle + knee_angle - np.pi/2

# 应用偏移
hip_angle_total = hip_angle + ik_hip_pitch_offset
knee_angle_total = knee_angle + ik_knee_offset
ankle_angle_total = ankle_angle + ik_ankle_pitch_offset

hip_pitch = default_hip_pitch + hip_angle_total
knee_pitch = default_knee + knee_angle_total
ankle_pitch = default_ankle_pitch + ankle_angle_total

print(f"  IK 输入: dx={dx}, dz={dz}, r={r:.4f}")
print(f"  IK 几何解: hip_angle={hip_angle:.4f}, knee_angle={knee_angle:.4f}")
print(f"  偏移补偿后: hip={hip_angle_total:.4f}, knee={knee_angle_total:.4f}, ankle={ankle_angle_total:.4f}")
print(f"  最终关节角: hip={hip_pitch:.4f}, knee={knee_pitch:.4f}, ankle={ankle_pitch:.4f}")

# FK 验证
fk_x = L1 * np.sin(hip_pitch) + L2 * np.sin(hip_pitch - knee_pitch)
fk_z = -L1 * np.cos(hip_pitch) - L2 * np.cos(hip_pitch - knee_pitch)
print(f"  FK 足端位置: x={fk_x:.4f}, z={fk_z:.4f}")
print(f"  目标偏移: dx={dx:.4f}, dz={dz:.4f}")
print(f"  FK-目标误差: x={fk_x-foot_x_default-dx:.4f}, z={fk_z-foot_z_default-dz:.4f}")

# 测试 3：抬脚 0.10m
print("\n─── 测试 3：抬脚 0.10m ───")
dx = 0.0
dz = 0.10
r = np.sqrt(dx**2 + dz**2)
r_clamped = min(r, (L1+L2)*0.95)
cos_knee = (L1**2 + L2**2 - r_clamped**2) / (2 * L1 * L2)
cos_knee = np.clip(cos_knee, -1, 1)
knee_angle = np.arccos(cos_knee)
alpha = np.arctan2(dz, dx)
cos_alpha = (L1**2 + r_clamped**2 - L2**2) / (2 * L1 * r_clamped)
cos_alpha = np.clip(cos_alpha, -1, 1)
alpha_offset = np.arccos(cos_alpha)
hip_angle = alpha + alpha_offset
ankle_angle = hip_angle + knee_angle - np.pi/2

hip_pitch = default_hip_pitch + hip_angle + ik_hip_pitch_offset
knee_pitch = default_knee + knee_angle + ik_knee_offset
ankle_pitch = default_ankle_pitch + ankle_angle + ik_ankle_pitch_offset

print(f"  IK 输入: dx={dx}, dz={dz}, r={r:.4f}")
print(f"  IK 几何解: hip_angle={hip_angle:.4f}, knee_angle={knee_angle:.4f}")
print(f"  最终关节角: hip={hip_pitch:.4f}, knee={knee_pitch:.4f}, ankle={ankle_pitch:.4f}")

fk_x = L1 * np.sin(hip_pitch) + L2 * np.sin(hip_pitch - knee_pitch)
fk_z = -L1 * np.cos(hip_pitch) - L2 * np.cos(hip_pitch - knee_pitch)
print(f"  FK 足端位置: x={fk_x:.4f}, z={fk_z:.4f}")
print(f"  目标偏移: dx={dx:.4f}, dz={dz:.4f}")
print(f"  FK-目标误差: x={fk_x-foot_x_default-dx:.4f}, z={fk_z-foot_z_default-dz:.4f}")

# 测试 4：同时前移+抬脚
print("\n─── 测试 4：前移0.08m+抬脚0.12m ───")
dx = 0.08
dz = 0.12
r = np.sqrt(dx**2 + dz**2)
r_clamped = min(r, (L1+L2)*0.95)
cos_knee = (L1**2 + L2**2 - r_clamped**2) / (2 * L1 * L2)
cos_knee = np.clip(cos_knee, -1, 1)
knee_angle = np.arccos(cos_knee)
alpha = np.arctan2(dz, dx)
cos_alpha = (L1**2 + r_clamped**2 - L2**2) / (2 * L1 * r_clamped)
cos_alpha = np.clip(cos_alpha, -1, 1)
alpha_offset = np.arccos(cos_alpha)
hip_angle = alpha + alpha_offset
ankle_angle = hip_angle + knee_angle - np.pi/2

hip_pitch = default_hip_pitch + hip_angle + ik_hip_pitch_offset
knee_pitch = default_knee + knee_angle + ik_knee_offset
ankle_pitch = default_ankle_pitch + ankle_angle + ik_ankle_pitch_offset

print(f"  IK 输入: dx={dx}, dz={dz}, r={r:.4f}")
print(f"  IK 几何解: hip_angle={hip_angle:.4f}, knee_angle={knee_angle:.4f}")
print(f"  最终关节角: hip={hip_pitch:.4f}, knee={knee_pitch:.4f}, ankle={ankle_pitch:.4f}")

fk_x = L1 * np.sin(hip_pitch) + L2 * np.sin(hip_pitch - knee_pitch)
fk_z = -L1 * np.cos(hip_pitch) - L2 * np.cos(hip_pitch - knee_pitch)
print(f"  FK 足端位置: x={fk_x:.4f}, z={fk_z:.4f}")
print(f"  目标偏移: dx={dx:.4f}, dz={dz:.4f}")
print(f"  FK-目标误差: x={fk_x-foot_x_default-dx:.4f}, z={fk_z-foot_z_default-dz:.4f}")

# 关键问题检查
print("\n─── 关键问题检查 ───")
print(f"  L1+L2 = {L1+L2:.4f}m，最大步长/抬脚不能超过此值")
print(f"  抬脚0.15m时 r = {0.15:.4f}m，是否可达: {0.15 < L1+L2}")
print(f"  抬脚0.10m+前移0.12m时 r = {np.sqrt(0.10**2+0.12**2):.4f}m，是否可达: {np.sqrt(0.10**2+0.12**2) < L1+L2}")
