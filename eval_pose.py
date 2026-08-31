# -*- coding: utf-8 -*-
"""灵龙2.0 重映射姿态评估脚本：MPJPE（平均关节位置误差）+ 分部位 + 每关节误差

用法（wham_gmr 终端，仓库根目录）：
    python eval_pose.py --csv output/linglong2_run/csv/linglong2/live_motion.csv \
                        --npz output/linglong2_run/stream_demo/gmr_smplx_results.npz

原理：
    机器人侧：CSV(37列) -> MuJoCo FK -> 各 link 全局位置
    人体侧：npz(SMPL-X) -> smplx FK -> 各关节全局位置
    对齐：pelvis<->base_link 根对齐 + 按 hip-knee 长度做比例缩放（消除人机尺寸差）
    输出：MPJPE 总误差 + 分部位 + 每关节误差（mm）
"""
import argparse
import numpy as np
import mujoco
import torch
import smplx
from smplx.joint_names import JOINT_NAMES

ap = argparse.ArgumentParser()
ap.add_argument('--csv', required=True, help='live_motion.csv 路径')
ap.add_argument('--npz', required=True, help='gmr_smplx_results.npz 路径')
ap.add_argument('--xml', default='assets/LingLong2.0/LingLong2.0.xml', help='机器人 MJCF')
args = ap.parse_args()

# ---- 关节映射（与 smplx_to_linglong2.json 的 ik_match_table 一致）----
link_map = [
    'base_link', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
    'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'waist_yaw_link',
    'left_shoulder_pitch_link', 'left_elbow_link', 'left_wrist_yaw_link',
    'right_shoulder_pitch_link', 'right_elbow_link', 'right_wrist_yaw_link',
]
smpl_map = [
    'pelvis', 'left_hip', 'left_knee', 'left_foot',
    'right_hip', 'right_knee', 'right_foot', 'spine3',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist',
]

# ---- 读 CSV（37 列：pos3 + quat_xyzw4 + dof30）----
rows = []
with open(args.csv, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 37:
            rows.append(np.array([float(x) for x in parts[:37]]))
print('CSV rows:', len(rows))

# ---- 读 npz ----
data = np.load(args.npz, allow_pickle=True)
pose_body = data['pose_body'].astype(np.float32)
root_orient = data['root_orient'].astype(np.float32)
betas = np.asarray(data['betas'], dtype=np.float32).reshape(1, -1)
print('npz frames:', len(pose_body))

N = min(len(rows), len(pose_body))
print('evaluated frames:', N)

# ---- 人体 FK（smplx，与 handle_wham_gmr.py 相同的输入约定）----
bm = smplx.create('assets/body_models', 'smplx', gender='neutral', use_pca=False)
jnames = list(JOINT_NAMES[:22])

m = mujoco.MjModel.from_xml_path(args.xml)
d = mujoco.MjData(m)
body_ids = {ln: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ln) for ln in link_map}

robot_pos = np.zeros((N, len(link_map), 3), dtype=np.float64)
smpl_pos = np.zeros((N, len(link_map), 3), dtype=np.float64)

for i in range(N):
    # 机器人 FK
    q = rows[i]
    d.qpos[:3] = q[0:3]
    d.qpos[3:7] = q[3:7][[3, 0, 1, 2]]  # xyzw -> wxyz
    d.qpos[7:37] = q[7:37]
    mujoco.mj_forward(m, d)
    for j, ln in enumerate(link_map):
        robot_pos[i, j] = d.xpos[body_ids[ln]]

    # 人体 FK
    with torch.no_grad():
        out = bm(
            body_pose=torch.from_numpy(pose_body[i:i + 1]),
            global_orient=torch.from_numpy(root_orient[i:i + 1]),
            betas=torch.from_numpy(betas),
        )
    sj = out.joints[0].detach().cpu().numpy()
    for j, jn in enumerate(smpl_map):
        smpl_pos[i, j] = sj[jnames.index(jn)]

# ---- 根对齐 + 比例缩放（pelvis<->base_link；按 hip-knee 长度缩放）----
scale = (np.linalg.norm(robot_pos[:, 1] - robot_pos[:, 2], axis=1)
         / np.maximum(np.linalg.norm(smpl_pos[:, 1] - smpl_pos[:, 2], axis=1), 1e-6))
for i in range(N):
    smpl_pos[i] = (smpl_pos[i] - smpl_pos[i, 0]) * scale[i] + robot_pos[i, 0]

err = np.linalg.norm(smpl_pos - robot_pos, axis=2) * 1000.0  # mm
print('\n========== 评估结果 ==========')
print(f'MPJPE 总误差      : {err.mean():7.1f} mm')
parts = {
    'root(骨盆)     ': [0],
    '腿(髋膝踝)     ': [1, 2, 3, 4, 5, 6],
    '腰            ': [7],
    '臂(肩肘腕)     ': [8, 9, 10, 11, 12, 13],
}
for name, idx in parts.items():
    print(f'  {name}  : {err[:, idx].mean():7.1f} mm')
print('\n--- 每关节误差 ---')
for j, ln in enumerate(link_map):
    print(f'  {ln:28s} {err[:, j].mean():7.1f} mm')
