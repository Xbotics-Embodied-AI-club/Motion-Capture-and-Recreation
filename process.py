# -*- coding: utf-8 -*-
"""灵龙2.0 的 GMR IK 旋转 offset 推导脚本
原理：GMR 的目标旋转 = R_smpl(全局) × offset，在 T-pose 下反推：
    offset = (rot_fix * R_smpl_Tpose)⁻¹ × R_link(零位姿)
其中 rot_fix 为 Y-up -> Z-up 坐标转换（与 handle_wham_gmr.py _tail_apply_yup_to_zup 一致）。
运行后自动更新 general_motion_retargeting/ik_configs/smplx_to_linglong2.json
"""
import json
import io
import numpy as np
import mujoco
import torch
import smplx
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

# ---- 1. 灵龙2.0 零位姿(qpos=0)各 link 全局旋转（MuJoCo，Z-up）----
xml = 'assets/LingLong2.0/LingLong2.0.xml'
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)


def link_rot(body_name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert i >= 0, f'body not found: {body_name}'
    return R.from_matrix(d.xmat[i].reshape(3, 3))


# ---- 2. SMPL-X T-pose 各关节全局旋转（Y-up，再转 Z-up）----
bm = smplx.create('assets/body_models', 'smplx', gender='neutral', use_pca=False)
with torch.no_grad():
    out = bm(body_pose=torch.zeros(1, 63), global_orient=torch.zeros(1, 3), betas=torch.zeros(1, 10), return_full_pose=True)

so = out.global_orient[0].detach().cpu().numpy()          # 根旋转（轴角）
fp = out.full_pose[0].detach().cpu().numpy().reshape(-1, 3)  # 全身轴角
jnames = list(JOINT_NAMES[:len(bm.parents)])

jorients = []
for i in range(len(jnames)):
    if i == 0:
        jorients.append(R.from_rotvec(so))
    else:
        jorients.append(jorients[int(bm.parents[i])] * R.from_rotvec(fp[i]))

# Y-up -> Z-up（绕 X 轴，与 handle_wham_gmr.py 的 rotation_matrix 一致）
rot_fix = R.from_matrix(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float))

# ---- 3. 计算 offset ----
# GMR 的目标旋转 = R_smpl(handle 约定) * offset，其中 handle 约定下 T-pose 各关节全局旋转 = 单位阵
# （根旋转做了 yup_to_zup 但根为 0 仍是单位阵，局部旋转未转换且为 0）
# 因此 T-pose 对齐要求：offset = R_link(零位姿)，直接从 MuJoCo 取，不做任何坐标变换
mapping = {
    'base_link': 'pelvis',
    'left_hip_roll_link': 'left_hip',
    'left_knee_link': 'left_knee',
    'left_ankle_roll_link': 'left_foot',
    'right_hip_roll_link': 'right_hip',
    'right_knee_link': 'right_knee',
    'right_ankle_roll_link': 'right_foot',
    'waist_yaw_link': 'spine3',
    'left_shoulder_pitch_link': 'left_shoulder',
    'left_elbow_link': 'left_elbow',
    'left_wrist_yaw_link': 'left_wrist',
    'right_shoulder_pitch_link': 'right_shoulder',
    'right_elbow_link': 'right_elbow',
    'right_wrist_yaw_link': 'right_wrist',
}
offsets = {}
for link, joint in mapping.items():
    offset = link_rot(link)  # R_link(qpos=0)
    q = offset.as_quat()  # xyzw
    offsets[link] = [round(q[3], 8), round(q[0], 8), round(q[1], 8), round(q[2], 8)]  # wxyz

# ---- 4. 更新 smplx_to_linglong2.json（ik_match_table1/2 的旋转偏移）----
p = 'general_motion_retargeting/ik_configs/smplx_to_linglong2.json'
cfg = json.load(io.open(p, encoding='utf-8'))
for table in ('ik_match_table1', 'ik_match_table2'):
    for link in cfg[table]:
        if link in offsets:
            cfg[table][link][4] = offsets[link]
io.open(p, 'w', encoding='utf-8').write(json.dumps(cfg, indent=2, ensure_ascii=False))
print('offsets updated OK ->', p)

print(json.dumps(offsets, indent=1))
