"""Convert WHAM's `wham_output.pkl` (from demo.py --save_pkl) into an SMPL-X
`.npz` that `scripts/smplx_to_robot.py` understands.

WHAM outputs standard SMPL (24 joints: root + 23 body, 72-dim axis-angle pose).
SMPL-X body joints (root + 21 body) are the *first* 22 SMPL joints, in the same
order, so the body-pose mapping is a straight truncation `pose[:, 3:66]`.

By default this reads the *global* (SLAM world-frame) outputs `pose_world` /
`trans_world`. Pass `--local` to use the camera-local `pose` / `trans` instead.

Usage:
    python scripts/wham_pkl_to_smplx.py \
        --pkl output/linglong2_slam/dateset_vedio/wham_output.pkl \
        --out  output/linglong2_slam/dateset_vedio/wham_smplx.npz
"""
import argparse

import joblib
import numpy as np


def convert(pkl_path, out_path, gender="neutral", fps=30, use_world=True):
    data = joblib.load(pkl_path)

    # wham_output.pkl is {subject_id: {...}}; pick the subject with the most frames.
    subject = max(data.keys(), key=lambda k: len(data[k]["frame_ids"]))
    d = data[subject]

    pose_key = "pose_world" if use_world else "pose"
    trans_key = "trans_world" if use_world else "trans"

    pose = np.asarray(d[pose_key], dtype=np.float32)       # (T, 72)
    trans = np.asarray(d[trans_key], dtype=np.float32)     # (T, 3)
    betas = np.asarray(d["betas"], dtype=np.float32)       # (T, 10)

    # Single body shape: average over frames (WHAM betas is per-frame).
    betas = betas.mean(axis=0)                             # (10,)

    smplx = {
        "root_orient": pose[:, :3],        # global orientation (N, 3)
        "pose_body": pose[:, 3:66],        # 21 SMPL-X body joints (N, 63)
        "trans": trans,                    # translation (N, 3)
        "betas": betas,                    # (10,)
        "gender": np.array(gender),
        "mocap_frame_rate": np.array(fps),
        # Metadata so load_smplx_file's coord_fix='auto' knows this is WHAM y-up.
        "coord_system": np.array("y-up"),
        "producer": np.array("wham"),
    }

    np.savez(out_path, **smplx)
    print(f"Wrote {out_path}: root_orient{smplx['root_orient'].shape} "
          f"pose_body{smplx['pose_body'].shape} trans{smplx['trans'].shape} "
          f"betas{smplx['betas'].shape} ({'world' if use_world else 'local'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path to wham_output.pkl")
    parser.add_argument("--out", required=True, help="Output SMPL-X .npz path")
    parser.add_argument("--gender", default="neutral", choices=["male", "female", "neutral"])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--local", action="store_true",
                        help="Use camera-local pose/trans instead of global (world) ones")
    args = parser.parse_args()

    convert(args.pkl, args.out, args.gender, args.fps, use_world=not args.local)
