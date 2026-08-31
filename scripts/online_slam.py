"""Online (real-time) DPVO SLAM wrapper for the WHAM streaming pipeline.

Offline ``demo.py`` runs DPVO over a whole video file via ``video_stream`` in a
subprocess, then feeds the resulting camera trajectory into WHAM's trajectory
decoder as camera angular velocity (``cam_angvel``).  This module exposes the
same trajectory *online*: it consumes frames in order, updates DPVO
incrementally, and stores a camera-to-world pose per frame so the streaming
WHAM window can build its ``cam_angvel`` input in exactly the same convention
(``DPVO.terminate()`` output, i.e. ``[tx, ty, tz, qx, qy, qz, qw]``).

If DPVO / lietorch cannot be imported, every method degrades to a no-op and the
caller falls back to zero camera angular velocity (WHAM local motion).
"""
import os
import threading

import numpy as np

try:
    import cv2
    import torch
    from dpvo.dpvo import DPVO
    from dpvo.config import cfg as _dpvo_cfg
    from dpvo.lietorch import SE3
    _DPVO_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    _DPVO_AVAILABLE = False
    _DPVO_IMPORT_ERROR = _e

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPVO_CFG_PATH = os.path.join(ROOT_DIR, "third-party", "DPVO", "config", "default.yaml")
DPVO_CKPT_PATH = os.path.join(ROOT_DIR, "checkpoints", "dpvo.pth")


def _convert_dpvo_to_cam_angvel(traj, fps):
    """Camera angular velocity from a camera-to-world trajectory.

    Faithful copy of ``lib.data.datasets.dataset_custom.convert_dpvo_to_cam_angvel``
    (inlined to avoid importing the whole dataset module). ``traj`` is (T, 7) with
    columns ``[tx, ty, tz, qx, qy, qz, qw]``. Returns a (T, 6) torch CPU tensor.
    """
    from lib.utils import transforms

    quat = traj[:, 3:]
    # Convert (x,y,z,w) to (w,x,y,z)
    quat = quat[:, [3, 0, 1, 2]]

    # Quat is camera-to-world. Convert to world-to-camera rotation and take .mT.
    world2cam = transforms.quaternion_to_matrix(torch.from_numpy(quat)).float()
    R = world2cam.mT

    cam_angvel = transforms.matrix_to_axis_angle(R[:-1] @ R[1:].transpose(-1, -2))
    cam_angvel = transforms.matrix_to_rotation_6d(transforms.axis_angle_to_matrix(cam_angvel))
    cam_angvel = cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]]).to(cam_angvel)
    cam_angvel = cam_angvel * fps
    cam_angvel = torch.cat((cam_angvel, cam_angvel[:1]), dim=0)
    return cam_angvel


class OnlineSLAM:
    """Incremental DPVO SLAM with per-frame camera-to-world pose lookup."""

    def __init__(self, width, height, scale=0.5, buffer_size=1024, viz=False):
        self.enabled = _DPVO_AVAILABLE and os.path.exists(DPVO_CKPT_PATH)
        self.import_error = None if _DPVO_AVAILABLE else _DPVO_IMPORT_ERROR
        if not self.enabled:
            return

        self.scale = float(scale)
        self.buffer_size = int(buffer_size)
        self.viz = bool(viz)
        self.slam = None

        # Pinhole intrinsics for the *downscaled* image, matching SLAMModel /
        # compute_cam_intrinsics: focal = diagonal, principal point at center.
        focal = (height ** 2 + width ** 2) ** 0.5
        self._K_half = np.array(
            [focal * self.scale, focal * self.scale,
             width * self.scale / 2.0, height * self.scale / 2.0],
            dtype=np.float32,
        )

        self._lock = threading.Lock()
        self._poses = {}

    def _prep(self, frame):
        img = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        img = img[:h - h % 16, :w - w % 16]
        return img

    def _init_slam(self, frame):
        img = self._prep(frame)
        h, w = img.shape[:2]
        try:
            _dpvo_cfg.merge_from_file(DPVO_CFG_PATH)
            _dpvo_cfg.BUFFER_SIZE = self.buffer_size
            self.slam = DPVO(_dpvo_cfg, DPVO_CKPT_PATH, ht=h, wd=w, viz=self.viz)
        except Exception as e:
            self.slam = None
            self.enabled = False
            self.import_error = e
            raise RuntimeError(f"DPVO init failed: {e}")

    def process(self, frame, frame_id):
        """Feed one BGR frame (in order) to DPVO and record its camera pose."""
        if not self.enabled:
            return
        try:
            if self.slam is None:
                self._init_slam(frame)
            img = self._prep(frame)
            image = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).cuda()
            intrinsics = torch.from_numpy(self._K_half).cuda()
            self.slam(float(frame_id), image, intrinsics)
            self._record_pose(frame_id)
        except Exception as e:
            self.enabled = False
            self.import_error = e
            print(f"[SLAM] online SLAM disabled after error: {e}")

    def _record_pose(self, frame_id):
        n = self.slam.n
        if n <= 0:
            return
        # poses_[i] is world-to-camera (tvec + xyzw quat); invert -> camera-to-world,
        # matching DPVO.terminate() output consumed by _convert_dpvo_to_cam_angvel.
        idx = n - 1
        T_cw = SE3(self.slam.poses_[idx].clone()).inv()
        pose = T_cw.data.detach().cpu().numpy().astype(np.float64)
        with self._lock:
            self._poses[int(frame_id)] = pose

    def get_pose(self, frame_id):
        with self._lock:
            return self._poses.get(int(frame_id))

    def cam_angvel_for(self, frame_ids, fps, device):
        """Build (1, f, 6) camera angular velocity for a contiguous frame window.

        Returns a zero tensor (local-motion fallback) if SLAM is disabled or any
        frame pose is still missing (e.g. SLAM thread has not caught up yet).
        """
        f = len(frame_ids)
        if not self.enabled or f < 2:
            return torch.zeros((1, f, 6), device=device)

        traj = []
        for fid in frame_ids:
            p = self.get_pose(fid)
            if p is None:
                return torch.zeros((1, f, 6), device=device)
            traj.append(p)

        angvel = _convert_dpvo_to_cam_angvel(np.stack(traj, axis=0), float(fps))
        return angvel.unsqueeze(0).to(device=device, dtype=torch.float32)

    def close(self):
        if self.slam is not None:
            try:
                self.slam.terminate()
            except Exception:
                pass
            self.slam = None
