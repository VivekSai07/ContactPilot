"""Load the real-world eye-to-hand calibration result.

The lab calibration file (calibration_result.yaml) stores:
    T_cam_to_base : 4x4 row-major matrix
We interpret it by its name: the transform that maps points FROM the camera
frame TO the robot base frame, i.e. T_base_cam, with the camera in the usual
OpenCV convention (+Z = viewing direction). This matches the output of
standard eye-to-hand calibration tools (OpenCV calibrateHandEye / easy_handeye,
method HORAUD).

NOTE: the file's `translation:` block contains numpy python-object YAML tags
that yaml.safe_load rejects (and which should not be deserialized as code).
We therefore extract ONLY the plain-float matrix block before parsing.
"""

from pathlib import Path

import numpy as np
import yaml


def load_T_base_cam(path: str | Path) -> np.ndarray:
    """Parse only the `T_cam_to_base` matrix block from the calibration YAML.

    :returns: (4,4) T_base_cam (camera pose in robot base frame, OpenCV camera)
    """
    text = Path(path).read_text(encoding='utf-8')

    # Keep only the matrix block so yaml.safe_load never sees the
    # numpy python-object tags further down the file.
    lines, keep = text.splitlines(), []
    in_matrix = False
    for ln in lines:
        if ln.startswith('T_cam_to_base:'):
            in_matrix = True
            keep.append(ln)
            continue
        if in_matrix:
            if ln.startswith((' ', '-', '\t')):
                keep.append(ln)
            else:
                break
    if not keep:
        raise ValueError(f'No T_cam_to_base key found in {path}')

    data = yaml.safe_load('\n'.join(keep))
    T = np.asarray(data['T_cam_to_base'], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f'T_cam_to_base has shape {T.shape}, expected (4,4)')

    # sanity: valid rotation
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-3):
        raise ValueError('T_cam_to_base rotation block is not a valid rotation matrix')
    return T
