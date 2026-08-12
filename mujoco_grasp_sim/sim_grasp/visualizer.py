"""Visualization — reuses Contact-GraspNet's own Open3D grasp drawing so the
gripper rendering convention is guaranteed to match the network's output.

Also saves 2D observation images (RGB / depth / segmap) and a CGN-compatible
predictions .npz that contact_graspnet_pytorch/visualize_saved_scene.py can
re-open later.
"""

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # -- 2D observation dump (non-blocking, just files) -----------------------
    def save_observation(self, rgb, depth, segmap=None):
        matplotlib.use('Agg', force=False)
        import imageio.v2 as imageio
        imageio.imwrite(self.save_dir / 'rgb.png', rgb)
        np.save(self.save_dir / 'depth.npy', depth)

        d = depth.copy()
        d[d <= 0] = np.nan
        fig, axes = plt.subplots(1, 3 if segmap is not None else 2, figsize=(15, 5))
        axes[0].imshow(rgb); axes[0].set_title('RGB')
        im = axes[1].imshow(d, cmap='viridis'); axes[1].set_title('Depth [m]')
        fig.colorbar(im, ax=axes[1], fraction=0.046)
        if segmap is not None:
            axes[2].imshow(segmap, cmap='tab20'); axes[2].set_title('Segmentation')
        for ax in axes:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(self.save_dir / 'observation.png', dpi=110)
        plt.close(fig)

    # -- CGN-compatible artifact ----------------------------------------------
    def save_predictions_npz(self, pc_full, grasps_cam, scores, contact_pts, pc_colors):
        np.savez(self.save_dir / 'predictions_sim.npz',
                 pc_full=pc_full, pred_grasps_cam=grasps_cam, scores=scores,
                 contact_pts=contact_pts, pc_colors=pc_colors)

    # -- interactive 3D (blocking Open3D window) -------------------------------
    def show_grasps(self, pc_full, grasps_cam, scores, pc_colors=None):
        """Point cloud + predicted grasps in the camera frame.
        Reuses contact_graspnet_pytorch's visualize_grasps (green=best)."""
        from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps
        visualize_grasps(pc_full, grasps_cam, scores,
                         plot_opencv_cam=True, pc_colors=pc_colors)
