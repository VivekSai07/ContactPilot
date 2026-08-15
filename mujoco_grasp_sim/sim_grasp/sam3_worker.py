"""Subprocess SAM 3 worker — runs under the sam3_torch interpreter.

Usage:
    python sim_grasp/sam3_worker.py rgb.npy out.npz --prompt "the red box"
    python sim_grasp/sam3_worker.py rgb.npy out.npz --click 320,240
    python sim_grasp/sam3_worker.py rgb.npy out.npz --box 100,100,300,300

rgb.npy: (H,W,3) uint8 RGB image.
out.npz keys: masks (K,H,W) bool, scores (K,) float32, boxes (K,4) float32
    (pixel [x1,y1,x2,y2], computed from each mask's own bounding box).
"""
import argparse

import numpy as np
from PIL import Image


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Pixel-space [x1,y1,x2,y2] bounding box of the True region of a 2D
    boolean mask. Computed directly from the mask rather than trusted from
    the model's raw box output, whose exact coordinate convention wasn't
    independently verified against the installed version."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rgb_npy')
    ap.add_argument('out_npz')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt', type=str)
    group.add_argument('--click', type=str, help='X,Y pixel coordinates')
    group.add_argument('--box', type=str, help='X1,Y1,X2,Y2 pixel coordinates')
    ap.add_argument('--click-radius-px', type=int, default=15,
                    help='half-width of the box synthesized around a --click point')
    args = ap.parse_args()

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    rgb = np.load(args.rgb_npy)
    H, W = rgb.shape[:2]
    image = Image.fromarray(rgb)

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    state = processor.set_image(image)

    if args.prompt:
        output = processor.set_text_prompt(prompt=args.prompt, state=state)
    else:
        if args.click:
            x, y = (float(v) for v in args.click.split(','))
            r = args.click_radius_px
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
        else:
            x1, y1, x2, y2 = (float(v) for v in args.box.split(','))
        cx, cy = (x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        output = processor.add_geometric_prompt(
            box=[cx, cy, w, h], label=True, state=state)

    masks_t, scores_t = output['masks'], output['scores']
    masks = masks_t.cpu().numpy().astype(bool) if hasattr(masks_t, 'cpu') \
        else np.asarray(masks_t, dtype=bool)
    scores = scores_t.cpu().numpy().astype(np.float32) if hasattr(scores_t, 'cpu') \
        else np.asarray(scores_t, dtype=np.float32)
    if masks.ndim == 2:          # single instance: normalize to (1,H,W)
        masks = masks[None]
        scores = scores.reshape(1)

    boxes = np.array([mask_bbox(m) for m in masks], dtype=np.float32)

    np.savez(args.out_npz, masks=masks, scores=scores, boxes=boxes)
    print(f'[sam3-worker] {len(masks)} match(es), scores={scores.tolist()} -> {args.out_npz}')


if __name__ == '__main__':
    main()
