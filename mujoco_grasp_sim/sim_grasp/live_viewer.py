"""LiveViewer — a cv2-based window for interactive pick: shows the live
camera feed, captures mouse clicks, overlays SAM 3 mask results, and waits
for a keypress to confirm or retry a selection.

BGR vs RGB: OpenCV's imshow expects BGR pixel order; this project's cameras
(sim_grasp/camera.py) produce RGB. All display methods here convert
RGB -> BGR internally, so callers always pass RGB, matching every other
module in sim_grasp — no OpenCV-specific color convention leaks out.
"""
import threading
import time

import cv2
import numpy as np


def draw_status_text(rgb: np.ndarray, text: str) -> np.ndarray:
    """Pure compositing: `rgb` (H,W,3 uint8) with `text` drawn over a
    full-width translucent dark bar across the top -- deliberately bold
    and high-contrast so it reads as "clearly alive" at a glance, not just
    a small corner counter easy to miss. No window/display side effects --
    safe to unit-test directly."""
    out = np.ascontiguousarray(rgb).copy()
    h, w = out.shape[:2]
    bar_h = min(48, h)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), thickness=-1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
    cv2.putText(out, text, (14, bar_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
               (255, 255, 255), 2, cv2.LINE_AA)
    return out


def compose_mask_overlay(rgb: np.ndarray, mask: np.ndarray,
                         color: tuple[int, int, int] = (255, 80, 40),
                         alpha: float = 0.45) -> np.ndarray:
    """Pure compositing: `rgb` (H,W,3 uint8) with `mask` (H,W bool) blended
    toward `color` (RGB tuple) by `alpha` (0 = unchanged, 1 = fully
    replaced). No window/display side effects — safe to unit-test directly
    with synthetic arrays."""
    out = np.ascontiguousarray(rgb).astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


class LiveViewer:
    def __init__(self, window_name: str = 'ContactPilot -- interactive pick'):
        self.window_name = window_name
        self._click_xy: tuple[int, int] | None = None
        self._closed = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_xy = (x, y)

    def _pump(self, delay_ms: int = 1) -> int:
        """Process window events once; returns the pressed key code (-1 if
        none). Also detects the window being closed via its X button."""
        key = cv2.waitKey(delay_ms) & 0xFF
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0
        if visible < 1:
            self._closed = True
        return key

    @property
    def closed(self) -> bool:
        return self._closed

    def show_frame(self, rgb: np.ndarray) -> None:
        """Display an RGB frame; pumps the event loop once (non-blocking)."""
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, bgr)
        self._pump()

    def wait_for_click(self, rgb: np.ndarray) -> tuple[int, int] | None:
        """Display rgb and block (repeatedly re-showing it, pumping the
        window) until the user clicks a pixel or closes the window. Returns
        (x, y) pixel coordinates, or None if the window was closed."""
        self._click_xy = None
        while not self._closed:
            self.show_frame(rgb)
            if self._click_xy is not None:
                xy, self._click_xy = self._click_xy, None
                return xy
        return None

    def show_mask_overlay(self, rgb: np.ndarray, mask: np.ndarray,
                          color: tuple[int, int, int] = (255, 80, 40),
                          alpha: float = 0.45) -> None:
        """Display rgb with `mask` (H,W bool) highlighted in a translucent
        color (given as an RGB tuple)."""
        self.show_frame(compose_mask_overlay(rgb, mask, color=color, alpha=alpha))

    def wait_for_confirm(self) -> bool:
        """Block until Enter/Space confirms or Esc/'c' cancels-and-retries,
        or the window closes (treated as cancel). Returns True if
        confirmed."""
        while not self._closed:
            key = self._pump(delay_ms=30)
            if key in (13, 32):        # Enter, Space
                return True
            if key in (27, ord('c')):  # Esc, 'c'
                return False
        return False

    def run_blocking(self, rgb: np.ndarray, fn, message: str = 'Working...'):
        """Run `fn()` (no args) on a background thread while this thread
        keeps pumping the window, overlaying `message` + an elapsed-time
        counter on `rgb` -- otherwise the window sits un-pumped for the
        whole call (SAM 3 / GraspGen / CGN calls routinely take 5-30s) and
        the window manager marks it "not responding", indistinguishable
        from a crash. Returns fn()'s result; re-raises any exception fn()
        raised. Does not support cancellation mid-call -- these are opaque
        subprocess/model calls with no cancellation hook, so fn() always
        runs to completion even if the user closes the window meanwhile."""
        box = {}

        def _target():
            try:
                box['result'] = fn()
            except Exception as e:
                box['error'] = e

        spinner_frames = '|/-\\'
        t = threading.Thread(target=_target, daemon=True)
        t0 = time.time()
        t.start()
        while t.is_alive():
            elapsed = time.time() - t0
            spin = spinner_frames[int(elapsed * 6) % len(spinner_frames)]
            frame = draw_status_text(rgb, f'{spin} {message} {elapsed:.0f}s')
            self.show_frame(frame)
        t.join()
        if 'error' in box:
            raise box['error']
        return box['result']

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
