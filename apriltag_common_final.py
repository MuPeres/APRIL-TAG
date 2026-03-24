from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
from pupil_apriltags import Detector


def build_detector(tag_family: str = "tag36h11") -> Detector:
    return Detector(
        families=tag_family,
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def order_corners_tltrbrbl(pts4x2: np.ndarray) -> np.ndarray:
    pts = pts4x2.astype(np.float64)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.vstack([tl, tr, br, bl])


def wrap_pm90(a: float) -> float:
    a = float(a)
    while a > 90.0:
        a -= 180.0
    while a < -90.0:
        a += 180.0
    return a


def tag_width_px(c4: np.ndarray) -> float:
    a = np.linalg.norm(c4[1] - c4[0])
    b = np.linalg.norm(c4[2] - c4[3])
    return float(0.5 * (a + b))


def refine_corners_subpix(gray: np.ndarray, corners4x2: np.ndarray, win: int = 7) -> np.ndarray:
    h, w = gray.shape[:2]
    c = corners4x2.astype(np.float64)
    min_border = float(
        np.min(
            np.stack(
                [c[:, 0], c[:, 1], (w - 1) - c[:, 0], (h - 1) - c[:, 1]],
                axis=1,
            )
        )
    )
    max_win = int(max(1, min(win, min_border - 1)))
    if max_win < 2:
        return corners4x2.astype(np.float64)

    pts = c.reshape(-1, 1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    try:
        cv2.cornerSubPix(gray, pts, (max_win, max_win), (-1, -1), criteria)
        return pts.reshape(4, 2).astype(np.float64)
    except cv2.error:
        return corners4x2.astype(np.float64)


def reproj_err_mean_px_safe(
    objp: np.ndarray,
    imgp: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(imgp.reshape(-1, 2) - proj, axis=1)))


def rot_diff_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    Rrel = Ra.T @ Rb
    tr = np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def compute_plane_angles(R: np.ndarray, imgp: np.ndarray) -> tuple[float, float, float, float]:
    zc = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = R[:, 2].reshape(3)
    if float(np.dot(n, zc)) < 0.0:
        n = -n

    theta = math.degrees(
        math.atan2(np.linalg.norm(np.cross(n, zc)), float(np.dot(n, zc)))
    )
    yaw_plane = wrap_pm90(math.degrees(math.atan2(float(n[0]), float(n[2]))))
    pitch_plane = wrap_pm90(math.degrees(math.atan2(float(-n[1]), float(n[2]))))

    v = (imgp[1] - imgp[0]).reshape(2)
    roll_img = math.degrees(math.atan2(float(v[1]), float(v[0])))
    return theta, yaw_plane, pitch_plane, roll_img


def pose_ippe_anchor(
    imgp: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    objp: np.ndarray,
    w_rot_deg: float = 2.0,
    w_tz_m: float = 5.0,
    R_ref: Optional[np.ndarray] = None,
    tz_ref: Optional[float] = None,
) -> Optional[dict]:
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        objectPoints=objp,
        imagePoints=imgp,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not ok or rvecs is None or len(rvecs) == 0:
        return None

    best = None
    best_fallback = None

    for i in range(len(rvecs)):
        rvec = rvecs[i]
        tvec = tvecs[i]
        tz = float(tvec.reshape(3)[2])

        if tz <= 0.0:
            continue

        R, _ = cv2.Rodrigues(rvec)
        n = R[:, 2].reshape(3)

        e = reproj_err_mean_px_safe(objp, imgp, rvec, tvec, K, dist)
        cost = e

        if R_ref is not None:
            cost += w_rot_deg * rot_diff_deg(R_ref, R)
        if tz_ref is not None:
            cost += w_tz_m * abs(tz - tz_ref)

        candidate = {
            "rvec": rvec,
            "tvec": tvec,
            "R": R,
            "reproj": e,
            "cost": cost,
        }

        # Guarda sempre um fallback válido
        if best_fallback is None or candidate["cost"] < best_fallback["cost"]:
            best_fallback = candidate

        # Prefere a solução com normal apontando para frente
        if float(n[2]) > 0.01:
            candidate["cost"] -= 0.25 * float(n[2])
            if best is None or candidate["cost"] < best["cost"]:
                best = candidate

    return best if best is not None else best_fallback


def make_tag_object_points(tag_size_m: float, dtype=np.float64) -> np.ndarray:
    s = float(tag_size_m)
    # Ordem exigida pelo IPPE_SQUARE:
    # 0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left
    # com Y do objeto apontando para cima
    return np.array(
        [
            [-s / 2, +s / 2, 0.0],  # TL
            [+s / 2, +s / 2, 0.0],  # TR
            [+s / 2, -s / 2, 0.0],  # BR
            [-s / 2, -s / 2, 0.0],  # BL
        ],
        dtype=dtype,
    )


def adjust_intrinsics(
    K_cal: np.ndarray,
    w_cal: int,
    h_cal: int,
    w_img: int,
    h_img: int,
    image_mode: str,
    crop_x0: int = 0,
    crop_y0: int = 0,
) -> np.ndarray:
    K = K_cal.copy().astype(np.float64)
    if image_mode == "full":
        return K

    if image_mode == "resize":
        sx = w_img / float(w_cal)
        sy = h_img / float(h_cal)
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        return K

    if image_mode == "crop":
        K[0, 2] -= float(crop_x0)
        K[1, 2] -= float(crop_y0)
        return K

    raise ValueError("image_mode inválido. Use 'full', 'resize' ou 'crop'.")


def grid_cell(cx: float, cy: float, w: int, h: int, grid_n: int = 3) -> tuple[int, int]:
    gx = min(grid_n - 1, max(0, int((cx / w) * grid_n)))
    gy = min(grid_n - 1, max(0, int((cy / h) * grid_n)))
    return gx, gy


def trapezoid_ratios(c4: np.ndarray) -> tuple[float, float]:
    top = np.linalg.norm(c4[1] - c4[0])
    bottom = np.linalg.norm(c4[2] - c4[3])
    left = np.linalg.norm(c4[3] - c4[0])
    right = np.linalg.norm(c4[2] - c4[1])
    return float(top / max(bottom, 1e-9)), float(left / max(right, 1e-9))


def roll_img_deg(c4: np.ndarray) -> float:
    v = (c4[1] - c4[0]).reshape(2)
    return float(np.degrees(np.arctan2(v[1], v[0])))


def draw_text_block(frame: np.ndarray, lines: list[str], x: int = 12, y0: int = 28) -> None:
    y = y0
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26