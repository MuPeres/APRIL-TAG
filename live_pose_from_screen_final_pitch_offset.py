from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import mss
import numpy as np

from apriltag_common_final import (
    adjust_intrinsics,
    build_detector,
    compute_plane_angles,
    draw_text_block,
    make_tag_object_points,
    order_corners_tltrbrbl,
    pose_ippe_anchor,
    refine_corners_subpix,
    tag_width_px,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pose ao vivo via captura do preview da câmera na tela.")
    p.add_argument("--calib-npz", required=True, help="Arquivo .npz gerado pela calibração.")
    p.add_argument("--tag-id", type=int, default=35)
    p.add_argument("--tag-family", default="tag36h11")
    p.add_argument("--tag-size-m", type=float, default=0.028)
    p.add_argument("--image-mode", choices=["full", "resize", "crop"], default="resize")
    p.add_argument("--crop-x0", type=int, default=0)
    p.add_argument("--crop-y0", type=int, default=0)
    p.add_argument("--subpix-win", type=int, default=7)
    p.add_argument("--min-tag-width-px", type=float, default=20.0)
    p.add_argument("--select-roi", action="store_true")
    p.add_argument("--roi-file", default="preview_roi.json")
    p.add_argument("--monitor-index", type=int, default=1)
    p.add_argument("--save-csv", default="")
    return p.parse_args()


def load_calibration(calib_path: Path):
    cal = np.load(str(calib_path))
    K_cal = cal["K"].astype(np.float64)
    dist = cal["dist"].astype(np.float64).reshape(-1, 1)
    w_cal, h_cal = cal["image_size"].astype(int).ravel()
    return K_cal, dist, int(w_cal), int(h_cal)


def select_roi_and_save(sct: mss.mss, monitor_index: int, roi_file: Path) -> dict:
    mon = sct.monitors[monitor_index]
    raw = np.array(sct.grab(mon))
    img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    roi = cv2.selectROI("Selecione o preview da câmera e pressione ENTER", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Selecione o preview da câmera e pressione ENTER")
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI inválida. Rode novamente com --select-roi.")
    region = {"left": mon["left"] + x, "top": mon["top"] + y, "width": w, "height": h}
    roi_file.write_text(json.dumps(region, indent=2), encoding="utf-8")
    print(f"ROI salva em: {roi_file}")
    print(region)
    return region


def load_roi(roi_file: Path) -> dict:
    if not roi_file.exists():
        raise FileNotFoundError(f"Arquivo ROI não encontrado: {roi_file}. Rode primeiro com --select-roi.")
    return json.loads(roi_file.read_text(encoding="utf-8"))


YAW_TOL_DEG = 2.0
PITCH_TOL_DEG = 2.0
ROLL_TOL_DEG = 2.0
THETA_TOL_DEG = 3.0
INVERT_YAW_CMD = False
INVERT_PITCH_CMD = False
INVERT_ROLL_CMD = False

# Offsets manuais de zeragem (solucao rapida)
USE_MANUAL_OFFSETS = True
THETA_OFFSET_DEG = 0.0
YAW_OFFSET_DEG = 0.0
PITCH_OFFSET_DEG = 0
ROLL_OFFSET_DEG = 0.0


def _cmd_from_value(v: float, tol: float, cmd_pos: str, cmd_neg: str, invert: bool = False):
    vv = -v if invert else v
    if vv > tol:
        return cmd_pos
    if vv < -tol:
        return cmd_neg
    return None


def build_guidance(theta: float, yaw: float, pitch: float, roll: float):
    cmds = []
    yaw_cmd = _cmd_from_value(yaw, YAW_TOL_DEG, "ajuste horizontal: gire para ESQUERDA", "ajuste horizontal: gire para DIREITA", INVERT_YAW_CMD)
    if yaw_cmd:
        cmds.append(yaw_cmd)
    pitch_cmd = _cmd_from_value(pitch, PITCH_TOL_DEG, "ajuste vertical: gire para BAIXO", "ajuste vertical: gire para CIMA", INVERT_PITCH_CMD)
    if pitch_cmd:
        cmds.append(pitch_cmd)
    if abs(yaw) <= YAW_TOL_DEG and abs(pitch) <= PITCH_TOL_DEG:
        roll_cmd = _cmd_from_value(roll, ROLL_TOL_DEG, "ajuste roll: gire no sentido HORÁRIO", "ajuste roll: gire no sentido ANTI-HORÁRIO", INVERT_ROLL_CMD)
        if roll_cmd:
            cmds.append(roll_cmd)
    aligned = abs(theta) <= THETA_TOL_DEG and abs(yaw) <= YAW_TOL_DEG and abs(pitch) <= PITCH_TOL_DEG and abs(roll) <= ROLL_TOL_DEG
    if aligned:
        return "ALINHADO", True
    if not cmds:
        return "quase alinhado", False
    return " | ".join(cmds), False


def main() -> None:
    args = parse_args()
    calib_path = Path(args.calib_npz)
    roi_file = Path(args.roi_file)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibração não encontrada: {calib_path}")

    K_cal, dist, w_cal, h_cal = load_calibration(calib_path)
    detector = build_detector(args.tag_family)
    objp = make_tag_object_points(args.tag_size_m)
    prev_R = None
    prev_tz = None
    last_t = time.perf_counter()
    fps_ema = None
    last_pose = None

    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp_s", "theta_deg", "yaw_deg", "pitch_deg", "roll_deg", "tz_m", "reproj_px", "tag_width_px", "aligned", "guidance"])
        print(f"CSV será salvo em: {csv_path}")

    with mss.mss() as sct:
        if args.select_roi:
            select_roi_and_save(sct, args.monitor_index, roi_file)
            print("Seleção concluída. Rode novamente sem --select-roi.")
            return
        region = load_roi(roi_file)
        print(f"Usando ROI: {region}")

        while True:
            raw = np.array(sct.grab(region))
            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            h_img, w_img = frame.shape[:2]
            K = adjust_intrinsics(K_cal, w_cal, h_cal, w_img, h_img, args.image_mode, args.crop_x0, args.crop_y0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
            det = next((r for r in results if int(r.tag_id) == args.tag_id), None)
            lines = []

            if det is None:
                lines.append("Tag alvo nao detectada")
                prev_R = None
                prev_tz = None
                last_pose = None
            else:
                corners = order_corners_tltrbrbl(np.array(det.corners, dtype=np.float64))
                corners = refine_corners_subpix(gray, corners, win=args.subpix_win)
                width_px = tag_width_px(corners)
                pts_int = corners.astype(int)
                cv2.polylines(frame, [pts_int], True, (0, 255, 0), 2)
                for i, p in enumerate(pts_int):
                    cv2.circle(frame, tuple(p), 4, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), tuple((p + np.array([6, -6])).tolist()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if width_px < args.min_tag_width_px:
                    lines.append(f"Tag detectada, mas pequena: {width_px:.1f}px")
                    prev_R = None
                    prev_tz = None
                    last_pose = None
                else:
                    sol = pose_ippe_anchor(corners, K, dist, objp, R_ref=prev_R, tz_ref=prev_tz)
                    if sol is None:
                        lines.append("Falha no solvePnP/IPPE")
                        prev_R = None
                        prev_tz = None
                        last_pose = None
                    else:
                        rvec, tvec, R, reproj = sol["rvec"], sol["tvec"], sol["R"], sol["reproj"]
                        tz = float(tvec.reshape(3)[2])
                        theta_raw, yaw_raw, pitch_raw, roll_raw = compute_plane_angles(R, corners)

                        if USE_MANUAL_OFFSETS:
                            theta = theta_raw - THETA_OFFSET_DEG
                            yaw   = yaw_raw   - YAW_OFFSET_DEG
                            pitch = pitch_raw - PITCH_OFFSET_DEG
                            roll  = roll_raw  - ROLL_OFFSET_DEG
                        else:
                            theta, yaw, pitch, roll = theta_raw, yaw_raw, pitch_raw, roll_raw

                        guidance, aligned = build_guidance(theta, yaw, pitch, roll)
                        prev_R = R.copy()
                        prev_tz = tz
                        last_pose = {"theta": theta, "yaw": yaw, "pitch": pitch, "roll": roll, "tz": tz, "reproj": reproj, "tag_width_px": width_px, "guidance": guidance, "aligned": aligned}
                        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, args.tag_size_m * 0.5)
                        lines.extend([
                            f"ID: {args.tag_id}",
                            f"theta: {theta:+.2f} deg",
                            f"yaw:   {yaw:+.2f} deg",
                            f"pitch: {pitch:+.2f} deg",
                            f"roll:  {roll:+.2f} deg",
                            f"tz:    {tz:+.4f} m",
                            f"reproj:{reproj:.2f} px",
                            f"tag_w: {width_px:.1f} px",
                            f"status: {'ALINHADO' if aligned else 'AJUSTAR'}",
                            f"ajuste: {guidance}",
                        ])
                        if csv_writer is not None:
                            csv_writer.writerow([time.time(), theta, yaw, pitch, roll, tz, reproj, width_px, int(aligned), guidance])

            now = time.perf_counter()
            fps = 1.0 / max(now - last_t, 1e-9)
            last_t = now
            fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps
            lines.insert(0, f"fps: {fps_ema:.1f}")
            draw_text_block(frame, lines)
            cv2.imshow("AprilTag live pose", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('r'):
                prev_R = None
                prev_tz = None
                print("Ancora resetada.")
            if key == ord('s'):
                out = Path("snapshot_live_pose.png")
                cv2.imwrite(str(out), frame)
                print(f"Snapshot salvo em {out.resolve()}")
            if key == ord('p') and last_pose is not None:
                print(last_pose)

    if csv_file is not None:
        csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
