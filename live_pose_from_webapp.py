from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
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
    rot_diff_deg,
    tag_width_px,
)
from webapp_frame_source import WebAppFrameSource


YAW_TOL_DEG = 2.0
PITCH_TOL_DEG = 2.0
ROLL_TOL_DEG = 2.0
THETA_TOL_DEG = 3.0

INVERT_YAW_CMD = False
INVERT_PITCH_CMD = False
INVERT_ROLL_CMD = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pose ao vivo lendo o preview nativo do Web-App da câmera."
    )
    p.add_argument("--url", required=True, help="URL do Web-App da câmera, ex.: http://169.254.125.172/")
    p.add_argument("--calib-npz", required=True, help="Arquivo .npz gerado pela calibração.")
    p.add_argument("--tag-id", type=int, default=35)
    p.add_argument("--tag-family", default="tag36h11")
    p.add_argument("--tag-size-m", type=float, default=0.028)
    p.add_argument("--image-mode", choices=["full", "resize", "crop"], default="full")
    p.add_argument("--crop-x0", type=int, default=0)
    p.add_argument("--crop-y0", type=int, default=0)
    p.add_argument("--subpix-win", type=int, default=7)
    p.add_argument("--min-tag-width-px", type=float, default=50.0)
    p.add_argument("--max-jump-deg", type=float, default=25.0)
    p.add_argument("--max-reproj-for-jump", type=float, default=1.0)
    p.add_argument("--display-scale", type=float, default=0.5)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--save-csv", default="")
    p.add_argument("--desired-width", type=int, default=0, help="0 = usar largura da calibração")
    p.add_argument("--desired-height", type=int, default=0, help="0 = usar altura da calibração")
    p.add_argument("--theta-offset-deg", type=float, default=0.0)
    p.add_argument("--yaw-offset-deg", type=float, default=0.0)
    p.add_argument("--pitch-offset-deg", type=float, default=0.0)
    p.add_argument("--roll-offset-deg", type=float, default=0.0)
    return p.parse_args()


def load_calibration(calib_path: Path):
    cal = np.load(str(calib_path))
    K_cal = cal["K"].astype(np.float64)
    dist = cal["dist"].astype(np.float64).reshape(-1, 1)
    w_cal, h_cal = cal["image_size"].astype(int).ravel()
    return K_cal, dist, int(w_cal), int(h_cal)


def _cmd_from_value(v: float, tol: float, cmd_pos: str, cmd_neg: str, invert: bool = False):
    vv = -v if invert else v
    if vv > tol:
        return cmd_pos
    if vv < -tol:
        return cmd_neg
    return None


def build_guidance(theta: float, yaw: float, pitch: float, roll: float):
    cmds = []

    yaw_cmd = _cmd_from_value(
        yaw,
        YAW_TOL_DEG,
        "ajuste horizontal: gire para ESQUERDA",
        "ajuste horizontal: gire para DIREITA",
        INVERT_YAW_CMD,
    )
    if yaw_cmd:
        cmds.append(yaw_cmd)

    pitch_cmd = _cmd_from_value(
        pitch,
        PITCH_TOL_DEG,
        "ajuste vertical: gire para BAIXO",
        "ajuste vertical: gire para CIMA",
        INVERT_PITCH_CMD,
    )
    if pitch_cmd:
        cmds.append(pitch_cmd)

    if abs(yaw) <= YAW_TOL_DEG and abs(pitch) <= PITCH_TOL_DEG:
        roll_cmd = _cmd_from_value(
            roll,
            ROLL_TOL_DEG,
            "ajuste roll: gire no sentido HORÁRIO",
            "ajuste roll: gire no sentido ANTI-HORÁRIO",
            INVERT_ROLL_CMD,
        )
        if roll_cmd:
            cmds.append(roll_cmd)

    aligned = (
        abs(theta) <= THETA_TOL_DEG
        and abs(yaw) <= YAW_TOL_DEG
        and abs(pitch) <= PITCH_TOL_DEG
        and abs(roll) <= ROLL_TOL_DEG
    )

    if aligned:
        return "ALINHADO", True

    if not cmds:
        return "quase alinhado", False

    return " | ".join(cmds), False


def main() -> None:
    args = parse_args()

    calib_path = Path(args.calib_npz)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibração não encontrada: {calib_path}")

    K_cal, dist, w_cal, h_cal = load_calibration(calib_path)
    desired_w = int(args.desired_width or w_cal)
    desired_h = int(args.desired_height or h_cal)

    source = WebAppFrameSource(
        url=args.url,
        desired_width=desired_w,
        desired_height=desired_h,
        headless=args.headless,
    )
    detector = build_detector(args.tag_family)
    objp = make_tag_object_points(args.tag_size_m)

    theta_offset = float(args.theta_offset_deg)
    yaw_offset = float(args.yaw_offset_deg)
    pitch_offset = float(args.pitch_offset_deg)
    roll_offset = float(args.roll_offset_deg)

    prev_R = None
    prev_tz = None
    last_pose = None
    last_raw = None
    last_t = time.perf_counter()
    fps_ema = None

    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "timestamp_s",
                "theta_deg",
                "yaw_deg",
                "pitch_deg",
                "roll_deg",
                "tz_m",
                "reproj_px",
                "tag_width_px",
                "jump_deg",
                "aligned",
                "guidance",
            ]
        )
        print(f"CSV será salvo em: {csv_path}")

    source.start()
    try:
        frame, info = source.wait_first_frame()
        print(f"Fonte conectada: {info}")
        print("Teclas: q/ESC=sair | r=reset âncora | s=salvar PNG | p=print pose | z=zerar offsets | x=limpar offsets")

        while True:
            frame, info = source.get_frame()
            if frame is None:
                continue

            h_img, w_img = frame.shape[:2]
            K = adjust_intrinsics(
                K_cal,
                w_cal,
                h_cal,
                w_img,
                h_img,
                args.image_mode,
                args.crop_x0,
                args.crop_y0,
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
            det = next((r for r in results if int(r.tag_id) == args.tag_id), None)

            lines = [
                f"src: {info.get('kind')} raw={info.get('width')}x{info.get('height')} view={info.get('clientWidth')}x{info.get('clientHeight')}"
            ]

            if det is None:
                lines.append("Tag alvo não detectada")
                prev_R = None
                prev_tz = None
                last_pose = None
                last_raw = None

            else:
                corners = order_corners_tltrbrbl(np.array(det.corners, dtype=np.float64))
                corners = refine_corners_subpix(gray, corners, win=args.subpix_win)
                width_px = tag_width_px(corners)

                pts_int = corners.astype(int)
                cv2.polylines(frame, [pts_int], True, (0, 255, 0), 2)
                for i, p in enumerate(pts_int):
                    cv2.circle(frame, tuple(p), 4, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        str(i),
                        tuple((p + np.array([6, -6])).tolist()),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                if width_px < args.min_tag_width_px:
                    lines.append(f"Tag detectada, mas pequena: {width_px:.1f}px")
                    prev_R = None
                    prev_tz = None
                    last_pose = None
                    last_raw = None

                else:
                    sol = pose_ippe_anchor(corners, K, dist, objp, R_ref=prev_R, tz_ref=prev_tz)
                    if sol is None:
                        lines.append("Falha no solvePnP/IPPE")
                        prev_R = None
                        prev_tz = None
                        last_pose = None
                        last_raw = None

                    else:
                        rvec = sol["rvec"]
                        tvec = sol["tvec"]
                        R = sol["R"]
                        reproj = float(sol["reproj"])
                        tz = float(tvec.reshape(3)[2])

                        jump_deg = rot_diff_deg(prev_R, R) if prev_R is not None else 0.0
                        if (
                            prev_R is not None
                            and jump_deg > args.max_jump_deg
                            and reproj > args.max_reproj_for_jump
                        ):
                            lines.append(f"Pose rejeitada: salto angular {jump_deg:.1f} deg")
                            lines.append(f"reproj: {reproj:.2f} px")
                            lines.append(f"tag_w: {width_px:.1f} px")
                        else:
                            theta_raw, yaw_raw, pitch_raw, roll_raw = compute_plane_angles(R, corners)

                            theta = theta_raw - theta_offset
                            yaw = yaw_raw - yaw_offset
                            pitch = pitch_raw - pitch_offset
                            roll = roll_raw - roll_offset

                            guidance, aligned = build_guidance(theta, yaw, pitch, roll)

                            prev_R = R.copy()
                            prev_tz = tz
                            last_raw = (theta_raw, yaw_raw, pitch_raw, roll_raw)
                            last_pose = {
                                "theta": theta,
                                "yaw": yaw,
                                "pitch": pitch,
                                "roll": roll,
                                "tz": tz,
                                "reproj": reproj,
                                "tag_width_px": width_px,
                                "jump_deg": jump_deg,
                                "guidance": guidance,
                                "aligned": aligned,
                            }

                            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, args.tag_size_m * 0.5)

                            lines.extend(
                                [
                                    f"ID: {args.tag_id}",
                                    f"theta: {theta:+.2f} deg",
                                    f"yaw:   {yaw:+.2f} deg",
                                    f"pitch: {pitch:+.2f} deg",
                                    f"roll:  {roll:+.2f} deg",
                                    f"tz:    {tz:+.4f} m",
                                    f"reproj:{reproj:.2f} px",
                                    f"tag_w: {width_px:.1f} px",
                                    f"jump:  {jump_deg:.1f} deg",
                                    f"status: {'ALINHADO' if aligned else 'AJUSTAR'}",
                                    f"ajuste: {guidance}",
                                ]
                            )

                            if csv_writer is not None:
                                csv_writer.writerow(
                                    [
                                        time.time(),
                                        theta,
                                        yaw,
                                        pitch,
                                        roll,
                                        tz,
                                        reproj,
                                        width_px,
                                        jump_deg,
                                        int(aligned),
                                        guidance,
                                    ]
                                )

            now = time.perf_counter()
            fps = 1.0 / max(now - last_t, 1e-9)
            last_t = now
            fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps
            lines.insert(0, f"fps: {fps_ema:.1f}")

            disp = frame
            if args.display_scale != 1.0:
                disp = cv2.resize(
                    frame,
                    (int(w_img * args.display_scale), int(h_img * args.display_scale)),
                    interpolation=cv2.INTER_AREA,
                )

            draw_text_block(disp, lines)
            cv2.imshow("AprilTag live pose (WebApp)", disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                prev_R = None
                prev_tz = None
                print("Âncora resetada.")
            if key == ord("s"):
                out = Path("snapshot_live_pose_webapp.png")
                cv2.imwrite(str(out), frame)
                print(f"Snapshot salvo em {out.resolve()}")
            if key == ord("p") and last_pose is not None:
                print(last_pose)
            if key == ord("z") and last_raw is not None:
                theta_offset, yaw_offset, pitch_offset, roll_offset = last_raw
                print(
                    f"Offsets zerados na pose atual: "
                    f"theta={theta_offset:+.2f}, yaw={yaw_offset:+.2f}, "
                    f"pitch={pitch_offset:+.2f}, roll={roll_offset:+.2f}"
                )
            if key == ord("x"):
                theta_offset = yaw_offset = pitch_offset = roll_offset = 0.0
                print("Offsets limpos.")

    finally:
        source.stop()
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()