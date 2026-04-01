from __future__ import annotations
from gooey import Gooey, GooeyParser
import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from apriltag_common import (
    build_detector,
    grid_cell,
    laplacian_sharpness,
    order_corners_tltrbrbl,
    refine_corners_subpix,
    roll_img_deg,
    tag_width_px,
    trapezoid_ratios,
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}


def per_view_reproj_error(objp, imgp, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(imgp - proj, axis=1)))


def parse_args() -> argparse.Namespace:
    p = GooeyParser(description="Calibração intrínseca com um ou mais vídeos locais usando AprilTag.")
    p.add_argument("--video", nargs="+", required=True, widget="MultiFileChooser",  help="Seleciona videos para calibração")
    p.add_argument("--out-dir", default="output_calib", help="Pasta de saída.")
    p.add_argument("--tag-family", default="tag36h11")
    p.add_argument("--tag-id", type=int, default=35)
    p.add_argument("--tag-size-mm", type=float, gooey_options={'min':0, 'max':150, 'increment':0.25}, default=45.0)
    p.add_argument("--process-every", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0 = sem limite por vídeo")
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--min-sharp-abs", type=float, default=10.0)
    p.add_argument("--seed-views", type=int, default=30)
    p.add_argument("--seed-min-sharp", type=float, default=15.0)
    p.add_argument("--seed-min-width-px", type=float, widget="IntegerField", gooey_options={"min":10, "max":240, "increment": 1}, default=240.0)
    p.add_argument("--min-tag-width-px", type=float, widget="IntegerField", gooey_options={"min":10, "max":240, "increment": 1}, default=220.0)
    p.add_argument("--top-sharpness-keep", type=float, default=0.70)
    p.add_argument("--grid-n", type=int, default=3)
    p.add_argument("--per-cell-keep", type=int, default=120)
    p.add_argument("--target-views", type=int, default=120)
    p.add_argument("--min-views", type=int, default=40)
    p.add_argument("--diversity-alpha", type=float, default=0.20)
    p.add_argument("--subpix-win", type=int, default=11)
    p.add_argument("--outlier-iters", type=int, default=4)
    p.add_argument("--remove-worst-fraction", type=float, default=0.08)
    p.add_argument("--min-keep-views", type=int, default=40)
    p.add_argument("--min-grid-ok", type=int, default=5)
    p.add_argument("--max-calib-views", type=int, default=80)
    p.add_argument("--min-frame-gap", type=int, default=8)
    p.add_argument("--fix-k1", action="store_true")
    p.add_argument("--fix-k2", action="store_true")
    p.add_argument("--fix-k3", action="store_true")
    return p.parse_args()


def resolve_video_paths(items: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for fp in sorted(p.iterdir()):
                if fp.is_file() and fp.suffix in VIDEO_EXTS:
                    out.append(fp)
        else:
            raise FileNotFoundError(f"Caminho não encontrado: {p}")
    out = [p for p in out if p.suffix in VIDEO_EXTS or p.is_file()]
    if not out:
        raise FileNotFoundError("Nenhum vídeo válido encontrado nos caminhos informados.")
    return out


def coverage_count(cands, grid_n: int) -> int:
    return len({(c["gy"], c["gx"]) for c in cands})


def principal_point_valid(Km, image_size_):
    w_, h_ = image_size_
    cx_, cy_ = float(Km[0, 2]), float(Km[1, 2])
    return np.isfinite(cx_) and np.isfinite(cy_) and (0.0 <= cx_ < w_) and (0.0 <= cy_ < h_)


def _seed_quality(c, W, H):
    sharp = float(c["sharp"])
    width = float(c["width_px"])
    tb = float(c["tb"])
    lr = float(c["lr"])
    cx_n = float(c["cx"]) / float(W)
    cy_n = float(c["cy"]) / float(H)
    center_pen = abs(cx_n - 0.5) + abs(cy_n - 0.5)
    trap_pen = abs(np.log(max(tb, 1e-6))) + abs(np.log(max(lr, 1e-6)))
    return 1.2 * sharp + 0.35 * width - 40.0 * center_pen - 20.0 * trap_pen


def _filter_seed_candidates(cands, W, H, min_sharp, min_width, center_margin, trap_min, trap_max):
    out = []
    for c in cands:
        sharp = float(c["sharp"])
        width = float(c["width_px"])
        tb = float(c["tb"])
        lr = float(c["lr"])
        cx_n = float(c["cx"]) / float(W)
        cy_n = float(c["cy"]) / float(H)
        if sharp < min_sharp:
            continue
        if width < min_width:
            continue
        if not (center_margin <= cx_n <= 1.0 - center_margin):
            continue
        if not (center_margin <= cy_n <= 1.0 - center_margin):
            continue
        if not (trap_min <= tb <= trap_max):
            continue
        if not (trap_min <= lr <= trap_max):
            continue
        out.append(c)
    return out

@Gooey(
    program_name="Calibração LVA - AprilTag",
    language='english',
    default_size=(800, 700)
)
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = resolve_video_paths(args.video)
    detector = build_detector(args.tag_family)

    cands = []
    global_frame_idx = 0
    image_size = None

    print("Vídeos de entrada:", flush=True)
    for vp in video_paths:
        print(f"- {vp}", flush=True)

    for vid_idx, video_path in enumerate(video_paths, start=1):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Não consegui abrir o vídeo: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = 30.0 if src_fps is None or src_fps <= 1e-3 else src_fps

        if image_size is None:
            image_size = (W, H)
        elif image_size != (W, H):
            raise RuntimeError(
                f"Todos os vídeos precisam ter a mesma resolução. Esperado {image_size}, mas {video_path.name} tem {(W, H)}."
            )

        print(
            f"\n[Vídeo {vid_idx}/{len(video_paths)}] {video_path.name} | {W}x{H} | fps≈{src_fps:.2f} | process_every={args.process_every}",
            flush=True,
        )

        local_frame_idx = 0
        seen = 0
        max_frames = None if args.max_frames <= 0 else args.max_frames

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            local_frame_idx += 1
            global_frame_idx += 1

            if local_frame_idx % args.print_every == 0:
                print(
                    f"[LENDO] video={video_path.name} | frame={local_frame_idx} | candidatos={len(cands)}",
                    flush=True,
                )

            if args.process_every > 1 and local_frame_idx % args.process_every != 0:
                continue
            if max_frames is not None and seen >= max_frames:
                break
            seen += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = laplacian_sharpness(gray)
            res = detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
            rsel = next((r for r in res if int(r.tag_id) == args.tag_id), None)
            if rsel is None:
                continue

            c4 = order_corners_tltrbrbl(np.array(rsel.corners, dtype=np.float64))
            c4 = refine_corners_subpix(gray, c4, win=args.subpix_win)
            width = tag_width_px(c4)
            if width < args.min_tag_width_px:
                continue

            cx, cy = float(np.mean(c4[:, 0])), float(np.mean(c4[:, 1]))
            gx, gy = grid_cell(cx, cy, W, H, args.grid_n)
            tb, lr = trapezoid_ratios(c4)
            roll = roll_img_deg(c4)

            cands.append(
                {
                    "global_frame_idx": global_frame_idx,
                    "frame_idx": local_frame_idx,
                    "video_path": str(video_path),
                    "video_name": video_path.name,
                    "sharp": sharp,
                    "width_px": width,
                    "cx": cx,
                    "cy": cy,
                    "gx": gx,
                    "gy": gy,
                    "tb": tb,
                    "lr": lr,
                    "roll": roll,
                    "imgp": c4.astype(np.float64),
                }
            )

            if len(cands) % 25 == 0:
                print(
                    f"[TAG] video={video_path.name} | frame={local_frame_idx} | candidatos={len(cands)} | sharp={sharp:.1f} | width_px={width:.1f}",
                    flush=True,
                )

        cap.release()

    if image_size is None:
        raise RuntimeError("Nenhum frame foi lido.")

    W, H = image_size
    print(f"\nCandidatos brutos: {len(cands)}", flush=True)
    print(f"Cobertura bruta: {coverage_count(cands, args.grid_n)}/{args.grid_n * args.grid_n}", flush=True)
    if len(cands) < args.min_views:
        raise RuntimeError(f"Poucos candidatos ({len(cands)}).")

    # Filtro de nitidez corrigido
    sharp_vals = np.array([c["sharp"] for c in cands], dtype=np.float64)
    sharp_thr_pct = float(np.quantile(sharp_vals, 1.0 - args.top_sharpness_keep))
    sharp_thr = max(sharp_thr_pct, float(args.min_sharp_abs))
    cands2 = [c for c in cands if float(c["sharp"]) >= sharp_thr]
    print(f"Filtro nitidez: mantendo {len(cands2)}/{len(cands)} (thr≈{sharp_thr:.1f})", flush=True)
    print(f"Cobertura após nitidez: {coverage_count(cands2, args.grid_n)}/{args.grid_n * args.grid_n}", flush=True)
    if len(cands2) < args.min_views:
        raise RuntimeError("Poucas vistas após filtro de nitidez. Reduza --min-sharp-abs ou grave vídeos mais nítidos.")

    # Deduplicação temporal
    cands2 = sorted(cands2, key=lambda c: (str(c.get("video_name", "")), int(c["frame_idx"])))
    dedup = []
    last_frame_by_video = {}
    for c in cands2:
        vname = str(c.get("video_name", "video"))
        fidx = int(c["frame_idx"])
        if vname not in last_frame_by_video or (fidx - last_frame_by_video[vname]) >= args.min_frame_gap:
            dedup.append(c)
            last_frame_by_video[vname] = fidx
    cands2 = dedup
    print(f"Após deduplicação temporal: {len(cands2)}", flush=True)
    print(f"Cobertura após deduplicação: {coverage_count(cands2, args.grid_n)}/{args.grid_n * args.grid_n}", flush=True)

    by_cell = {(gy, gx): [] for gy in range(args.grid_n) for gx in range(args.grid_n)}
    for c in cands2:
        by_cell[(c["gy"], c["gx"])].append(c)
    for k in list(by_cell.keys()):
        by_cell[k].sort(key=lambda d: (d["sharp"], d["width_px"]), reverse=True)
        by_cell[k] = by_cell[k][: args.per_cell_keep]

    pool = []
    for k in by_cell:
        pool.extend(by_cell[k])
    pool = list({(c["video_name"], c["frame_idx"]): c for c in pool}.values())
    if len(pool) < args.min_views:
        raise RuntimeError(f"Poucas vistas no pool após filtros ({len(pool)}).")
    print(f"Pool após per-cell: {len(pool)} | cobertura grade: {coverage_count(pool, args.grid_n)}/{args.grid_n * args.grid_n}", flush=True)

    def features(c):
        x = c["cx"] / W
        y = c["cy"] / H
        w = np.log(max(c["width_px"], 1e-6))
        tb = np.log(max(c["tb"], 1e-6))
        lr = np.log(max(c["lr"], 1e-6))
        rl = c["roll"] / 90.0
        sh = np.log(max(c["sharp"], 1e-6))
        return np.array([x, y, w, tb, lr, rl, sh], dtype=np.float64)

    F = np.vstack([features(c) for c in pool])
    mu = F.mean(axis=0)
    sd = F.std(axis=0) + 1e-9
    Fn = (F - mu) / sd

    q_width = np.array([c["width_px"] for c in pool], dtype=np.float64)
    q_sharp = np.array([c["sharp"] for c in pool], dtype=np.float64)
    q_width = (q_width - q_width.min()) / (q_width.max() - q_width.min() + 1e-12)
    q_sharp = (q_sharp - q_sharp.min()) / (q_sharp.max() - q_sharp.min() + 1e-12)
    q = 0.45 * q_width + 0.55 * q_sharp

    idx0 = int(np.argmax(q))
    selected_idx = [idx0]
    min_dist = np.linalg.norm(Fn - Fn[idx0], axis=1)
    while len(selected_idx) < min(args.target_views, len(pool)):
        score = (1.0 - args.diversity_alpha) * min_dist + args.diversity_alpha * q
        score[selected_idx] = -1e9
        j = int(np.argmax(score))
        selected_idx.append(j)
        d = np.linalg.norm(Fn - Fn[j], axis=1)
        min_dist = np.minimum(min_dist, d)

    selected = [pool[i] for i in selected_idx]
    print(f"Selecionadas: {len(selected)} | cobertura grade: {coverage_count(selected, args.grid_n)}/{args.grid_n * args.grid_n}", flush=True)
    if len(selected) < args.min_views:
        raise RuntimeError(f"Poucas vistas selecionadas ({len(selected)}).")
    if coverage_count(selected, args.grid_n) < args.min_grid_ok:
        raise RuntimeError(f"Cobertura espacial insuficiente: {coverage_count(selected, args.grid_n)}/{args.grid_n * args.grid_n}.")
    if len(selected) > args.max_calib_views:
        selected = selected[: args.max_calib_views]
        print(f"Ajuste de segurança: limitando para {len(selected)} vistas antes da calibração.", flush=True)

    s = float(args.tag_size_mm)
    objp = np.array([
        [-s / 2, +s / 2, 0.0],
        [+s / 2, +s / 2, 0.0],
        [+s / 2, -s / 2, 0.0],
        [-s / 2, -s / 2, 0.0],
    ], dtype=np.float32).reshape(-1, 1, 3)

    object_points_list = [objp.copy() for _ in selected]
    image_points_list = [c["imgp"].astype(np.float32).reshape(-1, 1, 2) for c in selected]

    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST
    if args.fix_k1:
        flags |= cv2.CALIB_FIX_K1
    if args.fix_k2:
        flags |= cv2.CALIB_FIX_K2
    if args.fix_k3:
        flags |= cv2.CALIB_FIX_K3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 120, 1e-10)

    def make_seed(obj_list, img_list):
        K_seed = cv2.initCameraMatrix2D(obj_list, img_list, image_size, 0)
        if not principal_point_valid(K_seed, image_size):
            K_seed = K_seed.astype(np.float64)
            K_seed[0, 2] = image_size[0] / 2.0
            K_seed[1, 2] = image_size[1] / 2.0
        dist_seed = np.zeros((5, 1), dtype=np.float64)
        return K_seed, dist_seed

    def run_calib(obj_list, img_list, K_init, dist_init, flags_local):
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_list, img_list, image_size, K_init, dist_init, flags=flags_local, criteria=criteria
        )
        if (not np.isfinite(rms)) or (not np.isfinite(K).all()) or (not np.isfinite(dist).all()):
            return np.nan, K, dist, None
        if not principal_point_valid(K, image_size):
            return np.nan, K, dist, None
        per_view = np.array([
            per_view_reproj_error(obj_list[i], img_list[i].reshape(-1, 2), rvecs[i], tvecs[i], K, dist)
            for i in range(len(obj_list))
        ], dtype=np.float64)
        return rms, K, dist, per_view

    # Seed pool: mais estável (nítida, central, pouco trapezoidal) antes da diversidade
    seed_min_sharp = max(float(args.seed_min_sharp), float(args.min_sharp_abs))
    seed_min_width = max(float(args.seed_min_width_px), float(args.min_tag_width_px))

    seed_pool = _filter_seed_candidates(selected, W, H, seed_min_sharp, seed_min_width, 0.15, 0.80, 1.25)
    seed_mode = "estrito"
    if len(seed_pool) < max(20, min(args.seed_views, len(selected))):
        seed_pool = _filter_seed_candidates(
            selected,
            W,
            H,
            max(float(args.min_sharp_abs), 12.0),
            max(float(args.min_tag_width_px), 220.0),
            0.10,
            0.70,
            1.43,
        )
        seed_mode = "relaxado"
    if len(seed_pool) < max(16, min(args.seed_views, len(selected))):
        seed_pool = sorted(selected, key=lambda c: _seed_quality(c, W, H), reverse=True)[: max(30, args.seed_views)]
        seed_mode = "fallback-qualidade"
    else:
        seed_pool = sorted(seed_pool, key=lambda c: _seed_quality(c, W, H), reverse=True)[: max(50, args.seed_views)]

    print(f"[Calib-Seed] Pool {seed_mode}: {len(seed_pool)} candidatas", flush=True)

    seed_obj_all = [objp.copy() for _ in seed_pool]
    seed_img_all = [c["imgp"].astype(np.float32).reshape(-1, 1, 2) for c in seed_pool]
    F_seed = np.vstack([features(c) for c in seed_pool])
    mu_seed = F_seed.mean(axis=0)
    sd_seed = F_seed.std(axis=0) + 1e-9
    Fn_seed = (F_seed - mu_seed) / sd_seed
    q_seed = np.array([_seed_quality(c, W, H) for c in seed_pool], dtype=np.float64)
    q_seed = (q_seed - q_seed.min()) / (q_seed.max() - q_seed.min() + 1e-12)
    idx0 = int(np.argmax(q_seed))
    seed_idx = [idx0]
    min_dist = np.linalg.norm(Fn_seed - Fn_seed[idx0], axis=1)
    seed_n = min(args.seed_views, len(seed_pool))
    while len(seed_idx) < seed_n:
        score = 0.70 * min_dist + 0.30 * q_seed
        score[seed_idx] = -1e9
        j = int(np.argmax(score))
        seed_idx.append(j)
        d = np.linalg.norm(Fn_seed - Fn_seed[j], axis=1)
        min_dist = np.minimum(min_dist, d)

    seed_obj = [seed_obj_all[i] for i in seed_idx]
    seed_img = [seed_img_all[i] for i in seed_idx]

    print(f"\n[Calib-Seed] Entrando no calibrateCamera com {len(seed_obj)} vistas para obter chute inicial...", flush=True)
    flags_seed = flags | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    K_seed0, dist_seed0 = make_seed(seed_obj, seed_img)
    rms_seed, K_seed, dist_seed, _ = run_calib(seed_obj, seed_img, K_seed0, dist_seed0, flags_seed)

    if not np.isfinite(rms_seed):
        flags_seed |= cv2.CALIB_FIX_K1
        seed_obj2 = seed_obj[: min(20, len(seed_obj))]
        seed_img2 = seed_img[: min(20, len(seed_img))]
        K_seed0, dist_seed0 = make_seed(seed_obj2, seed_img2)
        print(f"[Calib-Seed] Primeira tentativa falhou. Tentando seed conservador com {len(seed_obj2)} vistas...", flush=True)
        rms_seed, K_seed, dist_seed, _ = run_calib(seed_obj2, seed_img2, K_seed0, dist_seed0, flags_seed)

    if not np.isfinite(rms_seed):
        raise RuntimeError("Falha ao obter chute inicial estável.")

    print(f"[Calib-Seed] RMS={rms_seed:.4f} px | seed_views={len(seed_obj)}", flush=True)

    # Crescimento progressivo até 80 vistas
    obj_list = object_points_list
    img_list = image_points_list
    meta_list = selected

    order_idx = sorted(range(len(meta_list)), key=lambda i: (-float(meta_list[i]["sharp"]), -float(meta_list[i]["width_px"])))
    obj_ord = [obj_list[i] for i in order_idx]
    img_ord = [img_list[i] for i in order_idx]
    meta_ord = [meta_list[i] for i in order_idx]

    current_n = min(len(seed_obj), len(obj_ord))
    current_obj = obj_ord[:current_n]
    current_img = img_ord[:current_n]
    current_meta = meta_ord[:current_n]

    K_curr, dist_curr = K_seed, dist_seed
    rms_curr, K_curr, dist_curr, per_view_curr = run_calib(current_obj, current_img, K_curr, dist_curr, flags)

    if not np.isfinite(rms_curr):
        raise RuntimeError("Falha logo após a seed. Tente aumentar --min-sharp-abs ou --min-tag-width-px.")

    for target_n in range(current_n + 10, len(obj_ord) + 10, 10):
        target_n = min(target_n, len(obj_ord))
        trial_obj = obj_ord[:target_n]
        trial_img = img_ord[:target_n]
        K_trial, dist_trial = make_seed(trial_obj, trial_img)

        if principal_point_valid(K_curr, image_size):
            K_trial = K_curr.copy()
            dist_trial = dist_curr.copy()

        rms_try, K_try, dist_try, per_view_try = run_calib(trial_obj, trial_img, K_trial, dist_trial, flags)

        if np.isfinite(rms_try):
            current_obj, current_img = trial_obj, trial_img
            current_meta = meta_ord[:target_n]
            rms_curr, K_curr, dist_curr, per_view_curr = rms_try, K_try, dist_try, per_view_try
            print(f"[Calib-Prog] RMS={rms_curr:.4f}px | vistas={len(current_obj)}", flush=True)
        else:
            K_trial, dist_trial = make_seed(trial_obj, trial_img)
            rms_try, K_try, dist_try, per_view_try = run_calib(trial_obj, trial_img, K_trial, dist_trial, flags_seed)
            if np.isfinite(rms_try):
                current_obj, current_img = trial_obj, trial_img
                current_meta = meta_ord[:target_n]
                rms_curr, K_curr, dist_curr, per_view_curr = rms_try, K_try, dist_try, per_view_try
                print(f"[Calib-Prog] RMS={rms_curr:.4f}px | vistas={len(current_obj)} | fallback-seed", flush=True)
            else:
                print(f"[Calib-Prog] Falhou ao expandir para {target_n} vistas; mantendo {len(current_obj)} vistas.", flush=True)
                break

    print(f"\n[Calib] Entrando no refinamento final com {len(current_obj)} vistas...", flush=True)
    obj_list, img_list, meta_list = current_obj, current_img, current_meta
    rms, K, dist, per_view = rms_curr, K_curr, dist_curr, per_view_curr

    print(f"[Calib] Iter 0: RMS={rms:.4f} | vistas={len(obj_list)}", flush=True)
    print("K iter 0:\n", K, flush=True)
    print("dist iter 0:\n", dist.ravel(), flush=True)

    for it in range(1, args.outlier_iters + 1):
        n = len(per_view)
        k_remove = int(round(n * args.remove_worst_fraction))
        if n - k_remove < args.min_keep_views:
            k_remove = max(0, n - args.min_keep_views)
        if k_remove < 1:
            break

        idx_sorted = np.argsort(per_view)
        keep_idx = idx_sorted[: n - k_remove]

        obj_list = [obj_list[i] for i in keep_idx]
        img_list = [img_list[i] for i in keep_idx]
        meta_list = [meta_list[i] for i in keep_idx]

        K_seed_it, dist_seed_it = make_seed(obj_list, img_list)
        rms, K, dist, per_view = run_calib(obj_list, img_list, K_seed_it, dist_seed_it, flags)

        if not np.isfinite(rms):
            raise RuntimeError(
                f"Calibração falhou na iteração {it}. Tente aumentar --min-sharp-abs, elevar --min-tag-width-px ou fixar mais coeficientes."
            )

        print(f"[Calib] Iter {it}: RMS={rms:.4f}px | vistas={len(obj_list)} | removeu {k_remove}", flush=True)

    print("\n--- RESULTADOS FINAIS ---", flush=True)
    print(f"RMS final: {rms:.4f} px | vistas: {len(obj_list)}", flush=True)
    print("K:\n", K, flush=True)
    print("dist:\n", dist.ravel(), flush=True)
    print(
        f"Erro por vista (px): média={float(per_view.mean()):.3f} | mediana={float(np.median(per_view)):.3f} | max={float(per_view.max()):.3f}",
        flush=True,
    )

    out_json = out_dir / "calib_intrinsics_apriltag.json"
    out_npz = out_dir / "calib_intrinsics_apriltag.npz"

    payload = {
        "video_paths": [str(v) for v in video_paths],
        "image_size": {"width": W, "height": H},
        "apriltag": {"family": args.tag_family, "id": int(args.tag_id), "tag_size_mm": float(args.tag_size_mm)},
        "selection": {
            "process_every_n_frames": args.process_every,
            "min_tag_width_px": args.min_tag_width_px,
            "min_sharp_abs": args.min_sharp_abs,
            "top_sharpness_keep": args.top_sharpness_keep,
            "grid_n": args.grid_n,
            "per_cell_keep": args.per_cell_keep,
            "target_views": args.target_views,
            "diversity_alpha": args.diversity_alpha,
            "subpix_win": args.subpix_win,
            "min_grid_ok": args.min_grid_ok,
            "max_calib_views": args.max_calib_views,
            "min_frame_gap": args.min_frame_gap,
            "seed_views": args.seed_views,
            "seed_min_sharp": args.seed_min_sharp,
            "seed_min_width_px": args.seed_min_width_px,
        },
        "calibration": {
            "rms_px": float(rms),
            "K": K.tolist(),
            "dist": dist.ravel().tolist(),
            "per_view_error_px": per_view.tolist(),
        },
        "accepted_views": [
            {
                k: c[k]
                for k in ["video_name", "frame_idx", "sharp", "width_px", "cx", "cy", "gx", "gy", "tb", "lr", "roll"]
            }
            for c in meta_list
        ],
    }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        out_npz,
        K=K,
        dist=dist,
        image_size=np.array([W, H], dtype=np.int32),
        rms=np.array([rms], dtype=np.float64),
        per_view_error=np.array(per_view, dtype=np.float64),
    )

    print(f"\nSalvo:\n- {out_json}\n- {out_npz}", flush=True)


if __name__ == "__main__":
    main()