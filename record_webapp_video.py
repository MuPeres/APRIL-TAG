from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from webapp_frame_source import WebAppFrameSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grava vídeo direto do preview nativo do Web-App da câmera."
    )
    p.add_argument("--url", required=True, help="URL do Web-App da câmera, ex.: http://169.254.125.172/")
    p.add_argument("--out-video", required=True, help="Arquivo MP4 de saída.")
    p.add_argument("--desired-width", type=int, default=1920)
    p.add_argument("--desired-height", type=int, default=1080)
    p.add_argument("--write-fps", type=float, default=10.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--display-scale", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    source = WebAppFrameSource(
        url=args.url,
        desired_width=args.desired_width,
        desired_height=args.desired_height,
        headless=args.headless,
    )
    source.start()

    try:
        frame, info = source.wait_first_frame()
        h, w = frame.shape[:2]

        out_path = Path(args.out_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(args.write_fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Falha ao abrir VideoWriter para: {out_path}")

        print(f"Gravando {w}x{h} em {out_path}")
        print("Teclas: q/ESC=sair | s=salvar PNG")

        while True:
            frame, info = source.get_frame()
            if frame is None:
                continue

            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            writer.write(frame)

            disp = frame
            if args.display_scale != 1.0:
                disp = cv2.resize(
                    frame,
                    (int(w * args.display_scale), int(h * args.display_scale)),
                    interpolation=cv2.INTER_AREA,
                )

            txt = (
                f"src={info.get('kind')} {info.get('width')}x{info.get('height')} "
                f"-> gravando {w}x{h}"
            )
            cv2.putText(
                disp,
                txt,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Record WebApp Video", disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                png_path = out_path.with_suffix(".png")
                cv2.imwrite(str(png_path), frame)
                print(f"Snapshot salvo em: {png_path}")

        writer.release()

    finally:
        source.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()