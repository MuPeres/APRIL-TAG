from __future__ import annotations

import base64
import time
from typing import Optional

import cv2
import numpy as np
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

JS_PICK_AND_SNAPSHOT = r"""
(args) => {
  const dw = args.dw || 0;
  const dh = args.dh || 0;
  const preferCanvas = !!args.preferCanvas;
  const jpegQuality = args.jpegQuality || 0.95;

  function visibleScore(el) {
    const style = window.getComputedStyle(el);
    const visible = style && style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
    return visible ? 1 : 0;
  }

  function collectCanvases() {
    return [...document.querySelectorAll('canvas')].map((el, idx) => ({
      kind: 'canvas',
      idx,
      width: Number(el.width || 0),
      height: Number(el.height || 0),
      clientWidth: Number(el.clientWidth || 0),
      clientHeight: Number(el.clientHeight || 0),
      vis: visibleScore(el),
    }));
  }

  function collectImgs() {
    return [...document.querySelectorAll('img')].map((el, idx) => ({
      kind: 'img',
      idx,
      width: Number(el.naturalWidth || el.width || 0),
      height: Number(el.naturalHeight || el.height || 0),
      clientWidth: Number(el.clientWidth || 0),
      clientHeight: Number(el.clientHeight || 0),
      vis: visibleScore(el),
      src: el.currentSrc || el.src || '',
    }));
  }

  function score(item) {
    let s = 0;
    const realArea = item.width * item.height;
    const clientArea = item.clientWidth * item.clientHeight;

    s += realArea * 0.01;
    s += clientArea;

    if (item.vis) s += 1_000_000;
    if (preferCanvas && item.kind === 'canvas') s += 100_000;

    if (dw > 0 && dh > 0) {
      const err = Math.abs(item.width - dw) + Math.abs(item.height - dh);
      s -= err * 1000;
    }
    return s;
  }

  const candidates = [...collectCanvases(), ...collectImgs()]
    .filter((x) => x.width > 0 && x.height > 0)
    .sort((a, b) => score(b) - score(a));

  if (!candidates.length) {
    return { ok: false, error: 'Nenhum canvas/img válido encontrado.' };
  }

  const best = candidates[0];

  function snapshotCanvas(el) {
    return el.toDataURL('image/jpeg', jpegQuality);
  }

  function snapshotImg(el, outW, outH) {
    const off = document.createElement('canvas');
    off.width = outW;
    off.height = outH;
    const ctx = off.getContext('2d');
    ctx.drawImage(el, 0, 0, outW, outH);
    return off.toDataURL('image/jpeg', jpegQuality);
  }

  let dataUrl = null;

  if (best.kind === 'canvas') {
    const el = document.querySelectorAll('canvas')[best.idx];
    dataUrl = snapshotCanvas(el);
  } else {
    const el = document.querySelectorAll('img')[best.idx];
    dataUrl = snapshotImg(el, best.width, best.height);
  }

  return {
    ok: true,
    kind: best.kind,
    idx: best.idx,
    width: best.width,
    height: best.height,
    clientWidth: best.clientWidth,
    clientHeight: best.clientHeight,
    dataUrl,
  };
}
"""


class WebAppFrameSource:
    def __init__(
        self,
        url: str,
        desired_width: Optional[int] = None,
        desired_height: Optional[int] = None,
        headless: bool = False,
        viewport_width: int = 1600,
        viewport_height: int = 1200,
        prefer_canvas: bool = True,
        jpeg_quality: float = 0.95,
    ) -> None:
        self.url = url
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.prefer_canvas = prefer_canvas
        self.jpeg_quality = jpeg_quality

        self._play = None
        self._browser = None
        self._page = None
        self.last_info = None

    def start(self) -> None:
        self._play = sync_playwright().start()
        self._browser = self._play.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page(
            viewport={"width": self.viewport_width, "height": self.viewport_height}
        )
        self._page.goto(self.url, wait_until="domcontentloaded")
        self._page.wait_for_timeout(1500)

    def stop(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            if self._play is not None:
                self._play.stop()
        self._browser = None
        self._play = None
        self._page = None

    def _snapshot_payload(self):
        if self._page is None:
            raise RuntimeError("Fonte não iniciada. Chame start() primeiro.")

        try:
            payload = self._page.evaluate(
                JS_PICK_AND_SNAPSHOT,
                {
                    "dw": int(self.desired_width or 0),
                    "dh": int(self.desired_height or 0),
                    "preferCanvas": bool(self.prefer_canvas),
                    "jpegQuality": float(self.jpeg_quality),
                },
            )
        except PlaywrightError as exc:
            return {"ok": False, "error": str(exc)}

        return payload

    def get_frame(self):
        payload = self._snapshot_payload()
        if not payload or not payload.get("ok"):
            self.last_info = payload
            return None, payload

        data_url = payload.get("dataUrl")
        if not data_url or "," not in data_url:
            self.last_info = {"ok": False, "error": "dataUrl inválido."}
            return None, self.last_info

        raw = base64.b64decode(data_url.split(",", 1)[1])
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.last_info = {"ok": False, "error": "Falha ao decodificar frame."}
            return None, self.last_info

        info = {
            "ok": True,
            "kind": payload.get("kind"),
            "idx": payload.get("idx"),
            "width": int(payload.get("width", frame.shape[1])),
            "height": int(payload.get("height", frame.shape[0])),
            "clientWidth": int(payload.get("clientWidth", 0)),
            "clientHeight": int(payload.get("clientHeight", 0)),
        }
        self.last_info = info
        return frame, info

    def wait_first_frame(self, timeout_s: float = 15.0, sleep_s: float = 0.15):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            frame, info = self.get_frame()
            if frame is not None:
                return frame, info
            time.sleep(sleep_s)

        raise TimeoutError(
            f"Não foi possível obter frame do Web-App em {timeout_s:.1f}s. "
            f"Último info: {self.last_info}"
        )