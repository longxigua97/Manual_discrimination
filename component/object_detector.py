"""
object_detector.py
------------------
YOLOv8n 物体検出モジュール。

主要クラス ObjectDetector:
    detect(frame_bgr)           -> 検出結果のリスト（各対象を dict で表現）
    draw(frame_bgr, detections) -> フレームに検出ボックスとラベルを描画
    draw_predicted(frame_bgr, tracks) -> KF 予測中心を描画
"""

import cv2
import os
from typing import List, Tuple


# ─────────────────────────────────────────────
# 検出ボックス描画用の固定色 (B, G, R)
# ─────────────────────────────────────────────
DETECT_COLOR    = (0,  255,  0)    # 検出ボックス - 緑
PREDICTED_COLOR = (0,  200, 255)   # KF 予測円 - 薄いオレンジ
LOST_COLOR      = (0,   0,  200)   # ロストトラック - 赤


def _draw_dashed_line(frame_bgr, p1, p2, color, thickness=1, dash_len=8, gap_len=6):
    """破線を描画する。"""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 0:
        return

    step = float(dash_len + gap_len)
    nx = dx / length
    ny = dy / length

    dist = 0.0
    while dist < length:
        seg_start = dist
        seg_end = min(dist + dash_len, length)
        sx = int(x1 + nx * seg_start)
        sy = int(y1 + ny * seg_start)
        ex = int(x1 + nx * seg_end)
        ey = int(y1 + ny * seg_end)
        cv2.line(frame_bgr, (sx, sy), (ex, ey), color, thickness)
        dist += step


def _draw_dashed_rect(frame_bgr, x1, y1, x2, y2, color, thickness=1, dash_len=8, gap_len=6):
    """破線の矩形ボックスを描画する。"""
    _draw_dashed_line(frame_bgr, (x1, y1), (x2, y1), color, thickness, dash_len, gap_len)
    _draw_dashed_line(frame_bgr, (x2, y1), (x2, y2), color, thickness, dash_len, gap_len)
    _draw_dashed_line(frame_bgr, (x2, y2), (x1, y2), color, thickness, dash_len, gap_len)
    _draw_dashed_line(frame_bgr, (x1, y2), (x1, y1), color, thickness, dash_len, gap_len)


class ObjectDetector:
    """
    YOLOv8n でフレーム内の物体を検出する。

        複数のバックエンド高速化をサポート:
            - 'pt'   : 素の PyTorch（最遅）
            - 'ncnn' : NCNN 形式（Raspberry Pi で最速、推奨）
      - 'onnx' : ONNX Runtime

    Examples
    --------
    # NCNN が最速（推奨）
    detector = ObjectDetector(model_path="yolov8n_ncnn_model", imgsz=320)

    # ONNX は次点
    detector = ObjectDetector(model_path="yolov8n.onnx", imgsz=320)
    """

    def __init__(
        self,
        model_path: str = "yolov26n_ncnn_model",
        conf_threshold: float = 0.25,
        imgsz: int = 320,          # 推論解像度を下げるのが最も簡単な高速化手段
        half: bool = False,        # FP16。NCNN/ONNX で利用可能
        num_threads: int = 4,
    ):
        """
        Parameters
        ----------
        model_path     : モデルパス（.pt / .onnx / ncnn フォルダ）
        conf_threshold : 信頼度フィルタしきい値
        imgsz          : 推論入力サイズ（320 は 640 より約 3-4 倍高速）
        half           : FP16 を使用するか（Raspberry Pi CPU では False 推奨）
        num_threads    : CPU スレッド数（Raspberry Pi 5 なら 4 推奨）
        """
        self.num_threads = max(1, int(num_threads))

        # 主要バックエンドのスレッド数を明示的に制限し、CPU コア数を制御する。
        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.num_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(self.num_threads)

        try:
            cv2.setNumThreads(self.num_threads)
        except Exception:
            pass

        try:
            import torch
            torch.set_num_threads(self.num_threads)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.half = half
        self.class_names = self.model.names
        self._last_center_by_label = {}

    # ── 検出 ──────────────────────────────────
    def detect(self, frame_bgr) -> List[dict]:
        """
        BGR フレームに対して推論を実行する。

        Returns
        -------
        list[dict] 各要素の内容:
          - 'label'  : str
          - 'conf'   : float
          - 'bbox'   : (x1, y1, x2, y2)
          - 'center' : (cx, cy)
        """
        results = self.model(
            frame_bgr,
            conf=self.conf_threshold,
            imgsz=self.imgsz,      # 推論解像度
            half=self.half,
            verbose=False,
        )
        detections: List[dict] = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = self.class_names.get(cls, str(cls))
                cx    = (x1 + x2) // 2
                cy    = (y1 + y2) // 2
                if y2 >= 460:  # 低すぎる検出ボックスを除外（床や背景の可能性）
                    continue

                detections.append(
                    {
                        "label":  label,
                        "conf":   conf,
                        "bbox":   (x1, y1, x2, y2),
                        "center": (cx, cy),
                    }
                )

        filtered = self._filter_one_per_label(detections)
        self._last_center_by_label = {
            det["label"]: det["center"]
            for det in filtered
        }

        return filtered

    def _filter_one_per_label(self, detections: List[dict]) -> List[dict]:
        grouped = {}
        for det in detections:
            label = det["label"]
            grouped.setdefault(label, []).append(det)

        selected: List[dict] = []
        for label, items in grouped.items():
            if len(items) == 1:
                selected.append(items[0])
                continue

            prev_center = self._last_center_by_label.get(label)
            if prev_center is None:
                picked = max(items, key=lambda d: d["conf"])
            else:
                px, py = prev_center
                picked = min(
                    items,
                    key=lambda d: (d["center"][0] - px) ** 2 + (d["center"][1] - py) ** 2,
                )

            selected.append(picked)

        return selected

    # ── 検出結果の描画 ──────────────────────────
    def draw(self, frame_bgr, detections: List[dict], scale: Tuple[float, float] = (1.0, 1.0)) -> None:
        """
        フレーム上に検出ボックス、クラスラベル、信頼度を描画する。
        
        Parameters
        ----------
        scale : (scale_x, scale_y)
            座標スケール。
        """
        sx, sy = scale
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            label = f"{det['label']} {det['conf']:.2f}"
            cx, cy = det.get("center", ((x1 + x2) // 2, (y1 + y2) // 2))
            cx, cy = int(cx * sx), int(cy * sy)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), DETECT_COLOR, 2)
            text_y = max(y1 - 16, 32)
            cv2.putText(
                frame_bgr,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                DETECT_COLOR,
                2,
            )
            # 検出フレームで中心点を表示
            cv2.circle(frame_bgr, (cx, cy), 5, DETECT_COLOR, -1)

    # ── KF 予測トラックの描画 ──────────────────────
    def draw_predicted(self, frame_bgr, tracks: List[dict], scale: Tuple[float, float] = (1.0, 1.0)) -> None:
        """
        フレーム上に ObjectKFTracker の予測中心とトラック ID を描画する。

        Parameters
        ----------
        tracks : ObjectKFTracker.active_tracks() の戻り値
        scale : (scale_x, scale_y)
            座標スケール。
        """
        sx, sy = scale
        for t in tracks:
            px, py = int(t["pred_center"][0] * sx), int(t["pred_center"][1] * sy)
            lost   = t["lost_count"] > 0
            color  = LOST_COLOR if lost else PREDICTED_COLOR

            # 予測中心の円
            cv2.circle(frame_bgr, (px, py), 7, color, -1)

            # 予測ボックスを予測中心へ追従させる: 最新 bbox の幅高さを使い pred_center を中心に再構築
            if (not lost) and t["bbox"] is not None:
                x1, y1, x2, y2 = t["bbox"]
                bw = max(2, int((x2 - x1) * sx))
                bh = max(2, int((y2 - y1) * sy))
                pred_x1 = px - bw // 2
                pred_y1 = py - bh // 2
                pred_x2 = pred_x1 + bw
                pred_y2 = pred_y1 + bh
                _draw_dashed_rect(frame_bgr, pred_x1, pred_y1, pred_x2, pred_y2, color, thickness=1)

            # トラック ID とカテゴリ
            label = f"#{t['track_id']} {t['label']}_KF"
            if lost:
                label += " (pred)"

            text_x = px + 14
            text_y = max(py - 14, 32)
            cv2.putText(
                frame_bgr,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
            )
