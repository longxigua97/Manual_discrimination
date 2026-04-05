"""
kf_tracker.py
-------------
Kalman Filter トラッキングモジュール。

以下の 2 つの高水準クラスを提供:
  - HandKFTracker   : 各手の 5 指先に独立した KF を保持
  - ObjectKFTracker : 任意数の物体に KF を適用し、最近傍対応で ID を付与
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict


# ─────────────────────────────────────────────
# 基本: 単一点 Kalman Filter [x, y, vx, vy]
# ─────────────────────────────────────────────
class PointKF:
    """単一の 2D 点 (x, y) を追跡する。状態ベクトルに速度 (vx, vy) を含む。"""

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 1e-1):
        kf = cv2.KalmanFilter(4, 2)
        # 状態遷移行列: 等速モデル
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32
        )
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.kf = kf
        self.initialized = False
        self._predicted_this_cycle = False  # この周期で predict 済みかどうか

    def init(self, x: float, y: float):
        """初期座標で状態を初期化する。"""
        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], np.float32)
        self.initialized = True
        self._predicted_this_cycle = False

    def predict(self) -> Tuple[float, float]:
        """予測座標 (px, py) を返し、状態を 1 ステップ進める。"""
        p = self.kf.predict()
        self._predicted_this_cycle = True
        return float(p[0]), float(p[1])

    def correct(self, x: float, y: float):
        """観測値を与えて KF を更新する。未初期化なら init を先に実行する。"""
        if not self.initialized:
            self.init(x, y)
            return
        # この周期でまだ predict していなければ先に実行
        if not self._predicted_this_cycle:
            self.kf.predict()
        self.kf.correct(np.array([[x], [y]], np.float32))
        self._predicted_this_cycle = False  # 次周期に向けてリセット

    def get_state(self) -> Tuple[float, float]:
        """現在の状態を取得する（時間ステップは進めない）。"""
        s = self.kf.statePost
        return float(s[0]), float(s[1])


# ─────────────────────────────────────────────
# 指先 KF マネージャ
# ─────────────────────────────────────────────
FINGERTIP_IDS   = (4, 8, 12, 16, 20)
FINGERTIP_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")

class HandKFTracker:
    """
    最大 max_hands 本の手に対し、各 5 指先の PointKF を保持する。

    update(hand_tips_list) : 検出した指先座標を入力して KF を更新
    predict()              : 各手 5 指先の予測座標を返す（状態も進める）
    """

    def __init__(self, max_hands: int = 2):
        self.max_hands = max_hands
        # kfs[hand_idx][tip_idx] = PointKF
        self.kfs: List[List[PointKF]] = [
            [PointKF() for _ in FINGERTIP_IDS]
            for _ in range(max_hands)
        ]
        self.ready = [[False] * len(FINGERTIP_IDS) for _ in range(max_hands)]

    def update(self, hand_tips_list: List[Optional[List[Tuple[int, int]]]]):
        """
        Parameters
        ----------
        hand_tips_list : list[Optional[list[(x,y)]]]
            外側は手インデックス、内側は 5 指先座標。欠損手は None 可。
        """
        for hand_idx, tips in enumerate(hand_tips_list[:self.max_hands]):
            if tips is None:
                continue
            for tip_idx, (x, y) in enumerate(tips):
                self.kfs[hand_idx][tip_idx].correct(float(x), float(y))
                self.ready[hand_idx][tip_idx] = True

    def predict(self) -> List[List[Optional[Tuple[float, float]]]]:
        """
        各手・各指先の予測座標を返し、KF 状態を進める。
        当該 KF が未初期化の場合は None を返す。
        """
        result = []
        for hand_idx in range(self.max_hands):
            hand_preds = []
            for tip_idx in range(len(FINGERTIP_IDS)):
                if self.ready[hand_idx][tip_idx]:
                    hand_preds.append(self.kfs[hand_idx][tip_idx].predict())
                else:
                    hand_preds.append(None)
            result.append(hand_preds)
        return result


# ─────────────────────────────────────────────
# 物体 KF マネージャ（最近傍マッチング）
# ─────────────────────────────────────────────
class _ObjectTrack:
    """単一物体トラックの内部状態。"""
    def __init__(self, track_id: int, cx: float, cy: float, label: str, bbox: Tuple):
        self.track_id   = track_id
        self.label      = label
        self.bbox       = bbox           # 最新の検出ボックス (x1,y1,x2,y2)
        self.pred_center: Tuple[float, float] = (cx, cy)
        self.lost_count = 0
        self.kf = PointKF()
        self.kf.init(cx, cy)


class ObjectKFTracker:
    """
    検出物体に ID を割り当て、PointKF で中心位置を予測する。

    update(detections) : 現フレーム検出を入力 -> 対応/新規トラック作成と KF 更新
    predict_all()      : 全アクティブトラックを予測
    active_tracks()    : 現在のアクティブトラックを返す（予測中心と最新 bbox を含む）
    """

    def __init__(self, max_distance: int = 120, max_lost_frames: int = 8):
        """
        Parameters
        ----------
        max_distance     : 最近傍対応で許容する最大距離（ピクセル）
        max_lost_frames  : 連続未検出がこのフレーム数を超えたら削除
        """
        self.max_distance   = max_distance
        self.max_lost_frames = max_lost_frames
        self.tracks: Dict[int, _ObjectTrack] = {}
        self._next_id = 0

    def update(self, detections: List[dict]):
        """
        Parameters
        ----------
        detections : list[dict]
            各 dict は 'center'(cx,cy), 'label', 'bbox'(x1,y1,x2,y2), 'conf' を含む。
        """
        matched_track_ids = set()
        matched_det_indices = set()
        new_track_ids = set()

        # ── 最近傍マッチング ──
        for det_idx, det in enumerate(detections):
            cx, cy = det["center"]
            det_label = det.get("label")
            best_id, best_dist = None, float("inf")
            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                if det_label is not None and track.label != det_label:
                    continue
                px, py = track.pred_center
                dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, tid

            if best_id is not None and best_dist < self.max_distance:
                matched_track_ids.add(best_id)
                matched_det_indices.add(det_idx)
                t = self.tracks[best_id]
                t.kf.correct(float(cx), float(cy))
                # pred_center を補正後状態に更新
                t.pred_center = t.kf.get_state()
                t.label = det["label"]
                t.bbox  = det["bbox"]
                t.lost_count = 0

        # ── 新規トラック追加 ──
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                cx, cy = det["center"]
                new_track = _ObjectTrack(
                    self._next_id, cx, cy, det["label"], det["bbox"]
                )
                self.tracks[self._next_id] = new_track
                new_track_ids.add(self._next_id)
                self._next_id += 1

        # ── 未検出カウント更新と期限切れトラック削除 ──
        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in matched_track_ids and tid not in new_track_ids:
                track.lost_count += 1
                if track.lost_count > self.max_lost_frames:
                    to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

    def predict_all(self):
        """全アクティブトラックで KF 予測を実行し、内部 pred_center を更新する。"""
        for track in self.tracks.values():
            px, py = track.kf.predict()
            track.pred_center = (px, py)

    def active_tracks(self) -> List[dict]:
        """
        Returns
        -------
        list[dict] キー: track_id, label, bbox, pred_center, lost_count
        """
        return [
            {
                "track_id":    t.track_id,
                "label":       t.label,
                "bbox":        t.bbox,
                "pred_center": t.pred_center,
                "lost_count":  t.lost_count,
            }
            for t in self.tracks.values()
        ]
