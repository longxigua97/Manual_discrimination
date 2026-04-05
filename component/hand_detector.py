"""
hand_detector.py
----------------
MediaPipe 手部検出モジュール。

主要クラス HandDetector:
    detect(frame_bgr)             -> MediaPipe の生結果
    get_fingertips(result, w, h)  -> 各手の 5 指先ピクセル座標リスト
    draw(frame_bgr, result)       -> フレームにキーポイントと骨格を描画
    draw_predicted(frame_bgr, predictions) -> KF 予測指先（円）を描画
    close()                       -> リソースを解放
"""

import cv2
import mediapipe as mp
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
# 5 本指先の MediaPipe ランドマークインデックス
# ─────────────────────────────────────────────
FINGERTIP_IDS   = (4, 8, 12, 16, 20)
FINGERTIP_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")

# 各指に対応する描画色 (B, G, R)
FINGERTIP_COLORS = [
    (0,   165, 255),   # 親指   - オレンジ
    (0,   255,   0),   # 人差し指 - 緑
    (255,   0,   0),   # 中指   - 青
    (0,   255, 255),   # 薬指   - 黄
    (255,   0, 255),   # 小指   - 紫
]


class HandDetector:
    """
    MediaPipe Hands で手のキーポイントを検出する。

    Examples
    --------
    detector = HandDetector()
    result   = detector.detect(frame_bgr)
    tips     = detector.get_fingertips(result, frame_w, frame_h)
    detector.draw(frame_bgr, result)
    detector.close()
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float  = 0.5,
    ):
        self._mp_hands  = mp.solutions.hands
        self._mp_draw   = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ── 検出 ──────────────────────────────────
    def detect(self, frame_bgr) -> Optional[object]:
        """
        BGR フレームに MediaPipe を適用し、生結果オブジェクトを返す。
        手が未検出の場合、result.multi_hand_landmarks は None。
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._hands.process(frame_rgb)

    # ── 指先座標の抽出 ──────────────────────────
    def get_fingertips(
        self,
        result,
        frame_w: int,
        frame_h: int,
    ) -> List[List[Tuple[int, int]]]:
        """
        MediaPipe 結果から全ての手の指先ピクセル座標を抽出する。

        Returns
        -------
        list[list[(x, y)]]
            外側は手の順序、内側は 5 つの指先 (x, y)。
            順序は FINGERTIP_IDS / FINGERTIP_NAMES と一致。
        """
        all_hands: List[List[Tuple[int, int]]] = []
        if result and result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                tips = []
                for tip_id in FINGERTIP_IDS:
                    lm = hand_landmarks.landmark[tip_id]
                    x  = int(lm.x * frame_w)
                    y  = int(lm.y * frame_h)
                    tips.append((x, y))
                all_hands.append(tips)
        return all_hands

    # ── 描画 ──────────────────────────────────
    def draw(self, frame_bgr, result, scale: Tuple[float, float] = (1.0, 1.0)) -> None:
        """
        フレーム上に MediaPipe 標準のキーポイントと骨格線を描画する。
        
        Parameters
        ----------
        scale : (scale_x, scale_y)
            座標スケール。検出座標を描画先フレームサイズへ写像するために使用。
        """
        if result and result.multi_hand_landmarks:
            frame_h, frame_w = frame_bgr.shape[:2]
            sx, sy = scale
            for hand_landmarks in result.multi_hand_landmarks:
                if sx == 1.0 and sy == 1.0:
                    # スケーリング不要: MediaPipe 標準描画を使用
                    self._mp_draw.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        self._mp_hands.HAND_CONNECTIONS,
                        self._mp_styles.get_default_hand_landmarks_style(),
                        self._mp_styles.get_default_hand_connections_style(),
                    )
                else:
                    # スケーリングが必要: キーポイントと線分を手動描画
                    landmarks = hand_landmarks.landmark
                    # 骨格線を描画
                    for connection in self._mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        start_lm = landmarks[start_idx]
                        end_lm = landmarks[end_idx]
                        x1 = int(start_lm.x * frame_w)
                        y1 = int(start_lm.y * frame_h)
                        x2 = int(end_lm.x * frame_w)
                        y2 = int(end_lm.y * frame_h)
                        cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # キーポイントを描画
                    for lm in landmarks:
                        x = int(lm.x * frame_w)
                        y = int(lm.y * frame_h)
                        cv2.circle(frame_bgr, (x, y), 4, (255, 0, 0), -1)

    def draw_predicted(
        self,
        frame_bgr,
        predictions: List[List[Optional[Tuple[float, float]]]],
        scale: Tuple[float, float] = (1.0, 1.0),
    ) -> None:
        """
        フレーム上に KF 予測の指先位置を描画する（中空円、実測と同色）。

        Parameters
        ----------
        predictions : list[list[(px, py) or None]]
            HandKFTracker.predict() の戻り値。
        scale : (scale_x, scale_y)
            座標スケール。
        """
        sx, sy = scale
        for hand_preds in predictions:
            for tip_idx, pred in enumerate(hand_preds):
                if pred is None:
                    continue
                px, py = int(pred[0] * sx), int(pred[1] * sy)
                color  = FINGERTIP_COLORS[tip_idx]
                cv2.circle(frame_bgr, (px, py), 8, color, 2)          # 中空円
                cv2.putText(
                    frame_bgr,
                    FINGERTIP_NAMES[tip_idx][0],                       # 先頭文字ラベル
                    (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

    # ── リソース解放 ──────────────────────────────
    def close(self) -> None:
        """MediaPipe リソースを解放する。終了時に呼び出す。"""
        self._hands.close()
