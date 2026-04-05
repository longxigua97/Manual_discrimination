"""
メイン処理: 指検出（MediaPipe）と物体検出（YOLOv26n）を同時実行し、
非検出フレームは Kalman Filter で位置を補完する。

フレーム割り当て戦略（10 fps）:
    奇数フレーム (1, 3, 5 …) -> MediaPipe 指検出
    偶数フレーム (0, 2, 4 …) -> YOLOv26n 物体検出

各フレームで両モジュールの最新/予測結果を重ね描画する。
"""

import argparse
import os
import sys
import time
from collections import deque
from functools import lru_cache
from typing import Dict, Optional, Tuple

import cv2
import joblib
import numpy as np


from PIL import Image, ImageDraw, ImageFont


# ── 自作モジュール ──────────────────────────────────
from component.hand_detector   import HandDetector, FINGERTIP_IDS
from component.object_detector import ObjectDetector
from component.kf_tracker      import HandKFTracker, ObjectKFTracker
from tools.csv_logger      import PositionCSVLogger
from component.svm_train_dataset_from_positions import (
    FINGER_NAMES,
    FeatureState,
    FrameSample,
    compute_feature_vector,
)


WRIST_ID = 0
JOINT_IDS = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)

JP_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKJP-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.otf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)

ACTION_SCREW_TIGHTEN = "ネジ締め"
ACTION_VISUAL_CHECK = "外観チェック"


def try_open_picamera2(width: int, height: int, fps: int):
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
        )
        cam.configure(config)
        cam.video_configuration.controls.FrameRate = float(fps)
        cam.start()
        return cam
    except Exception as e:
        print(f"[Camera] Picamera2 は利用不可: {e}")
        return None


def resolve_display(no_display: bool, display_env: str) -> bool:
    if no_display:
        return False
    if display_env:
        os.environ["DISPLAY"] = display_env

    is_ssh     = bool(os.environ.get("SSH_CONNECTION"))
    has_local  = os.path.exists("/tmp/.X11-unix/X0")
    cur_disp   = os.environ.get("DISPLAY", "")

    if is_ssh and cur_disp.startswith("localhost:") and has_local:
        os.environ["DISPLAY"] = ":0"
        os.environ.setdefault("XAUTHORITY", os.path.expanduser("~/.Xauthority"))

    return bool(os.environ.get("DISPLAY", ""))


@lru_cache(maxsize=1)
def _resolve_jp_font_path() -> str:
    for path in JP_FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return ""


@lru_cache(maxsize=4)
def _load_jp_font(size: int):
    if ImageFont is None:
        return None
    font_path = _resolve_jp_font_path()
    if not font_path:
        return None
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return None


def _put_unicode_text_pil(frame, text: str, x: int, y: int, color_bgr=(0, 255, 255), font_size: int = 28) -> bool:
    """Pillow で Unicode 文字列を描画し、成功時は True を返す。"""
    if not text or Image is None or ImageDraw is None:
        return False
    font = _load_jp_font(font_size)
    if font is None:
        return False

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text((x, y), text, font=font, fill=color_rgb)
    frame[:, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return True


def draw_hud(frame, fps: float, frame_idx: int, mode: str, action: str = "", conf: float = 0.0):
    """左上に FPS と現在の検出モードを描画する。"""
    cv2.putText(frame, f"FPS: {fps:.1f}",    (16, 56),  cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255,   0), 3)
    cv2.putText(frame, f"Mode: {mode}",       (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 3)
    cv2.putText(frame, f"Frame: {frame_idx}", (16, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    if action:
        action_text = f"Action: {action}"
        drawn = _put_unicode_text_pil(frame, action_text, 16, 192, color_bgr=(0, 255, 255), font_size=56)
        if not drawn:
            cv2.putText(frame, action_text, (16, 224), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        cv2.putText(frame, f"Conf: {conf:.2f}", (16, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 3)


def _symbol_action_text(label: str) -> str:
    """SVM ラベルを Action 表示文字列へ変換する。該当なしは空文字列。"""
    raw = str(label).strip()
    if not raw:
        return ""

    try:
        label_id = int(float(raw))
    except ValueError:
        return ""

    if label_id == 2:
        return ACTION_SCREW_TIGHTEN
    if label_id == 3:
        return ACTION_VISUAL_CHECK
    return ""


def _vote_action_text(window) -> Tuple[str, float]:
    """5 フレームのスライディング投票: 同一動作が 3 回以上で出力。"""
    screw_confs = [conf for action, conf in window if action == ACTION_SCREW_TIGHTEN]
    visual_confs = [conf for action, conf in window if action == ACTION_VISUAL_CHECK]

    screw_hits = len(screw_confs)
    visual_hits = len(visual_confs)

    if screw_hits >= 3 and screw_hits >= visual_hits:
        return ACTION_SCREW_TIGHTEN, float(sum(screw_confs) / max(screw_hits, 1))
    if visual_hits >= 3 and visual_hits > screw_hits:
        return ACTION_VISUAL_CHECK, float(sum(visual_confs) / max(visual_hits, 1))
    return "", 0.0


def _to_point(value) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _build_three_fingers(hand_tips, hand_kf_tips) -> Dict[str, Optional[Tuple[float, float]]]:
    det_points = [_to_point(p) for p in (hand_tips or [])]
    kf_points = [_to_point(p) for p in (hand_kf_tips or [])]

    source = det_points if any(p is not None for p in det_points) else kf_points
    if len(source) < 5:
        source = source + [None] * (5 - len(source))

    return {
        "thumb": source[0],
        "index": source[1],
        "middle": source[2],
    }


def _compute_angles(fingers: Dict[str, Optional[Tuple[float, float]]]) -> Dict[str, Optional[float]]:
    thumb = fingers.get("thumb")
    index = fingers.get("index")
    middle = fingers.get("middle")
    if thumb is None or index is None or middle is None:
        return {"thumb": None, "index": None, "middle": None}

    cx = (thumb[0] + index[0] + middle[0]) / 3.0
    cy = (thumb[1] + index[1] + middle[1]) / 3.0

    return {
        "thumb": float(np.arctan2(thumb[1] - cy, thumb[0] - cx)),
        "index": float(np.arctan2(index[1] - cy, index[0] - cx)),
        "middle": float(np.arctan2(middle[1] - cy, middle[0] - cx)),
    }


def _has_any_hand_point(fingers: Dict[str, Optional[Tuple[float, float]]]) -> bool:
    return any(fingers.get(k) is not None for k in FINGER_NAMES)


def _extract_object_center_for_svm(is_odd_frame: bool, detections, object_tracks) -> Optional[Tuple[float, float]]:
    if (not is_odd_frame) and detections:
        center = detections[0].get("center")
        point = _to_point(center)
        if point is not None:
            return point

    if object_tracks:
        ordered = sorted(object_tracks, key=lambda t: t.get("track_id", 1e9))
        center = ordered[0].get("pred_center")
        return _to_point(center)

    return None


def _extract_track_map_for_svm(object_tracks) -> Dict[int, Tuple[float, float]]:
    track_map: Dict[int, Tuple[float, float]] = {}
    for t in object_tracks:
        tid = t.get("track_id")
        center = _to_point(t.get("pred_center"))
        if isinstance(tid, int) and center is not None:
            track_map[tid] = center
    return track_map


def extract_left_right_hand_points(hand_result, frame_w: int, frame_h: int):
    """MediaPipe 結果から左右手の手首・関節・指先座標を抽出する。"""
    left_tips = []
    right_tips = []
    left_wrist = []
    right_wrist = []
    left_joints = []
    right_joints = []

    if not hand_result or not hand_result.multi_hand_landmarks:
        return left_tips, right_tips, left_wrist, right_wrist, left_joints, right_joints

    handedness_list = getattr(hand_result, "multi_handedness", []) or []
    for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
        tips = []
        for tip_id in FINGERTIP_IDS:
            lm = hand_landmarks.landmark[tip_id]
            tips.append((int(lm.x * frame_w), int(lm.y * frame_h)))

        wrist_lm = hand_landmarks.landmark[WRIST_ID]
        wrist = [int(wrist_lm.x * frame_w), int(wrist_lm.y * frame_h)]

        joints = []
        for joint_id in JOINT_IDS:
            lm = hand_landmarks.landmark[joint_id]
            joints.append((int(lm.x * frame_w), int(lm.y * frame_h)))

        hand_label = ""
        if idx < len(handedness_list) and handedness_list[idx].classification:
            hand_label = handedness_list[idx].classification[0].label.lower()

        if hand_label == "left":
            left_tips = tips
            left_wrist = wrist
            left_joints = joints
        elif hand_label == "right":
            right_tips = tips
            right_wrist = wrist
            right_joints = joints
        else:
            if not left_tips:
                left_tips = tips
                left_wrist = wrist
                left_joints = joints
            elif not right_tips:
                right_tips = tips
                right_wrist = wrist
                right_joints = joints

    return left_tips, right_tips, left_wrist, right_wrist, left_joints, right_joints


# ═══════════════════════════════════════════════
# メイン関数
# ═══════════════════════════════════════════════

def main():
    # ── 引数解析 ──────────────────────────────
    parser = argparse.ArgumentParser(description="指 + 物体の同期検出（MP4 またはカメラ入力、KF 予測補完）")
    parser.add_argument("--input",          type=str,   default="",    help="入力 MP4 パス（未指定時はカメラ入力）")
    parser.add_argument("--output",         type=str,   default="",    help="出力 MP4 パス（任意）")
    parser.add_argument("--fps",            type=int,   default=10,    help="表示のフォールバック用目標 FPS")
    parser.add_argument("--width",          type=int,   default=640,  help="処理/出力フレーム幅")
    parser.add_argument("--height",         type=int,   default=480,   help="処理/出力フレーム高さ")
    parser.add_argument("--maxHands",       type=int,   default=2,     help="追跡する最大手数")
    parser.add_argument("--minDetection",   type=float, default=0.5,   help="MediaPipe 検出しきい値")
    parser.add_argument("--minTracking",    type=float, default=0.5,   help="MediaPipe 追跡しきい値")
    parser.add_argument("--yoloModel",      type=str,   default="yoloweight/yolov26n-ncnn.pt", help="YOLOv26n モデルパス")
    parser.add_argument("--yoloConf",       type=float, default=0.5,   help="YOLOv26n 信頼度しきい値")
    parser.add_argument("--noDisplay",      action="store_true",       help="OpenCV ウィンドウを表示しない（ヘッドレス）")
    parser.add_argument("--display",        type=str,   default=":0",    help="DISPLAY を強制指定（例: :0）")
    parser.add_argument("--csv",            type=str,   default="",    help="フレーム位置 CSV 出力パス（既定: 入力名_positions.csv）")
    parser.add_argument("--svmModel",       type=str,   default="svmweight/svm_model_from_positions.joblib",    help="SVM モデルパス")
    parser.add_argument("--svmMinProb",     type=float, default=0.5,   help="SVM 出力の最小信頼度しきい値")
    args = parser.parse_args()

    # ── 表示ウィンドウ設定 ───────────────────────────
    show_window = resolve_display(args.noDisplay, args.display)
    if show_window:
        print(f"[Display] 表示, DISPLAY={os.environ.get('DISPLAY')}")
    else:
        print("[Display] 非表示モード。動画処理完了後に終了します。")

    # 処理サイズ
    target_w = max(64, int(args.width))
    target_h = max(64, int(args.height))

    # ── 入力ソース初期化（MP4 / Picamera2） ─────────────────────
    cap = None
    picam = None
    using_picamera2 = False
    is_file_input = bool(args.input)

    if is_file_input:
        if not os.path.exists(args.input):
            sys.exit(f"[Error] 入力ファイルが存在しません: {args.input}")

        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            sys.exit(f"[Error] 入力動画を開けません: {args.input}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 0:
            src_fps = float(args.fps)

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if orig_w <= 0 or orig_h <= 0:
            orig_w, orig_h = target_w, target_h

        print(f"[Input] MP4 入力: {args.input}")
    else:
        picam = try_open_picamera2(target_w, target_h, args.fps)
        if picam is not None:
            using_picamera2 = True
            src_fps = float(args.fps)
            orig_w, orig_h = target_w, target_h
            print("[Input] Picamera2 を使用します。")
        else:
            sys.exit("[Error] Picamera2 カメラを開けません。")

    # ── 出力動画設定 ─────────────────────────
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (orig_w, orig_h))
        if not writer.isOpened():
            if cap is not None:
                cap.release()
            sys.exit(f"[Error] 出力動画を作成できません: {args.output}")

    # ── CSV 出力 ─────────────────────────────
    if args.csv:
        csv_path = args.csv
    else:
        if is_file_input:
            input_stem = os.path.splitext(os.path.basename(args.input))[0]
            csv_path = f"{input_stem}_positions.csv"
        else:
            csv_path = f"camera_{int(time.time())}_positions.csv"

    csv_logger = PositionCSVLogger(csv_path)

    # ── 検出器の初期化 ───────────────────────────
    print("[Init] MediaPipe Hands を読み込み中...")
    hand_detector = HandDetector(
        max_hands=args.maxHands,
        min_detection_confidence=args.minDetection,
        min_tracking_confidence=args.minTracking,
    )

    print("[Init] YOLOv26n を読み込み中")
    object_detector = ObjectDetector(
        model_path=args.yoloModel,
        conf_threshold=args.yoloConf,
        num_threads=4,
    )

    # ── Kalman Filter トラッカー初期化 ────────────
    hand_kf_tracker   = HandKFTracker(max_hands=max(2, args.maxHands))
    object_kf_tracker = ObjectKFTracker(max_distance=120, max_lost_frames=3)

    # ── SVM モデル読み込み ────────────────────
    svm_enabled = False
    svm_model = None
    svm_feature_columns = []
    svm_classes = []
    svm_state = None
    svm_action = ""
    svm_conf = 0.0
    svm_vote_window = deque(maxlen=5)
    svm_display_action = ""
    svm_display_conf = 0.0

    if args.svmModel:
        if not os.path.exists(args.svmModel):
            if cap is not None:
                cap.release()
            csv_logger.close()
            sys.exit(f"[Error] SVM モデルが存在しません: {args.svmModel}")
        try:
            artifact = joblib.load(args.svmModel)
            svm_model = artifact["model"]
            svm_feature_columns = artifact.get("feature_columns", [])
            svm_classes = artifact.get("classes", [])
            svm_state = FeatureState(
                prev_left_fingers={k: None for k in FINGER_NAMES},
                prev_right_fingers={k: None for k in FINGER_NAMES},
                prev_left_angles={k: None for k in FINGER_NAMES},
                prev_right_angles={k: None for k in FINGER_NAMES},
                prev_object_center=None,
                prev_object_tracks={},
            )
            svm_enabled = True
            print(f"[Init] SVM モデルを読み込みました: {args.svmModel}")
        except Exception as e:
            if cap is not None:
                cap.release()
            csv_logger.close()
            sys.exit(f"[Error] SVM モデルの読み込みに失敗: {e}")

    # ── 実行状態変数 ───────────────────────────
    frame_index       = 0
    prev_time         = time.time()
    last_hand_result  = None
    last_left_tips    = []
    last_right_tips   = []
    last_left_wrist   = []
    last_right_wrist  = []
    last_left_joints  = []
    last_right_joints = []
    last_detections   = []

    WINDOW_NAME = "Hand + Object Detection (Video)"

    print("[Main] 動画処理を開始します。q または ESC で早期終了できます... ")

    try:
        while True:
            if using_picamera2:
                frame_rgb_o = picam.capture_array()
                ok = frame_rgb_o is not None
                frame_bgr_o = cv2.cvtColor(frame_rgb_o, cv2.COLOR_RGB2BGR) if ok else None
            else:
                ok, frame_bgr_o = cap.read()

            if not ok:
                if is_file_input:
                    print("[Main] 動画の読み取りが終了しました。")
                else:
                    print("[Main] カメラフレーム取得に失敗したため終了します。")
                break

            # 元フレームサイズ取得（描画用）
            orig_frame_h, orig_frame_w = frame_bgr_o.shape[:2]
            # 検出計算用に target サイズへリサイズ
            frame_bgr = cv2.resize(frame_bgr_o, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            frame_h, frame_w = frame_bgr.shape[:2]
            # 座標スケール比（リサイズ画像 -> 元画像）
            scale = (orig_frame_w / target_w, orig_frame_h / target_h)

            # 各フレーム先頭: 先に KF predict を実行して予測を取得
            hand_predictions = hand_kf_tracker.predict()
            object_kf_tracker.predict_all()
            object_tracks = object_kf_tracker.active_tracks()

            # 奇数フレーム -> 指検出、偶数フレーム -> 物体検出
            is_odd_frame = (frame_index % 2 == 1)

            if is_odd_frame:
                mode = "HAND (MediaPipe)"
                hand_result = hand_detector.detect(frame_bgr)
                last_hand_result = hand_result

                left_tips, right_tips, left_wrist, right_wrist, left_joints, right_joints = extract_left_right_hand_points(hand_result, frame_w, frame_h)
                last_left_tips = left_tips
                last_right_tips = right_tips
                last_left_wrist = left_wrist
                last_right_wrist = right_wrist
                last_left_joints = left_joints
                last_right_joints = right_joints
                # 検出結果がある場合は KF を補正
                if left_tips or right_tips:
                    hand_kf_tracker.update([
                        left_tips if left_tips else None,
                        right_tips if right_tips else None,
                    ])

            else:
                mode = "OBJECT (YOLOv26n)"
                detections = object_detector.detect(frame_bgr)
                last_detections = detections
                # 検出結果がある場合は KF を補正
                object_kf_tracker.update(detections)
                # 更新後のトラック情報を再取得
                object_tracks = object_kf_tracker.active_tracks()

            # YOLO 検出フレームで「今フレーム未検出だが履歴に存在する」対象を抽出し KF 補償
            missing_object_tracks = []
            if not is_odd_frame:
                missing_object_tracks = [t for t in object_tracks if t.get("lost_count", 0) > 0]

            # 元画像上へ描画（座標スケール使用）
            if last_hand_result:
                hand_detector.draw(frame_bgr_o, last_hand_result, scale=scale)
            if not is_odd_frame:
                hand_detector.draw_predicted(frame_bgr_o, hand_predictions, scale=scale)

            # 検出フレームは検出ボックスと信頼度を表示。KF フレームは信頼度非表示
            if (not is_odd_frame) and last_detections:
                object_detector.draw(frame_bgr_o, last_detections, scale=scale)
            # YOLO で履歴対象を見失ったフレームでは、同フレームに KF 予測を重畳
            if (not is_odd_frame) and missing_object_tracks:
                object_detector.draw_predicted(frame_bgr_o, missing_object_tracks, scale=scale)
            if is_odd_frame:
                object_detector.draw_predicted(frame_bgr_o, object_tracks, scale=scale)

            if is_odd_frame:
                left_hand_source = "DETECTION" if last_left_tips else "NONE"
                right_hand_source = "DETECTION" if last_right_tips else "NONE"
                left_hand_output = last_left_tips
                right_hand_output = last_right_tips
                left_wrist_output = last_left_wrist
                right_wrist_output = last_right_wrist
                left_joints_output = last_left_joints
                right_joints_output = last_right_joints
            else:
                left_preds = hand_predictions[0] if len(hand_predictions) > 0 else []
                right_preds = hand_predictions[1] if len(hand_predictions) > 1 else []

                left_hand_output = [
                    [float(p[0]), float(p[1])] if p is not None else None
                    for p in left_preds
                ]
                right_hand_output = [
                    [float(p[0]), float(p[1])] if p is not None else None
                    for p in right_preds
                ]

                left_hand_source = "KF" if any(p is not None for p in left_preds) else "NONE"
                right_hand_source = "KF" if any(p is not None for p in right_preds) else "NONE"
                left_wrist_output = []
                right_wrist_output = []
                left_joints_output = []
                right_joints_output = []

            left_hand_has_data = bool(left_hand_output) and any(p is not None for p in left_hand_output)
            right_hand_has_data = bool(right_hand_output) and any(p is not None for p in right_hand_output)
            hand_count = int(left_hand_has_data) + int(right_hand_has_data)

            left_preds = hand_predictions[0] if len(hand_predictions) > 0 else []
            right_preds = hand_predictions[1] if len(hand_predictions) > 1 else []
            left_hand_kf_output = [
                [float(p[0]), float(p[1])] if p is not None else None
                for p in left_preds
            ]
            right_hand_kf_output = [
                [float(p[0]), float(p[1])] if p is not None else None
                for p in right_preds
            ]

            # SVM リアルタイム推論（時系列特徴）
            if svm_enabled and svm_model is not None and svm_state is not None:
                left_fingers = _build_three_fingers(left_hand_output, left_hand_kf_output)
                right_fingers = _build_three_fingers(right_hand_output, right_hand_kf_output)

                object_center_for_svm = _extract_object_center_for_svm(
                    is_odd_frame=is_odd_frame,
                    detections=last_detections,
                    object_tracks=object_tracks,
                )
                object_track_map = _extract_track_map_for_svm(object_tracks)

                sample = FrameSample(
                    label=None,
                    frame_index=frame_index,
                    left_fingers=left_fingers,
                    right_fingers=right_fingers,
                    object_center=object_center_for_svm,
                    object_tracks=object_track_map,
                )

                can_infer = (
                    object_center_for_svm is not None
                    and (_has_any_hand_point(left_fingers) or _has_any_hand_point(right_fingers))
                )

                if can_infer:
                    feat = compute_feature_vector(sample, svm_state)
                    if len(feat) == len(svm_feature_columns):
                        probs = svm_model.predict_proba([feat])[0]
                        best_idx = int(np.argmax(probs))
                        best_label = str(svm_classes[best_idx]) if svm_classes else str(best_idx)
                        best_prob = float(probs[best_idx])

                        if best_prob >= args.svmMinProb:
                            svm_action = best_label
                        else:
                            svm_action = "UNKNOWN"
                        svm_conf = best_prob
                    else:
                        svm_action = "FEATURE_MISMATCH"
                        svm_conf = 0.0
                else:
                    svm_action = "NO_FEATURE"
                    svm_conf = 0.0

                # 前フレーム状態を更新（時系列特徴が依存）
                svm_state.prev_left_fingers = dict(left_fingers)
                svm_state.prev_right_fingers = dict(right_fingers)
                svm_state.prev_left_angles = _compute_angles(left_fingers)
                svm_state.prev_right_angles = _compute_angles(right_fingers)
                svm_state.prev_object_center = object_center_for_svm
                svm_state.prev_object_tracks = dict(object_track_map)

                instant_action = _symbol_action_text(svm_action)
                if instant_action:
                    svm_vote_window.append((instant_action, svm_conf))
                else:
                    svm_vote_window.append(("", 0.0))

                svm_display_action, svm_display_conf = _vote_action_text(svm_vote_window)

            if is_odd_frame:
                object_output = object_tracks
                object_source = "KF" if object_tracks else "NONE"
            else:
                if last_detections:
                    object_output = last_detections
                    object_source = "DETECTION"
                elif object_tracks:
                    object_output = object_tracks
                    object_source = "KF"
                else:
                    object_output = []
                    object_source = "NONE"

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(
                frame_bgr_o,
                fps,
                frame_index,
                mode,
                action=svm_display_action if svm_enabled else "",
                conf=svm_display_conf if svm_enabled else 0.0,
            )

            csv_logger.write_row(
                frame_index=frame_index,
                mode=mode,
                svm_label=svm_action if svm_enabled else "",
                svm_conf=svm_conf if svm_enabled else 0.0,
                left_hand_source=left_hand_source,
                left_hand_tips=left_hand_output,
                left_hand_kf_tips=left_hand_kf_output,
                left_wrist=left_wrist_output,
                left_joints=left_joints_output,
                right_hand_source=right_hand_source,
                right_hand_tips=right_hand_output,
                right_hand_kf_tips=right_hand_kf_output,
                right_wrist=right_wrist_output,
                right_joints=right_joints_output,
                hand_count=hand_count,
                object_source=object_source,
                object_count=len(object_output),
                objects=object_output,
                object_kf_tracks=object_tracks,
            )

            if writer is not None:
                writer.write(frame_bgr_o)

            if show_window:
                cv2.imshow(WINDOW_NAME, frame_bgr_o)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_index += 1

    except KeyboardInterrupt:
        print("[Main] Ctrl+C で終了します。")

    # ── リソース解放 ──────────────────────────────
    hand_detector.close()
    if cap is not None:
        cap.release()
    if picam is not None:
        try:
            picam.stop()
        except Exception:
            pass
        try:
            picam.close()
        except Exception:
            pass
    if writer is not None:
        writer.release()
    csv_logger.close()
    if show_window:
        cv2.destroyAllWindows()

    print(f"[Main] 終了しました。総フレーム数: {frame_index}")
    if args.output:
        print(f"[Main] 出力ファイル: {args.output}")
    print(f"[Main] CSV ファイル: {csv_path}")


if __name__ == "__main__":
    main()
