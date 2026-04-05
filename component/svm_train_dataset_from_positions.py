import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


Point = Tuple[float, float]
FINGER_NAMES = ("thumb", "index", "middle")


@dataclass
class FrameSample:
    """単一フレームのサンプル: 左右手 3 指と現在の物体情報を保持。"""

    label: Optional[str]
    frame_index: int
    left_fingers: Dict[str, Optional[Point]]
    right_fingers: Dict[str, Optional[Point]]
    object_center: Optional[Point]
    object_tracks: Dict[int, Point]


@dataclass
class FeatureState:
    """時系列特徴量計算に必要な前フレーム状態。"""

    prev_left_fingers: Dict[str, Optional[Point]]
    prev_right_fingers: Dict[str, Optional[Point]]
    prev_left_angles: Dict[str, Optional[float]]
    prev_right_angles: Dict[str, Optional[float]]
    prev_object_center: Optional[Point]
    prev_object_tracks: Dict[int, Point]


# ─────────────────────────────────────────────
# 基本パース関数
# ─────────────────────────────────────────────
def _safe_json_loads(raw: str, default: Any):
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _normalize_points(points: Any) -> List[Optional[Point]]:
    if not isinstance(points, list):
        return []

    normalized: List[Optional[Point]] = []
    for item in points:
        if item is None:
            normalized.append(None)
            continue
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            normalized.append(None)
            continue
        try:
            x = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError):
            normalized.append(None)
            continue
        normalized.append((x, y))
    return normalized


def _pick_hand_points(row: Dict[str, str], hand_side: str) -> List[Optional[Point]]:
    """
    片手の 5 指先を取得する。
    検出値を優先し、検出が空なら KF 値へフォールバックする。
    """
    if hand_side == "left":
        det_key = "left_hand_tips_json"
        kf_key = "left_hand_kf_tips_json"
    else:
        det_key = "right_hand_tips_json"
        kf_key = "right_hand_kf_tips_json"

    det_points = _normalize_points(_safe_json_loads(row.get(det_key, ""), []))
    kf_points = _normalize_points(_safe_json_loads(row.get(kf_key, ""), []))

    if any(p is not None for p in det_points):
        points = det_points
    else:
        points = kf_points

    if len(points) < 5:
        points = points + [None] * (5 - len(points))
    else:
        points = points[:5]

    return points


def _extract_object_center(row: Dict[str, str]) -> Optional[Point]:
    """
    現フレームの主物体中心を取得する。
    objects_json の center を優先し、無ければ object_kf_tracks_json の pred_center にフォールバック。
    """
    objects = _safe_json_loads(row.get("objects_json", ""), [])
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            center = obj.get("center")
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            try:
                return float(center[0]), float(center[1])
            except (TypeError, ValueError):
                continue

    track_map = _extract_track_centers(row)
    if track_map:
        first_tid = sorted(track_map.keys())[0]
        return track_map[first_tid]

    return None


def _extract_track_centers(row: Dict[str, str]) -> Dict[int, Point]:
    """
    object_kf_tracks_json から track_id -> pred_center を抽出する。
    「対応物体間のフレーム間距離変化」特徴量に使用する。
    """
    tracks = _safe_json_loads(row.get("object_kf_tracks_json", ""), [])
    result: Dict[int, Point] = {}

    if not isinstance(tracks, list):
        return result

    for item in tracks:
        if not isinstance(item, dict):
            continue
        track_id = item.get("track_id")
        center = item.get("pred_center")
        if not isinstance(track_id, int):
            continue
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        try:
            result[track_id] = (float(center[0]), float(center[1]))
        except (TypeError, ValueError):
            continue

    return result


def _to_three_fingers(points5: List[Optional[Point]]) -> Dict[str, Optional[Point]]:
    """5 指先から親指/人差し指/中指（0,1,2）を抽出する。"""
    return {
        "thumb": points5[0] if len(points5) > 0 else None,
        "index": points5[1] if len(points5) > 1 else None,
        "middle": points5[2] if len(points5) > 2 else None,
    }


def iter_frame_samples(csv_path: str) -> Iterable[FrameSample]:
    """positions.csv をフレームごとに読み、構造化サンプルを生成する。"""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "frame_index",
            "left_hand_tips_json",
            "left_hand_kf_tips_json",
            "right_hand_tips_json",
            "right_hand_kf_tips_json",
            "objects_json",
            "object_kf_tracks_json",
        }
        missing = [c for c in required_columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        for row in reader:
            try:
                frame_index = int(row.get("frame_index", "-1"))
            except ValueError:
                continue

            left_points5 = _pick_hand_points(row, "left")
            right_points5 = _pick_hand_points(row, "right")

            yield FrameSample(
                label=(str(row.get("label")).strip() if row.get("label") is not None and str(row.get("label")).strip() != "" else None),
                frame_index=frame_index,
                left_fingers=_to_three_fingers(left_points5),
                right_fingers=_to_three_fingers(right_points5),
                object_center=_extract_object_center(row),
                object_tracks=_extract_track_centers(row),
            )


# ─────────────────────────────────────────────
# 特徴量エンジニアリング関数
# ─────────────────────────────────────────────
def _point_distance(p1: Optional[Point], p2: Optional[Point]) -> float:
    if p1 is None or p2 is None:
        return 0.0
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _wrap_angle_delta(curr: Optional[float], prev: Optional[float]) -> float:
    if curr is None or prev is None:
        return 0.0
    delta = curr - prev
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _compute_finger_angles(fingers: Dict[str, Optional[Point]]) -> Dict[str, Optional[float]]:
    """
    指定された方法に基づく:
    1) 3 指から中心 Cx, Cy を算出
    2) 各指角度 theta_i = atan2(y_i - Cy, x_i - Cx)

    3 指のいずれかが欠損している場合、このフレーム角度は None とする。
    """
    thumb = fingers.get("thumb")
    index = fingers.get("index")
    middle = fingers.get("middle")

    if thumb is None or index is None or middle is None:
        return {"thumb": None, "index": None, "middle": None}

    cx = (thumb[0] + index[0] + middle[0]) / 3.0
    cy = (thumb[1] + index[1] + middle[1]) / 3.0

    return {
        "thumb": math.atan2(thumb[1] - cy, thumb[0] - cx),
        "index": math.atan2(index[1] - cy, index[0] - cx),
        "middle": math.atan2(middle[1] - cy, middle[0] - cx),
    }


def feature_columns() -> List[str]:
    """出力列を固定し、追記時も列順を一定に保つ。"""
    cols: List[str] = []

    # 1) 左右手 3 指: 前フレームに対する移動距離
    for side in ("left", "right"):
        for finger in FINGER_NAMES:
            cols.append(f"{side}_{finger}_delta_prev")

    # 2) 左右手 3 指から物体中心までの距離
    for side in ("left", "right"):
        for finger in FINGER_NAMES:
            cols.append(f"{side}_{finger}_to_object")

    # 3) 左右手 3 指の回転角増分 + 平均角増分
    for side in ("left", "right"):
        for finger in FINGER_NAMES:
            cols.append(f"{side}_{finger}_angle_delta")
        cols.append(f"{side}_angle_delta_avg")

    # 4) 対応物体（同一 track_id）のフレーム間移動量
    cols.extend([
        "object_track_motion_mean",
        "object_track_motion_max",
        "object_center_delta",
    ])

    return cols


def compute_feature_vector(sample: FrameSample, state: FeatureState) -> List[float]:
    """
    要件に従って特徴量を計算する:
    1) 左右手 3 指のフレーム間距離変化（同一指の前フレーム比）
    2) 左右手 3 指から現在物体中心までの距離
    3) 3 指の回転角増分（atan2 ベース）+ 平均増分
    4) 対応物体（同一 track_id）のフレーム間距離変化
    """
    feats: List[float] = []

    # 1) 3 指のフレーム間移動量（現指先 vs 前フレーム同指先）
    for side, curr, prev in (
        ("left", sample.left_fingers, state.prev_left_fingers),
        ("right", sample.right_fingers, state.prev_right_fingers),
    ):
        _ = side
        for finger in FINGER_NAMES:
            feats.append(_point_distance(curr.get(finger), prev.get(finger)))

    # 2) 3 指から物体中心までの距離
    obj_center = sample.object_center
    for side_fingers in (sample.left_fingers, sample.right_fingers):
        for finger in FINGER_NAMES:
            feats.append(_point_distance(side_fingers.get(finger), obj_center))

    # 3) 3 指の角度増分 + 平均角度増分
    left_angles = _compute_finger_angles(sample.left_fingers)
    right_angles = _compute_finger_angles(sample.right_fingers)

    for curr_angles, prev_angles in (
        (left_angles, state.prev_left_angles),
        (right_angles, state.prev_right_angles),
    ):
        deltas: List[float] = []
        for finger in FINGER_NAMES:
            dtheta = _wrap_angle_delta(curr_angles.get(finger), prev_angles.get(finger))
            feats.append(dtheta)
            deltas.append(dtheta)
        feats.append(sum(deltas) / 3.0)

    # 4) 同一 track_id 物体のフレーム間移動量
    common_track_ids = set(sample.object_tracks.keys()) & set(state.prev_object_tracks.keys())
    track_moves: List[float] = []
    for tid in common_track_ids:
        curr_p = sample.object_tracks[tid]
        prev_p = state.prev_object_tracks[tid]
        track_moves.append(_point_distance(curr_p, prev_p))

    if track_moves:
        feats.append(sum(track_moves) / len(track_moves))
        feats.append(max(track_moves))
    else:
        feats.append(0.0)
        feats.append(0.0)

    feats.append(_point_distance(sample.object_center, state.prev_object_center))

    return feats


# ─────────────────────────────────────────────
# データセット出力 + SVM 学習
# ─────────────────────────────────────────────
def _has_any_hand_point(sample: FrameSample) -> bool:
    for finger in FINGER_NAMES:
        if sample.left_fingers.get(finger) is not None:
            return True
        if sample.right_fingers.get(finger) is not None:
            return True
    return False


def export_svm_dataset(input_csv: str, output_csv: str, label: Optional[str] = None, overwrite: bool = False) -> int:
    """
    positions.csv を読み、SVM データセットを生成する。

    - 1 行 1 サンプル（フレーム単位）
    - 列形式: label, f0, f1, ...
    - 「少なくとも片手あり + 物体中心あり + ラベルあり」の行のみ書き出す
    - ラベルは CSV の label 列を優先し、無ければ関数引数 label を使用
    """
    if overwrite and os.path.exists(output_csv):
        os.remove(output_csv)

    feature_names = feature_columns()
    rows_written = 0
    header_written = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0

    state = FeatureState(
        prev_left_fingers={k: None for k in FINGER_NAMES},
        prev_right_fingers={k: None for k in FINGER_NAMES},
        prev_left_angles={k: None for k in FINGER_NAMES},
        prev_right_angles={k: None for k in FINGER_NAMES},
        prev_object_center=None,
        prev_object_tracks={},
    )

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not header_written:
            writer.writerow(["label"] + [f"f{i}" for i in range(len(feature_names))])
            header_written = True

        for sample in iter_frame_samples(input_csv):
            current_left_angles = _compute_finger_angles(sample.left_fingers)
            current_right_angles = _compute_finger_angles(sample.right_fingers)

            row_label = sample.label if sample.label is not None else label
            can_export = (
                sample.object_center is not None
                and _has_any_hand_point(sample)
                and row_label is not None
            )
            if can_export:
                features = compute_feature_vector(sample, state)
                writer.writerow([row_label] + features)
                rows_written += 1

            # 常に状態を更新し、「前フレーム」意味を正しく保つ
            state.prev_left_fingers = dict(sample.left_fingers)
            state.prev_right_fingers = dict(sample.right_fingers)
            state.prev_left_angles = current_left_angles
            state.prev_right_angles = current_right_angles
            state.prev_object_center = sample.object_center
            state.prev_object_tracks = dict(sample.object_tracks)

    return rows_written


def load_dataset(csv_path: str):
    """label,f0..fN 形式のデータセットを読み込む。"""
    import numpy as np

    labels: List[str] = []
    features: List[List[float]] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        feature_cols = [c for c in (reader.fieldnames or []) if c.startswith("f")]
        if not feature_cols:
            raise ValueError("Dataset has no feature columns f0..fN")

        for row in reader:
            labels.append(str(row["label"]))
            features.append([float(row[c]) for c in feature_cols])

    if not labels:
        raise ValueError("Dataset is empty")

    return np.asarray(features, dtype=np.float32), np.asarray(labels), feature_cols


def train_svm_model(
    dataset_csv: str,
    model_out: str,
    test_size: float = 0.2,
    random_state: int = 42,
    kernel: str = "rbf",
    c_value: float = 5.0,
    gamma: str = "scale",
):
    """sklearn を用いて SVM を学習し、モデルを保存する。"""
    import joblib
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    x, y, feature_cols = load_dataset(dataset_csv)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes to train SVM.")

    stratify = y if (test_size > 0 and np.min(counts) >= 2) else None

    if test_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    else:
        x_train, y_train = x, y
        x_test, y_test = None, None

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=kernel, C=c_value, gamma=gamma, probability=True)),
        ]
    )
    model.fit(x_train, y_train)

    metrics = {}
    if x_test is not None:
        y_pred = model.predict(x_test)
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["report"] = classification_report(y_test, y_pred, zero_division=0)

    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "classes": model.named_steps["svm"].classes_.tolist(),
    }
    joblib.dump(artifact, model_out)

    return {
        "classes": classes.tolist(),
        "counts": counts.tolist(),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Build SVM dataset from positions CSV, and optionally train SVM")
    parser.add_argument("--input", type=str, required=True, help="Path to *_positions.csv")
    parser.add_argument("--output", type=str, default="svm_dataset_from_positions.csv", help="Output SVM CSV path")
    parser.add_argument("--label", type=str, default=None, help="デフォルトラベル（CSV 行に label 列が無い場合のみ使用）")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV before export")

    parser.add_argument("--train", action="store_true", help="Train SVM right after exporting dataset")
    parser.add_argument("--modelOut", type=str, default="svm_model_from_positions.joblib", help="Output SVM model path")
    parser.add_argument("--testSize", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--randomState", type=int, default=42)
    parser.add_argument("--kernel", type=str, default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--C", type=float, default=5.0)
    parser.add_argument("--gamma", type=str, default="scale")
    args = parser.parse_args()

    rows = export_svm_dataset(
        input_csv=args.input,
        output_csv=args.output,
        label=args.label,
        overwrite=args.overwrite,
    )
    print(f"Done. wrote {rows} samples to: {args.output}")

    if args.train:
        result = train_svm_model(
            dataset_csv=args.output,
            model_out=args.modelOut,
            test_size=args.testSize,
            random_state=args.randomState,
            kernel=args.kernel,
            c_value=args.C,
            gamma=args.gamma,
        )
        print("Class distribution:")
        for cls, cnt in zip(result["classes"], result["counts"]):
            print(f"  {cls}: {cnt}")

        metrics = result.get("metrics", {})
        if "accuracy" in metrics:
            print(f"Validation accuracy: {metrics['accuracy']:.4f}")
            print(metrics["report"])
        print(f"Saved model to: {args.modelOut}")


if __name__ == "__main__":
    main()
