"""
csv_logger.py
-------------
検出結果と KF 結果をフレーム単位で CSV に書き出す。
"""

import csv
import json
from typing import Any, List


class PositionCSVLogger:
    """フレームごとの位置データを記録する CSV ロガー。"""

    HEADER = [
        "frame_index",
        "mode",
        "svm_label",
        "svm_conf",
        "left_hand_source",
        "left_hand_tips_json",
        "left_hand_kf_tips_json",
        "left_wrist_json",
        "left_joints_json",
        "right_hand_source",
        "right_hand_tips_json",
        "right_hand_kf_tips_json",
        "right_wrist_json",
        "right_joints_json",
        "hand_count",
        "object_source",
        "object_count",
        "objects_json",
        "object_kf_tracks_json",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    @staticmethod
    def _to_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    def write_row(
        self,
        frame_index: int,
        mode: str,
        svm_label: str,
        svm_conf: float,
        left_hand_source: str,
        left_hand_tips: List[Any],
        left_hand_kf_tips: List[Any],
        left_wrist: List[Any],
        left_joints: List[Any],
        right_hand_source: str,
        right_hand_tips: List[Any],
        right_hand_kf_tips: List[Any],
        right_wrist: List[Any],
        right_joints: List[Any],
        hand_count: int,
        object_source: str,
        object_count: int,
        objects: List[Any],
        object_kf_tracks: List[Any],
    ):
        self._writer.writerow(
            [
                frame_index,
                mode,
                svm_label,
                svm_conf,
                left_hand_source,
                self._to_json(left_hand_tips),
                self._to_json(left_hand_kf_tips),
                self._to_json(left_wrist),
                self._to_json(left_joints),
                right_hand_source,
                self._to_json(right_hand_tips),
                self._to_json(right_hand_kf_tips),
                self._to_json(right_wrist),
                self._to_json(right_joints),
                hand_count,
                object_source,
                object_count,
                self._to_json(objects),
                self._to_json(object_kf_tracks),
            ]
        )

    def close(self):
        self._file.close()
