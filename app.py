import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tempfile
import os
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from typing import Dict, List, Optional, Union
import warnings

# ==========================================================
# 1. GaitMathCore クラス（数学的計算基盤）
# ==========================================================
class GaitMathCore:
    def __init__(self, fps: int = 60):
        self.fps = fps
        self.VISIBILITY_THRESHOLD = 0.5
        self.SAVGOL_WINDOW = 5
        self.SAVGOL_POLYORDER = 2

    @staticmethod
    def calculate_angle_3d(p1, p2, p3, use_z_axis=False):
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 < 1e-6 or norm_v2 < 1e-6: return None
        cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def preprocess_landmark_timeseries(self, df, coord_columns, apply_filter=True):
        df_processed = df.copy()
        time_points = np.arange(len(df_processed))
        for col in coord_columns:
            data = df_processed[col].values
            valid_idx = df_processed['visibility'] >= self.VISIBILITY_THRESHOLD
            if valid_idx.sum() < 3: continue
            interpolated = np.interp(time_points, time_points[valid_idx], data[valid_idx])
            if apply_filter and len(interpolated) >= self.SAVGOL_WINDOW:
                interpolated = savgol_filter(interpolated, self.SAVGOL_WINDOW, self.SAVGOL_POLYORDER)
            df_processed[col] = interpolated
        return df_processed

# ==========================================================
# 2. GaitLandmarkExtractor クラス（姿勢抽出）
# ==========================================================
class GaitLandmarkExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # iPhoneでの動作安定のため一旦1に設定
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        landmarks_list = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    name = self.mp_pose.PoseLandmark(idx).name
                    landmarks_list.append({
                        'frame': frame_count,
                        'landmark': name,
                        'x': landmark.x, 'y': landmark.y, 'z': landmark.z,
                        'visibility': landmark.visibility
                    })
            frame_count += 1
        cap.release()
        return pd.DataFrame(landmarks_list)

# ==========================================================
# 3. Streamlit UI 部分（iPhoneに表示される画面）
# ==========================================================
st.set_page_config(page_title="AI歩行ドック", layout="centered")

st.title("🏃‍♀️ AI歩行ドック")
st.write("理学療法士の知見 × 高精度AI分析")

uploaded_file = st.file_uploader("歩行動画をアップロード", type=["mp4", "mov"])

if uploaded_file:
    st.video(uploaded_file)
    if st.button("分析を開始する", use_container_width=True):
        with st.spinner("最新のエンジンで解析中..."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            extractor = GaitLandmarkExtractor()
            raw_df = extractor.extract_from_video(tfile.name)
            
            if not raw_df.empty:
                st.success("解析成功！")
                st.write("座標データの取得が完了しました。")
                st.dataframe(raw_df.head(20))
            else:
                st.error("姿勢を検出できませんでした。全身が写るように撮り直してください。")
            
            os.unlink(tfile.name)

st.divider()
st.caption("Phase 1: 数学的基盤エンジン稼働中")
