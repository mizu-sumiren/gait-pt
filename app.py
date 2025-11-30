import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import japanize_matplotlib

# ページ設定
st.set_page_config(page_title="AI歩行分析", page_icon="🚶")

st.title("🚶 AI歩行分析システム")
st.write("動画をアップロードすると、AIが「膝の角度」と「体幹の前傾」を解析します。")

# 動画アップロード機能
uploaded_file = st.file_uploader("歩行動画を選択してください", type=['mp4', 'mov'])

# --- 解析ロジック ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

if uploaded_file is not None:
    # 一時ファイルとして保存（OpenCVで読むため）
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    st.info("解析を開始します...少々お待ちください。")
    progress_bar = st.progress(0)
    
    mp_pose = mp.solutions.pose
    angle_log = []
    frame_idx = 0
    
    # 総フレーム数（進捗バー用）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 画像処理
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            h, w, _ = frame.shape
            
            # データ取得
            knee_angle = np.nan
            trunk_lean = np.nan
            pelvis_tilt = np.nan

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    # 座標
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
                    l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]

                    # 計算
                    knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                    
                    trunk_vec = np.array(l_sh) - np.array(l_hip)
                    vertical_vec = np.array([0, -1])
                    trunk_u = trunk_vec / np.linalg.norm(trunk_vec)
                    cos_theta = np.dot(trunk_u, vertical_vec)
                    trunk_lean = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    
                    pelvis_vec = np.array(l_hip) - np.array(r_hip)
                    pelvis_tilt = np.degrees(np.arctan2(pelvis_vec[1], pelvis_vec[0]))

                except:
                    pass
            
            angle_log.append([frame_idx, knee_angle, trunk_lean, pelvis_tilt])
            frame_idx += 1
            
            # 進捗更新
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    
    # --- 結果表示 ---
    st.success("解析完了！")
    
    if len(angle_log) > 0:
        df = pd.DataFrame(angle_log, columns=['Frame', 'KneeAngle', 'TrunkLean', 'PelvisTilt'])
        df_smooth = df.rolling(window=5, min_periods=1).mean()

        # グラフ描画
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 矢状面
        ax1.plot(df_smooth['Frame'], df_smooth['KneeAngle'], label='膝角度', color='blue')
        ax1.set_ylabel('膝角度 (deg)', color='blue')
        ax1.set_ylim(0, 180)
        ax1_r = ax1.twinx()
        ax1_r.plot(df_smooth['Frame'], df_smooth['TrunkLean'], label='体幹前傾', color='red', linestyle='--')
        ax1_r.set_ylabel('体幹前傾 (deg)', color='red')
        ax1.set_title('【矢状面】動作解析', fontsize=14)
        ax1.grid(True)
        
        # 前額面
        ax2.plot(df_smooth['Frame'], df_smooth['PelvisTilt'], label='骨盤の傾き', color='green')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_ylabel('傾き (deg)', fontsize=12)
        ax2.set_title('【前額面】左右バランス', fontsize=14)
        ax2.legend()
        ax2.grid(True)
        
        st.pyplot(fig)
        
        # 簡易コメント
        max_lean = df_smooth['TrunkLean'].max()
        st.markdown(f"### 📊 AI診断結果")
        st.write(f"- 最大体幹前傾: **{max_lean:.1f}度**")
        if max_lean > 20:
            st.error("⚠️ 前傾が強く、腰部への負担が懸念されます。")
        else:
            st.success("✅ 姿勢は概ね良好です。")

