import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import japanize_matplotlib

# ページ設定
st.set_page_config(page_title="AI歩行分析", page_icon="🚶")

st.title("🚶 AI歩行分析システム (骨格表示版)")
st.markdown("動画をアップロードすると、**AIが骨格を検出し**、膝の角度などを解析します。")

# MediaPipeの準備
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ファイルアップロード
uploaded_file = st.file_uploader("歩行動画を選択してください", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # 一時ファイルとして保存（OpenCVで読み込むため）
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # 動画情報の取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 解析用変数の準備
    knee_angles = []
    trunk_angles = []
    frames = []
    
    # 結果動画の保存準備
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # コーデック
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    st.info("解析と動画生成を開始します...（少し時間がかかります）")
    progress_bar = st.progress(0)
    
    # Pose推定の開始
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 色変換 BGR->RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 推定
            results = pose.process(image)
            
            # 描画のために色を戻す RGB->BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                # ★ここで骨格を描画しています★
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                
                # ランドマークの取得
                landmarks = results.pose_landmarks.landmark
                
                # 左側の座標取得（簡易的に左側のみ）
                # 23:左腰, 25:左膝, 27:左足首, 11:左肩
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                
                # 角度計算関数
                def calculate_angle(a, b, c):
                    a = np.array(a) # First
                    b = np.array(b) # Mid
                    c = np.array(c) # End
                    
                    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                    angle = np.abs(radians*180.0/np.pi)
                    
                    if angle > 180.0:
                        angle = 360-angle
                    return angle

                # 膝角度
                knee_angle = calculate_angle(hip, knee, ankle)
                knee_angles.append(knee_angle)
                
                # 体幹前傾（垂直線との角度）
                vertical_ref = [hip[0], hip[1] - 0.5] # 腰の真上
                trunk_angle = calculate_angle(vertical_ref, hip, shoulder)
                trunk_angles.append(trunk_angle)
                
            else:
                knee_angles.append(np.nan)
                trunk_angles.append(np.nan)

            # 加工したフレームを動画に書き込み
            out.write(image)
            
            frame_count += 1
            frames.append(frame_count)
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()
    
    st.success("解析完了！")
    
    # --- 結果の表示 ---
    
    # 1. 生成された動画を表示
    st.subheader("骨格検知動画")
    st.video(output_path)
    
    # 2. グラフを表示
    st.subheader("【矢状面】動作解析グラフ")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.set_xlabel('フレーム数')
    ax1.set_ylabel('膝角度 (deg)', color='blue')
    ax1.plot(frames, knee_angles, color='blue', label='膝角度')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('体幹前傾 (deg)', color='red')
    ax2.plot(frames, trunk_angles, color='red', linestyle='--', label='体幹前傾')
    ax2.tick_params(axis='y', labelcolor='red')
    
    st.pyplot(fig)
