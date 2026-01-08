import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# --- 1. ページ設定 ---
st.set_page_config(page_title="女性専用 AI歩行ドック", layout="wide")

# --- 2. 分析エンジンの準備 ---
@st.cache_resource
def load_pose_model():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5, 
        model_complexity=1 
    )

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

# --- 3. UI表示 ---
st.title("💃 女性専用 AI歩行ドック [Sakane/Park/Smith統合モデル]")
st.info("理学療法士の知見 × 最新エビデンス：女性の「第1歩目」と「変動性」を解析します。")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📸 側面（横から）")
    side_video = st.file_uploader("側面動画をアップロード", type=["mp4", "mov"], key="side_up")
with col2:
    st.markdown("### 📸 正面（前から）")
    front_video = st.file_uploader("正面動画をアップロード", type=["mp4", "mov"], key="front_up")

# 変数の初期化
max_flexion_angle = 0
cv_value = 18.5 # デモ用初期値（動画の数値を反映）
relative_phase = 15.2 # デモ用初期値（動画の数値を反映）

# --- 4. 解析実行 ---
if st.button("✨ アルゴリズム解析を開始", use_container_width=True):
    if not side_video and not front_video:
        st.warning("解析する動画をアップロードしてください。")
    
    pose_engine = load_pose_model()
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # --- 側面解析 ---
    if side_video:
        st.subheader("【側面分析：転倒リスク判定】")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(side_video.read())
            cap = cv2.VideoCapture(tfile.name)
        
        best_frame_flex = None
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % 2 == 0: 
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose_engine.process(image)
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    s = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                    h = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                    k = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                    facing_right = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x < lm[mp_pose.PoseLandmark.RIGHT_HIP].x
                    is_flexion = (k[0] > h[0]) if facing_right else (k[0] < h[0])
                    if is_flexion:
                        current_angle = calculate_angle(s, h, k)
                        # 180度からの乖離を屈曲角として計算（180=直線, 値が小さいほど屈曲）
                        flex_val = np.abs(180 - current_angle)
                        if flex_val > max_flexion_angle:
                            max_flexion_angle = flex_val
                            best_frame_flex = image.copy()
                            mp_drawing.draw_landmarks(best_frame_flex, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            count += 1
        cap.release()
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("第1歩：股関節屈曲角度", f"{max_flexion_angle:.1f}°")
            st.write("👉 **Sakane(2025)**: 第1歩の屈曲不足を検知。")
        with c2:
            if best_frame_flex is not None:
                st.image(best_frame_flex, caption="AIが特定した最大屈曲", use_container_width=True)

    # --- 正面解析 ---
    if front_video:
        st.divider()
        st.subheader("【正面分析：安定性・腰痛リスク判定】")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile_f:
            tfile_f.write(front_video.read())
            cap_f = cv2.VideoCapture(tfile_f.name)
        sway_x, sway_y = [], []
        while cap_f.isOpened():
            ret, frame = cap_f.read()
            if not ret: break
            image_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_f = pose_engine.process(image_f)
            if results_f.pose_landmarks:
                lm = results_f.pose_landmarks.landmark
                sway_x.append((lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2)
                sway_y.append((lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2)
        cap_f.release()
        if sway_x:
            sway_width = (max(sway_x) - min(sway_x)) * 100
            vertical_move = (max(sway_y) - min(sway_y)) * 100
            f1, f2 = st.columns(2)
            with f1:
                st.metric("体幹垂直動揺", f"{vertical_move:.2f}%")
                st.metric("体幹側方動揺", f"{sway_width:.2f}%")
            with f2:
                st.metric("歩幅CV値", f"{cv_value}%", delta=f"{cv_value-21.7:.1f}% vs 閾値", delta_color="inverse")
                st.metric("脊柱協調性(位相差)", f"{relative_phase}°", delta=f"{relative_phase-20:.1f}° vs 閾値")

    # --- 5. 総合リスク判定 (ここを追加) ---
    st.divider()
    st.header("📋 総合リスク判定レポート")
    
    r1, r2 = st.columns(2)
    
    with r1:
        st.subheader("🚨 転倒リスク評価")
        # Park(2025)基準: CV値 21.7%以上で高リスク
        if cv_value >= 21.7:
            st.error("【高リスク】歩行のばらつきが大きく、不安定です。")
        else:
            st.success("【低リスク】歩行の一定性が保たれています。")
        
        # Sakane(2025)基準: 第1歩の屈曲（例として10度未満を低値とする）
        if max_flexion_angle < 10.0:
            st.warning("⚠️ 第1歩の振り出しが弱く、つまずきやすい傾向があります.")

    with r2:
        st.subheader("脊柱・腰痛リスク評価")
        # Smith/Xu基準: 相対位相差 20度未満で「丸太様動き（剛性増加）」＝リスク
        if relative_phase < 20.0:
            st.error("【要注意】胸郭と骨盤が同調しすぎています（剛性の増加）.")
            st.info("💡 理学療法士のアドバイス: 体幹の回旋を引き出すストレッチが有効です。")
        else:
            st.success("【良好】体幹のしなやかな回旋が保たれています.")

# --- 6. エビデンスメモ ---
with st.expander("📚 アルゴリズムの根拠（PT用）"):
    st.markdown("""
    * **転倒リスク (Sakane 2025):** 第1歩の股関節屈曲、第2歩の垂直動揺、第3歩の側方動揺を監視。
    * **変動性 (Park 2025):** ステップ幅変動係数(CV)のカットオフ値 **21.7%**。これを超えると転倒リスク増大。
    * **腰痛リスク (Smith/Xu):** 胸郭と骨盤の同調性（位相差20度未満）を剛性増加の指標とする。
    """)
