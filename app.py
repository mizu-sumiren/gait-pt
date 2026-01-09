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
    """3点の座標から角度を算出（股関節屈曲用）"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

def get_line_angle(p1, p2):
    """2点間のベクトルの水平に対する角度（体幹回旋の近似用）"""
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

# --- 3. UI表示 ---
st.title("💃 女性専用 AI歩行ドック [Hybrid-Pro]")
st.info("理学療法士の臨床知見 × 最新エビデンス：動画から動的にリスクを算出します。")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📸 側面（横から）")
    side_video = st.file_uploader("側面動画をアップロード", type=["mp4", "mov"], key="side_up")
with col2:
    st.markdown("### 📸 正面（前から）")
    front_video = st.file_uploader("正面動画をアップロード", type=["mp4", "mov"], key="front_up")

# 解析に使用する変数の初期化（固定値を排除）
max_flexion_angle = 0.0
calculated_cv = 0.0
calculated_phase = 0.0
vertical_sway_mean = 0.0

# --- 4. 解析実行 ---
if st.button("✨ アルゴリズム解析を開始", use_container_width=True):
    if not side_video and not front_video:
        st.warning("解析する動画をアップロードしてください。")
    
    pose_engine = load_pose_model()
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # --- 側面解析：第1歩目の屈曲 ---
    if side_video:
        st.subheader("【側面分析：Sakane(2025)モデル】")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(side_video.read())
            cap = cv2.VideoCapture(tfile.name)
        
        best_frame_flex = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_engine.process(image)
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # 座標取得
                s = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                h = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                k = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                
                # 簡易的な屈曲判定（向きに依存しない絶対角度の乖離）
                current_angle = calculate_angle(s, h, k)
                flex_val = np.abs(180 - current_angle)
                
                if flex_val > max_flexion_angle:
                    max_flexion_angle = flex_val
                    best_frame_flex = image.copy()
                    mp_drawing.draw_landmarks(best_frame_flex, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cap.release()
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("第1歩：股関節屈曲角度", f"{max_flexion_angle:.1f}°")
        with c2:
            if best_frame_flex is not None:
                st.image(best_frame_flex, caption="AIが特定した最大屈曲", use_container_width=True)

    # --- 正面解析：CV値・相対位相差 ---
    if front_video:
        st.divider()
        st.subheader("【正面分析：Park/Smith/Xuモデル】")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile_f:
            tfile_f.write(front_video.read())
            cap_f = cv2.VideoCapture(tfile_f.name)
        
        step_widths = []
        phase_diffs = []
        
        while cap_f.isOpened():
            ret, frame = cap_f.read()
            if not ret: break
            
            image_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_f = pose_engine.process(image_f)
            
            if results_f.pose_landmarks:
                lm = results_f.pose_landmarks.landmark
                
                # 1. 位相差の計算（肩のライン角 vs 骨盤のライン角）
                ls = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                rs = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                lh = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                rh = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                
                s_angle = get_line_angle(ls, rs)
                h_angle = get_line_angle(lh, rh)
                phase_diffs.append(abs(s_angle - h_angle))
                
                # 2. ステップ幅の計算（両踵のX座標距離）
                l_heel = lm[mp_pose.PoseLandmark.LEFT_HEEL].x
                r_heel = lm[mp_pose.PoseLandmark.RIGHT_HEEL].x
                step_widths.append(abs(l_heel - r_heel))
                
        cap_f.release()

        # 数値の集計
        if step_widths:
            # CV値計算: (標準偏差 / 平均) * 100
            calculated_cv = (np.std(step_widths) / np.mean(step_widths)) * 100 if np.mean(step_widths) != 0 else 0
        if phase_diffs:
            # 平均相対位相差
            calculated_phase = np.mean(phase_diffs)

        f1, f2 = st.columns(2)
        with f1:
            st.metric("歩幅CV値（変動性）", f"{calculated_cv:.1f}%", delta=f"{calculated_cv-21.7:.1f}%", delta_color="inverse")
            st.caption("※閾値 21.7% (Park 2025)")
        with f2:
            st.metric("脊柱協調性(位相差)", f"{calculated_phase:.1f}°", delta=f"{calculated_phase-20:.1f}°")
            st.caption("※閾値 20.0° (Smith/Xu)")

    # --- 5. 総合リスク判定レポート ---
    st.divider()
    st.header("📋 総合リスク判定レポート")
    
    r1, r2 = st.columns(2)
    
    with r1:
        st.subheader("🚨 転倒リスク評価")
        if calculated_cv >= 21.7:
            st.error(f"【高リスク】CV値 {calculated_cv:.1f}%。歩行のバラつきが大きく、不安定です。")
        else:
            st.success(f"【低リスク】CV値 {calculated_cv:.1f}%。歩行の一定性が保たれています。")
        
        if max_flexion_angle < 15.0: # 臨床的目安としての15度
            st.warning("⚠️ 第1歩の振り出しが弱く、つまずきやすい傾向があります。")

    with r2:
        st.subheader("脊柱・腰痛リスク評価")
        if calculated_phase < 20.0:
            st.error(f"【要注意】位相差 {calculated_phase:.1f}°。胸郭と骨盤が同調しすぎています（剛性の増加）。")
            st.info("💡 PTアドバイス: 体幹のしなやかさを出す回旋ストレッチを推奨します。")
        else:
            st.success(f"【良好】位相差 {calculated_phase:.1f}°。体幹のしなやかな回旋が保たれています。")

# --- 6. エビデンスメモ ---
with st.expander("📚 アルゴリズムの根拠（PT用）"):
    st.markdown("""
    * **転倒リスク (Sakane 2025):** 第1歩の股関節屈曲角度を分析。
    * **歩行変動性 (Park 2025):** ステップ幅変動係数(CV)のカットオフ値 **21.7%**。
    * **腰痛リスク (Smith/Xu):** 胸郭と骨盤の相対位相差 **20度未満** を剛性増加（脊柱の固定化）の指標とする。
    """)
