import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import japanize_matplotlib

# ページ設定
st.set_page_config(page_title="予防特化型AI姿勢分析", page_icon="🛡️", layout="wide")

st.title("🛡️ 働く人のための身体機能・姿勢チェック")
st.markdown("「長く働ける身体」を作るために、現在の姿勢と身体機能、そして隠れたリスクを分析します。")

# --- サイドバー：問診 & 機能チェック ---
st.sidebar.header("📋 問診・身体機能データ")

# 1. 問診（痛み・既往）
st.sidebar.subheader("1. 問診")
pain_areas = st.sidebar.multiselect(
    "現在、痛みや違和感がある部位（複数選択可）",
    ["特になし", "首・肩", "腰", "股関節(右)", "股関節(左)", "膝(右)", "膝(左)", "足首・足部"]
)
history = st.sidebar.text_area("既往歴（過去の怪我や手術など）", placeholder="例：3年前に右足首を捻挫してから、なんとなく違和感がある")

# 2. 身体機能
st.sidebar.subheader("2. 身体機能測定")
st.sidebar.caption("※FFDは床につけばプラス、届かなければマイナス")
ffd = st.sidebar.number_input("FFD (立位体前屈) cm", value=13.6, help="床より下に行けばプラスの値")
hip_flex_l = st.sidebar.number_input("股屈曲筋力(左) kgf/kg", value=1.21)
hip_flex_r = st.sidebar.number_input("股屈曲筋力(右) kgf/kg", value=1.36)
one_leg_l = st.sidebar.number_input("片脚立位(左) 秒", value=120)
one_leg_r = st.sidebar.number_input("片脚立位(右) 秒", value=120)

# --- メイン：動画分析パート ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

uploaded_file = st.file_uploader("歩行または立位動画をアップロード", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 解析用
    data = []
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    st.info("リスク分析を実行中...")
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                def get_c(name): return [landmarks[name.value].x, landmarks[name.value].y]
                
                # 膝角度
                l_knee_ang = calculate_angle(get_c(mp_pose.PoseLandmark.LEFT_HIP), get_c(mp_pose.PoseLandmark.LEFT_KNEE), get_c(mp_pose.PoseLandmark.LEFT_ANKLE))
                r_knee_ang = calculate_angle(get_c(mp_pose.PoseLandmark.RIGHT_HIP), get_c(mp_pose.PoseLandmark.RIGHT_KNEE), get_c(mp_pose.PoseLandmark.RIGHT_ANKLE))
                
                # 肩の傾き (+は右下がり)
                l_sh = get_c(mp_pose.PoseLandmark.LEFT_SHOULDER)
                r_sh = get_c(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                shoulder_tilt = (r_sh[1] - l_sh[1]) * 100 
                
                # 骨盤の傾き (+は右下がり)
                l_hip = get_c(mp_pose.PoseLandmark.LEFT_HIP)
                r_hip = get_c(mp_pose.PoseLandmark.RIGHT_HIP)
                hip_tilt = (r_hip[1] - l_hip[1]) * 100

                data.append({
                    "frame": frame_idx, 
                    "l_knee": l_knee_ang, 
                    "r_knee": r_knee_ang, 
                    "shoulder_tilt": shoulder_tilt,
                    "hip_tilt": hip_tilt
                })
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            out.write(image)
            frame_idx += 1
            if total_frames > 0: progress_bar.progress(min(frame_idx/total_frames, 1.0))

    cap.release()
    out.release()
    df = pd.DataFrame(data)

    if not df.empty:
        st.success("解析完了！")
        
        # --- レイアウト ---
        col1, col2 = st.columns([1, 1])
        with col1:
            st.video(output_path)
            
            # --- ここで「問診」の内容を表示 ---
            st.markdown("### 🗣️ 問診情報")
            if "特になし" in pain_areas or not pain_areas:
                st.write("・現在、自覚している痛みはありません。")
            else:
                st.error(f"・痛み/違和感あり: **{', '.join(pain_areas)}**")
            
            if history:
                st.info(f"・既往歴: {history}")
            else:
                st.write("・既往歴の入力なし")

        with col2:
            st.subheader("📉 姿勢バランス解析")
            # グラフ: 肩と骨盤の傾きを同時に見る（連動性の確認）
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(df['frame'], df['shoulder_tilt'], label='肩の傾き', color='green', alpha=0.7)
            ax.plot(df['frame'], df['hip_tilt'], label='骨盤の傾き', color='orange', alpha=0.7, linestyle='--')
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_title("肩と骨盤の連動性 (＋は右下がり)")
            ax.legend()
            st.pyplot(fig)

        st.markdown("---")
        st.header("🛡️ 予防・改善アドバイス Report")
        
        # --- ロジックのアップデート ---
        
        # 1. 柔軟性評価（FFD修正版）
        st.subheader("1. 柔軟性と安定性")
        if ffd > 5.0:
            st.info(f"✅ **柔軟性は非常に高いです (FFD {ffd}cm)**\n\n身体はとても柔らかいですが、逆に**「関節が緩い（不安定）」**可能性があります。動画で肩や骨盤の揺れが見られる場合、ストレッチよりも**「体幹トレーニング」や「筋力強化」**で関節を安定させることが、将来の腰痛予防になります。")
        elif ffd < -5.0:
            st.warning(f"⚠️ **柔軟性低下 (FFD {ffd}cm)**\n\n身体が硬く、腰への負担が増しやすい状態です。ストレッチを重点的に行いましょう。")
        else:
            st.success(f"✅ 柔軟性は標準的です (FFD {ffd}cm)")

        # 2. 痛みと動作の関連付け (NEW!)
        st.subheader("2. リスク管理")
        risk_detected = False
        
        # 膝の痛みがある場合
        if "膝(右)" in pain_areas or "膝(左)" in pain_areas:
            risk_detected = True
            st.error("🚨 **膝の痛みをかばっている可能性があります**\n\n問診で膝の痛みがあります。動画のグラフで左右の膝の曲がり方に差がある場合、痛くない方の足に過剰な負担がかかっている恐れがあります。")
        
        # 腰痛がある場合
        if "腰" in pain_areas:
            risk_detected = True
            st.error("🚨 **腰へのストレス注意**\n\n問診で腰の違和感があります。FFDが柔らかすぎる（腹圧が抜けている）か、逆に硬すぎる（骨盤が動かない）ことが原因かもしれません。")

        if not risk_detected:
            st.success("現在、動作に影響を与えるような強い痛みリスクは見当たりませんが、引き続き「左右差」に注意して予防しましょう。")

        # 3. 筋力バランス
        st.subheader("3. 筋力バランス")
        diff_hip = hip_flex_r - hip_flex_l
        if abs(diff_hip) > 0.15:
            weak_side = "左" if diff_hip > 0 else "右"
            st.warning(f"⚠️ **{weak_side}側の股関節筋力が弱いです**\n\nこの左右差が、歩行時のふらつきや肩の下がりの原因になっている可能性があります。{weak_side}側の強化をおすすめします。")
        else:
            st.success("✅ 筋力の左右差は少なく良好です。")

else:
    st.info("動画をアップロードしてください")
