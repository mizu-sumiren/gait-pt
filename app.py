import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import io
import math
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# --- MediaPipe初期化 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- ページ設定 ---
st.set_page_config(page_title="AI歩行解析 All-in-One", page_icon="🏥", layout="wide")

# --- CSS設定（サイドバーボタン確保） ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- サイドバー：モード選択 ---
st.sidebar.header("⚙️ モード設定")
app_mode = st.sidebar.radio(
    "分析モードを選択してください",
    ["Pro版 (臨床・詳細評価)", "Lite版 (高速・動画のみ)"]
)

# --- タイトル切り替え ---
if app_mode == "Pro版 (臨床・詳細評価)":
    st.title("🏥 AI歩行ドック Pro")
    st.markdown("【臨床用】身体機能 × 姿勢制御 × 歩行の質")
else:
    st.title("⚡ AI歩行ドック Lite")
    st.markdown("【検診用】動画のみで即座にリスク判定")

# --- 変数初期化 ---
# Liteモードでもエラーが出ないように初期値を設定
toe_grip_l = toe_grip_r = 0
hip_flex_l = hip_flex_r = 0
one_leg_l = one_leg_r = 0
frt = ffd = 0
pain_areas = []

# --- サイドバー入力 ---
st.sidebar.header("📋 対象者情報")
client_name = st.sidebar.text_input("氏名", "テスト 太郎 様")

if app_mode == "Pro版 (臨床・詳細評価)":
    # --- Proモードの入力欄 ---
    with st.sidebar.expander("1. 問診・痛み", expanded=True):
        pain_areas = st.multiselect("痛み・違和感", ["特になし", "首", "肩", "腰", "股関節", "膝", "足首"])

    with st.sidebar.expander("2. 身体機能測定", expanded=True):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**左 (L)**")
            grip_l = st.number_input("握力(左)", value=20.0)
            hip_flex_l = st.number_input("股屈曲(左)", value=0.9)
            one_leg_l = st.number_input("片脚立位(左)", value=15.0)
            toe_grip_l = st.number_input("足趾把持(左)", value=10.0)
        with col_s2:
            st.markdown("**右 (R)**")
            grip_r = st.number_input("握力(右)", value=25.0)
            hip_flex_r = st.number_input("股屈曲(右)", value=1.2)
            one_leg_r = st.number_input("片脚立位(右)", value=60.0)
            toe_grip_r = st.number_input("足趾把持(右)", value=20.0)
        st.markdown("---")
        frt = st.number_input("FRT (cm)", value=25.0)
        ffd = st.number_input("FFD (cm)", value=0.0)
else:
    # --- Liteモードの入力欄 ---
    st.sidebar.caption("Liteモード起動中：身体機能チェックは省略されます。")
    pain_areas = st.sidebar.multiselect("痛み・違和感 (任意)", ["特になし", "首", "肩", "腰", "股関節", "膝", "足首"])


# --- 幾何学計算関数 ---
def calculate_angle_3points(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def calculate_slope(a, b):
    if a is None or b is None: return 0
    dy = a[1] - b[1]
    dx = a[0] - b[0]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def calculate_vertical_angle(a, b):
    if a is None or b is None: return 0
    dy = b[1] - a[1]
    dx = b[0] - a[0]
    angle_rad = math.atan2(dx, dy) 
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# --- 動画分析ロジック ---
def analyze_front_view(landmarks_history):
    if not landmarks_history: return None
    head_tilts, shoulder_slopes, hip_centers_x = [], [], []
    for lms in landmarks_history:
        l_ear, r_ear = [lms[7].x, lms[7].y], [lms[8].x, lms[8].y]
        head_tilts.append(calculate_slope(l_ear, r_ear))
        l_sh, r_sh = [lms[11].x, lms[11].y], [lms[12].x, lms[12].y]
        shoulder_slopes.append(calculate_slope(l_sh, r_sh))
        hip_centers_x.append((lms[23].x + lms[24].x) / 2)
    return {"head_tilt": np.mean(np.abs(head_tilts)), "shoulder_slope": np.mean(shoulder_slopes), "sway_amplitude": max(hip_centers_x) - min(hip_centers_x)}

def analyze_side_view(landmarks_history, fps):
    if not landmarks_history: return None
    ankle_distances, shin_lengths, trunk_leans, ankle_heights = [], [], [], []
    hip_ext_l_max, hip_ext_r_max = 0, 0
    for lms in landmarks_history:
        la, ra, lk = np.array([lms[27].x, lms[27].y]), np.array([lms[28].x, lms[28].y]), np.array([lms[25].x, lms[25].y])
        ankle_distances.append(np.linalg.norm(la - ra))
        shin_lengths.append(np.linalg.norm(lk - la))
        trunk_leans.append(calculate_vertical_angle([lms[11].x, lms[11].y], [lms[23].x, lms[23].y]))
        l_ang = calculate_angle_3points([lms[11].x, lms[11].y], [lms[23].x, lms[23].y], [lms[25].x, lms[25].y])
        r_ang = calculate_angle_3points([lms[12].x, lms[12].y], [lms[24].x, lms[24].y], [lms[26].x, lms[26].y])
        if l_ang > hip_ext_l_max: hip_ext_l_max = l_ang
        if r_ang > hip_ext_r_max: hip_ext_r_max = r_ang
        ankle_heights.append(lms[27].y)
    
    steps = 0
    threshold = np.mean(ankle_distances)
    for i in range(1, len(ankle_distances)-1):
        if ankle_distances[i] > ankle_distances[i-1] and ankle_distances[i] > ankle_distances[i+1] and ankle_distances[i] > threshold:
            steps += 1
            
    duration = len(landmarks_history) / fps
    cadence = (steps / duration) * 60 if duration > 0 else 0
    step_ratio = (np.mean(ankle_distances) / np.mean(shin_lengths)) if shin_lengths else 0 # 簡易計算
    return {"cadence": cadence, "step_ratio": step_ratio, "max_hip_ext_l": hip_ext_l_max, "max_hip_ext_r": hip_ext_r_max, "avg_trunk_lean": np.mean(trunk_leans), "foot_clearance_score": max(ankle_heights) - min(ankle_heights)}

# --- フィードバック生成 (モード分岐) ---
def generate_feedback(mode, data, front, side):
    fb = []
    
    # --- 1. 動画分析 (共通) ---
    if front:
        if front['head_tilt'] > 3.0: fb.append("⚠️ **【頭部の傾き】** 正面から見て頭が傾いています。")
        if abs(front['shoulder_slope']) > 3.0: fb.append("⚠️ **【肩の高さ】** 左右の肩の高さが揃っていません。")
        if front['sway_amplitude'] > 0.15: fb.append("⚠️ **【スウェイ】** 歩行時に腰が左右に揺れています。")
    if side:
        if abs(side['avg_trunk_lean']) > 10.0: fb.append("⚠️ **【猫背・前傾】** 上半身が前に倒れています。")
        if side['foot_clearance_score'] < 0.05: fb.append("⚠️ **【すり足】** 足があまり上がっていません。")
        if abs(side['max_hip_ext_l'] - side['max_hip_ext_r']) > 5.0: fb.append("⚠️ **【股関節伸展の左右差】** 片側の足の蹴り出しが弱くなっています。")
    
    # --- 2. 身体機能 (Proのみ) ---
    if mode == "Pro版 (臨床・詳細評価)":
        if (data['toe_l'] + data['toe_r'])/2 < 20: fb.append("ℹ️ **【足指機能低下】** 地面を掴む力が弱めです。")
        if abs(data['hip_l'] - data['hip_r']) > 0.15: fb.append("ℹ️ **【股関節筋力の左右差】** 筋力差が歩行の揺れを助長しています。")
        if data['ols_l'] < 20 or data['ols_r'] < 20: fb.append("ℹ️ **【バランス低下】** 片脚立位が不安定です。")

    if not fb: fb.append("✅ **素晴らしい状態です！** リスクとなる動きは見当たりません。")
    return fb

# --- 共通処理 ---
def process_video(uploaded_file, view_type):
    if uploaded_file is None: return None, None
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    history = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.line(image, (width//2, 0), (width//2, height), (0, 255, 255), 1) 
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                history.append(results.pose_landmarks.landmark)
            out.write(image)
    cap.release(); out.release()
    metrics = analyze_front_view(history) if view_type == 'front' else analyze_side_view(history, fps)
    return output_path, metrics

def create_pdf(mode, client_name, data, feedbacks, f_m, s_m):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    h = A4[1]
    c.setFont("Helvetica-Bold", 16); c.drawString(50, h-50, f"Gait Analysis Report ({mode[:3]})")
    c.setFont("Helvetica", 12); c.drawString(50, h-80, f"Name: {client_name}")
    
    y = h-120
    if s_m:
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Gait Metrics")
        c.setFont("Helvetica", 10); y-=20
        c.drawString(60, y, f"Hip Ext: L{s_m['max_hip_ext_l']:.0f} / R{s_m['max_hip_ext_r']:.0f}")
        c.drawString(200, y, f"Trunk: {s_m['avg_trunk_lean']:.1f}")
        y-=40

    if mode == "Pro版 (臨床・詳細評価)":
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Physical Data")
        c.setFont("Helvetica", 10); y-=20
        c.drawString(60, y, f"Toe Grip: L{data['toe_l']} / R{data['toe_r']}")
        c.drawString(200, y, f"One Leg: L{data['ols_l']} / R{data['ols_r']}")
        y-=40

    c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Feedback Summary")
    c.setFont("Helvetica", 10); y-=20
    c.drawString(60, y, "See app screen for details.")
    
    c.showPage(); c.save(); buffer.seek(0)
    return buffer

# --- メインレイアウト ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("① 正面動画")
    file_front = st.file_uploader("Front View", type=['mp4', 'mov'], key="f")
with col2:
    st.subheader("② 側面動画")
    file_side = st.file_uploader("Side View", type=['mp4', 'mov'], key="s")

if st.button("🚀 分析スタート"):
    path_f, metrics_f = process_video(file_front, 'front')
    path_s, metrics_s = process_video(file_side, 'side')
    
    st.markdown("---")
    vc1, vc2 = st.columns(2)
    with vc1: 
        if path_f: st.video(path_f)
    with vc2: 
        if path_s: st.video(path_s)
        
    st.subheader("📊 解析データ")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("##### 正面データ")
        if metrics_f:
            st.metric("頭部傾き", f"{metrics_f['head_tilt']:.1f}°")
            st.metric("スウェイ", f"{metrics_f['sway_amplitude']:.2f}")
    with dc2:
        st.markdown("##### 側面データ")
        if metrics_s:
            st.metric("体幹前傾", f"{metrics_s['avg_trunk_lean']:.1f}°")
            st.metric("伸展(L/R)", f"{int(metrics_s['max_hip_ext_l'])}° / {int(metrics_s['max_hip_ext_r'])}°")

    st.header("👨‍⚕️ AIフィードバック")
    input_data = {'toe_l': toe_grip_l, 'toe_r': toe_grip_r, 'hip_l': hip_flex_l, 'hip_r': hip_flex_r, 'ols_l': one_leg_l, 'ols_r': one_leg_r}
    feedbacks = generate_feedback(app_mode, input_data, metrics_f, metrics_s)
    for msg in feedbacks:
        if "⚠️" in msg: st.error(msg)
        elif "ℹ️" in msg: st.warning(msg)
        else: st.info(msg)

    st.subheader("📥 保存")
    rc1, rc2 = st.columns([3, 1])
    with rc2:
        pdf_data = create_pdf(app_mode, client_name, input_data, feedbacks, metrics_f, metrics_s)
        st.download_button("📄 PDF DL", pdf_data, "report.pdf", "application/pdf")
        st.markdown("---")
        if path_f:
            with open(path_f, 'rb') as v: st.download_button("🎥 正面動画 DL", v, "front.mp4", "video/mp4")
        if path_s:
            with open(path_s, 'rb') as v: st.download_button("🎥 側面動画 DL", v, "side.mp4", "video/mp4")
