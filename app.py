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
st.set_page_config(page_title="AI歩行解析アプリ Pro", page_icon="🏃‍♂️", layout="wide")

# --- CSS設定（修正版：上のバーを表示させてサイドバーボタンを復活！） ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            /* header {visibility: hidden;} ←ここを削除しました */
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🏃‍♂️ AI歩行ドック Pro - Clinical Gait Lab")
st.markdown("姿勢制御(頭部・体幹) × 歩行の質(すり足・伸展) × 身体機能")

# --- サイドバー入力 ---
st.sidebar.header("📋 測定データ入力")
with st.sidebar.expander("1. 基本情報・問診", expanded=True):
    client_name = st.text_input("氏名", "テスト 太郎 様")
    pain_areas = st.multiselect("痛み・違和感", ["特になし", "首", "肩", "腰", "股関節(右)", "股関節(左)", "膝(右)", "膝(左)", "足首"])

with st.sidebar.expander("2. 身体機能測定結果", expanded=True):
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

# --- 正面動画分析ロジック ---
def analyze_front_view(landmarks_history):
    if not landmarks_history: return None
    
    head_tilts = []
    shoulder_slopes = []
    hip_centers_x = []
    
    for lms in landmarks_history:
        l_ear = [lms[7].x, lms[7].y]
        r_ear = [lms[8].x, lms[8].y]
        head_tilts.append(calculate_slope(l_ear, r_ear))
        
        l_sh = [lms[11].x, lms[11].y]
        r_sh = [lms[12].x, lms[12].y]
        shoulder_slopes.append(calculate_slope(l_sh, r_sh))
        
        mid_hip_x = (lms[23].x + lms[24].x) / 2
        hip_centers_x.append(mid_hip_x)
        
    return {
        "head_tilt": np.mean(np.abs(head_tilts)),
        "shoulder_slope": np.mean(shoulder_slopes),
        "sway_amplitude": max(hip_centers_x) - min(hip_centers_x)
    }

# --- 側面動画分析ロジック ---
def analyze_side_view(landmarks_history, fps):
    if not landmarks_history: return None

    ankle_distances = []
    shin_lengths = []
    trunk_leans = []
    ankle_heights = []
    
    hip_ext_l_max = 0
    hip_ext_r_max = 0

    for lms in landmarks_history:
        la = np.array([lms[27].x, lms[27].y])
        ra = np.array([lms[28].x, lms[28].y])
        lk = np.array([lms[25].x, lms[25].y])
        
        ankle_distances.append(np.linalg.norm(la - ra))
        shin_lengths.append(np.linalg.norm(lk - la))
        
        trunk_angle = calculate_vertical_angle([lms[11].x, lms[11].y], [lms[23].x, lms[23].y])
        trunk_leans.append(trunk_angle)
        
        l_ang = calculate_angle_3points([lms[11].x, lms[11].y], [lms[23].x, lms[23].y], [lms[25].x, lms[25].y])
        r_ang = calculate_angle_3points([lms[12].x, lms[12].y], [lms[24].x, lms[24].y], [lms[26].x, lms[26].y])
        if l_ang > hip_ext_l_max: hip_ext_l_max = l_ang
        if r_ang > hip_ext_r_max: hip_ext_r_max = r_ang
        
        ankle_heights.append(lms[27].y)

    steps = 0
    peaks = []
    threshold = np.mean(ankle_distances)
    for i in range(1, len(ankle_distances)-1):
        if ankle_distances[i] > ankle_distances[i-1] and ankle_distances[i] > ankle_distances[i+1] and ankle_distances[i] > threshold:
            steps += 1
            peaks.append(ankle_distances[i])

    duration = len(landmarks_history) / fps
    cadence = (steps / duration) * 60 if duration > 0 else 0
    step_ratio = (np.mean(peaks) / np.mean(shin_lengths)) if peaks and shin_lengths else 0
    
    return {
        "cadence": cadence,
        "step_ratio": step_ratio,
        "max_hip_ext_l": hip_ext_l_max,
        "max_hip_ext_r": hip_ext_r_max,
        "avg_trunk_lean": np.mean(trunk_leans),
        "foot_clearance_score": max(ankle_heights) - min(ankle_heights)
    }

# --- フィードバック生成 ---
def generate_clinical_feedback(data, front_metrics, side_metrics):
    feedback = []
    
    if front_metrics:
        if front_metrics['head_tilt'] > 3.0:
            feedback.append("⚠️ **【頭部の傾き】** 正面から見て頭が傾いています。")
        slope = front_metrics['shoulder_slope']
        if abs(slope) > 3.0:
            side = "右" if slope > 0 else "左"
            feedback.append(f"⚠️ **【肩の下がり ({side}下がり)】** 肩のラインが水平ではありません。")
        if front_metrics['sway_amplitude'] > 0.15:
            feedback.append("⚠️ **【骨盤のスウェイ】** 歩行時に骨盤が左右に大きく揺れています。")

    if side_metrics:
        if abs(side_metrics['avg_trunk_lean']) > 10.0:
            feedback.append("⚠️ **【体幹の前傾】** 身体が前に倒れています。")
        if side_metrics['foot_clearance_score'] < 0.05:
            feedback.append("⚠️ **【すり足気味】** 足があまり上がっていません。つまずきに注意です。")
        
        ext_l = side_metrics['max_hip_ext_l']
        ext_r = side_metrics['max_hip_ext_r']
        if abs(ext_l - ext_r) > 5.0:
            weaker = "左" if ext_l < ext_r else "右"
            feedback.append(f"⚠️ **【股関節伸展の左右差 ({weaker}制限)】** {weaker}足の蹴り出しが弱くなっています。")

    if (data['toe_l'] + data['toe_r'])/2 < 20:
        feedback.append("ℹ️ **【足指機能低下】** 地面を掴む力が弱めです。")
    
    if not feedback:
        feedback.append("✅ **良好な歩行状態です！**")

    return feedback

# --- 共通処理 ---
def process_video(uploaded_file, view_type):
    if uploaded_file is None: return None, None
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    landmarks_history = []
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
                landmarks_history.append(results.pose_landmarks.landmark)
            out.write(image)
            
    cap.release()
    out.release()
    
    metrics = analyze_front_view(landmarks_history) if view_type == 'front' else analyze_side_view(landmarks_history, fps)
    return output_path, metrics

def create_pdf(client_name, data, feedbacks, f_metrics, s_metrics):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Gait Analysis Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Name: {client_name}")

    y = height - 120
    c.drawString(50, y, "See app for details.")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --- メイン処理 ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("① 正面動画")
    file_front = st.file_uploader("Front View", type=['mp4', 'mov'], key="f")
with col2:
    st.subheader("② 側面動画")
    file_side = st.file_uploader("Side View", type=['mp4', 'mov'], key="s")

if st.button("🚀 臨床詳細分析を実行"):
    path_f, metrics_f = process_video(file_front, 'front')
    path_s, metrics_s = process_video(file_side, 'side')
    
    st.markdown("---")
    
    v_c1, v_c2 = st.columns(2)
    with v_c1: 
        if path_f: st.video(path_f)
    with v_c2: 
        if path_s: st.video(path_s)
        
    st.subheader("📊 解析データ")
    d_c1, d_c2 = st.columns(2)
    with d_c1:
        st.markdown("##### 正面データ")
        if metrics_f:
            st.metric("頭部傾き", f"{metrics_f['head_tilt']:.1f}°")
            st.metric("肩傾き", f"{metrics_f['shoulder_slope']:.1f}°")
            st.metric("スウェイ", f"{metrics_f['sway_amplitude']:.2f}")
    with d_c2:
        st.markdown("##### 側面データ")
        if metrics_s:
            st.metric("体幹前傾", f"{metrics_s['avg_trunk_lean']:.1f}°")
            st.metric("伸展(L)", f"{int(metrics_s['max_hip_ext_l'])}°")
            st.metric("伸展(R)", f"{int(metrics_s['max_hip_ext_r'])}°")

    st.header("👨‍⚕️ AI理学療法士フィードバック")
    input_data = {
        'toe_l': toe_grip_l, 'toe_r': toe_grip_r,
        'hip_l': hip_flex_l, 'hip_r': hip_flex_r,
        'ols_l': one_leg_l, 'ols_r': one_leg_r,
    }
    
    feedbacks = generate_clinical_feedback(input_data, metrics_f, metrics_s)
    for msg in feedbacks:
        if "⚠️" in msg: st.error(msg)
        else: st.info(msg)

    st.subheader("📥 保存")
    rec_col1, rec_col2 = st.columns([3, 1])
    with rec_col2:
        pdf_data = create_pdf(client_name, input_data, feedbacks, metrics_f, metrics_s)
        st.download_button("📄 PDF DL", pdf_data, "report.pdf", "application/pdf")
        st.markdown("---")
        if path_f:
            with open(path_f, 'rb') as v: st.download_button("🎥 正面動画 DL", v, "front.mp4", "video/mp4")
        if path_s:
            with open(path_s, 'rb') as v: st.download_button("🎥 側面動画 DL", v, "side.mp4", "video/mp4")
