import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import io
import math
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import Image

# --- MediaPipe初期化 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- ページ設定 ---
st.set_page_config(page_title="AI姿勢・歩行分析ラボ", page_icon="🏥", layout="wide")

# --- CSS設定 ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- サイドバー：モード選択 ---
st.sidebar.header("⚙️ 分析モード")
app_mode = st.sidebar.radio(
    "モードを選択してください",
    ["動画：歩行分析 (Pro)", "動画：歩行分析 (Lite)", "静止画：姿勢分析 (立位/座位)"]
)

# --- タイトル表示 ---
if "歩行" in app_mode:
    st.title("🏃‍♂️ AI歩行ドック")
    st.markdown(f"モード: {app_mode}")
else:
    st.title("📸 AI姿勢分析ラボ")
    st.markdown("立位・座位の静止画アライメント評価")

# --- 変数初期化 ---
toe_grip_l = toe_grip_r = 0
hip_flex_l = hip_flex_r = 0
one_leg_l = one_leg_r = 0
frt = ffd = 0
pain_areas = []

# --- サイドバー入力 ---
st.sidebar.header("📋 対象者情報")
client_name = st.sidebar.text_input("氏名", "テスト 太郎 様")

# Proモードの場合のみ詳細入力
if app_mode == "動画：歩行分析 (Pro)":
    with st.sidebar.expander("1. 問診・痛み", expanded=True):
        pain_areas = st.multiselect("痛み", ["なし", "首", "肩", "腰", "股関節", "膝", "足首"])
    with st.sidebar.expander("2. 身体機能測定", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**左 (L)**")
            grip_l = st.number_input("握力L", 20.0); hip_flex_l = st.number_input("股屈曲L", 0.9)
            one_leg_l = st.number_input("片脚L", 15.0); toe_grip_l = st.number_input("足把持L", 10.0)
        with c2:
            st.markdown("**右 (R)**")
            grip_r = st.number_input("握力R", 25.0); hip_flex_r = st.number_input("股屈曲R", 1.2)
            one_leg_r = st.number_input("片脚R", 60.0); toe_grip_r = st.number_input("足把持R", 20.0)
        st.markdown("---")
        frt = st.number_input("FRT", 25.0); ffd = st.number_input("FFD", 0.0)

# --- 幾何学計算関数 ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def calculate_slope(a, b):
    if a is None or b is None: return 0
    return math.degrees(math.atan2(a[1]-b[1], a[0]-b[0]))

def calculate_vertical_angle(a, b):
    # 垂直線からの角度（前傾・後傾）
    if a is None or b is None: return 0
    return math.degrees(math.atan2(b[0]-a[0], b[1]-a[1]))

# --- 静止画分析ロジック (NEW!) ---
def analyze_static_image(image, posture_type):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks: return image, None

        h, w, _ = image.shape
        lms = results.pose_landmarks.landmark
        
        # 描画
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 座標取得ヘルパー
        def get_p(idx): return [lms[idx].x * w, lms[idx].y * h]
        
        metrics = {}
        
        # --- 共通評価: 頭部・肩の傾き (正面想定) ---
        l_ear, r_ear = get_p(7), get_p(8)
        l_sh, r_sh = get_p(11), get_p(12)
        metrics['head_tilt'] = calculate_slope(l_ear, r_ear)
        metrics['shoulder_slope'] = calculate_slope(l_sh, r_sh)

        # --- 側面評価用データの計算 ---
        # 1. 耳と肩の位置関係 (Forward Head Posture)
        # 耳(7)が肩(11)よりどれくらい前にあるか (X座標の差)
        # 正規化のため、肩幅か身長に対する比率で出すのが理想だが、ここでは簡易的にピクセル差を見る
        ear_x = (lms[7].x + lms[8].x) / 2
        shoulder_x = (lms[11].x + lms[12].x) / 2
        metrics['forward_head_score'] = (shoulder_x - ear_x) * 100 # 正の値なら耳が前
        
        # 2. 体幹の前傾
        metrics['trunk_lean'] = calculate_vertical_angle(l_sh, get_p(23))

        # --- 立位・座位ごとの特異的評価 ---
        if posture_type == "立位 (Standing)":
            # 膝の伸展度 (11-23-25) -> 立位なら180度近いか
            hip = get_p(23); knee = get_p(25); ankle = get_p(27)
            metrics['knee_angle'] = calculate_angle(hip, knee, ankle)
            # 重心線 (耳-肩-腰-膝-外果) のズレチェックは簡易的に「耳とくるぶしのX差」で
            metrics['plumb_line_dev'] = (lms[7].x - lms[27].x) * 100

        elif posture_type == "座位 (Sitting)":
            # 股関節屈曲角度 (11-23-25) -> 90度が理想
            sh = get_p(11); hip = get_p(23); knee = get_p(25)
            metrics['hip_angle'] = calculate_angle(sh, hip, knee)
            # 膝角度 -> 90度が理想
            ankle = get_p(27)
            metrics['knee_angle'] = calculate_angle(hip, knee, ankle)

        return annotated_image, metrics

# --- 静止画フィードバック生成 ---
def generate_static_feedback(metrics, posture_type):
    fb = []
    # 正面要素
    if abs(metrics['head_tilt']) > 3.0: fb.append("⚠️ **【頭部の傾き】** 首が傾いています。視覚や噛み合わせの影響が疑われます。")
    if abs(metrics['shoulder_slope']) > 3.0: fb.append("⚠️ **【肩の高さ】** 左右の肩の高さが違います。荷物の持ち癖や側弯のチェックを。")
    
    # 側面要素 (Forward Head) - 向きによるので絶対値で簡易判定
    # ※カメラの向きに依存するため、あくまで参考値として警告
    if abs(metrics['forward_head_score']) > 5.0: 
        fb.append("⚠️ **【スマホ首 (FHP)】** 頭が肩より前に出ています。首・肩こりの主原因です。")

    if posture_type == "立位 (Standing)":
        if metrics['knee_angle'] < 165: fb.append("⚠️ **【膝曲がり】** 膝が伸び切っていません。加齢による変形や筋力低下の可能性があります。")
        if abs(metrics['trunk_lean']) > 10: fb.append("⚠️ **【姿勢の崩れ】** 上半身が垂直から傾いています（猫背または反り腰）。")

    elif posture_type == "座位 (Sitting)":
        if metrics['hip_angle'] > 110: fb.append("ℹ️ **【骨盤後傾】** 椅子に浅く座り、背もたれに寄りかかりすぎています（仙骨座り）。")
        if metrics['knee_angle'] < 80: fb.append("ℹ️ **【足の引き込み】** 足を手前に引きすぎています。膝裏の血流が悪くなる原因です。")

    if not fb: fb.append("✅ **グッドポスチャー！** 非常に綺麗な姿勢です。")
    return fb

# --- 動画分析関数 (既存) ---
def analyze_video_metrics(history, fps):
    if not history: return None
    # (既存のロジックを簡略化して統合)
    dists = []
    for lms in history:
        la, ra = np.array([lms[27].x, lms[27].y]), np.array([lms[28].x, lms[28].y])
        dists.append(np.linalg.norm(la - ra))
    
    steps = 0; thresh = np.mean(dists)
    for i in range(1, len(dists)-1):
        if dists[i] > dists[i-1] and dists[i] > dists[i+1] and dists[i] > thresh: steps += 1
    
    duration = len(history) / fps
    cadence = (steps / duration) * 60 if duration > 0 else 0
    return {"cadence": cadence, "steps": steps}

def process_video(file):
    if not file: return None, None
    tfile = tempfile.NamedTemporaryFile(delete=False); tfile.write(file.read())
    cap = cv2.VideoCapture(tfile.name)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    history = []
    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret: break
            img.flags.writeable = False; res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img.flags.writeable = True; img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.line(img, (w//2,0), (w//2,h), (0,255,255), 1)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                history.append(res.pose_landmarks.landmark)
            out.write(img)
    cap.release(); out.release()
    return path, analyze_video_metrics(history, fps)

# --- PDF生成 (統合版) ---
def create_unified_pdf(mode, name, feedbacks, vid_met=None, stat_met=None):
    b = io.BytesIO()
    c = canvas.Canvas(b, pagesize=A4); h = A4[1]
    c.setFont("Helvetica-Bold", 16); c.drawString(50, h-50, f"Analysis Report: {mode}")
    c.setFont("Helvetica", 12); c.drawString(50, h-80, f"Name: {name}")
    
    y = h-120
    c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Metrics")
    y -= 20; c.setFont("Helvetica", 10)
    
    if vid_met: # 動画データ
        c.drawString(60, y, f"Cadence: {vid_met['cadence']:.1f} steps/min")
        c.drawString(200, y, f"Steps Detected: {vid_met['steps']}")
    elif stat_met: # 静止画データ
        c.drawString(60, y, f"Head Tilt: {stat_met['head_tilt']:.1f} deg")
        c.drawString(200, y, f"Shoulder Slope: {stat_met['shoulder_slope']:.1f} deg")
        y -= 20
        if 'knee_angle' in stat_met: c.drawString(60, y, f"Knee Angle: {stat_met['knee_angle']:.1f} deg")
        if 'hip_angle' in stat_met: c.drawString(200, y, f"Hip Angle: {stat_met['hip_angle']:.1f} deg")

    y -= 40; c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "AI Feedback")
    y -= 20; c.setFont("Helvetica", 10)
    c.drawString(60, y, "Please see the app screen for detailed feedback.")
    
    c.showPage(); c.save(); b.seek(0)
    return b

# --- メインロジック分岐 ---

# A. 静止画分析モード
if app_mode == "静止画：姿勢分析 (立位/座位)":
    st.info("📸 正面または側面の写真をアップロードしてください")
    
    posture_type = st.radio("分析対象の姿勢を選んでください", ["立位 (Standing)", "座位 (Sitting)"], horizontal=True)
    
    c1, c2 = st.columns(2)
    with c1:
        img_file = st.file_uploader("写真 (正面/側面)", type=['jpg', 'png', 'jpeg'])
    
    if img_file and st.button("🚀 姿勢分析を実行"):
        image = np.array(Image.open(img_file))
        annotated_img, metrics = analyze_static_image(image, posture_type)
        
        if metrics:
            st.image(annotated_img, caption="解析結果", use_container_width=True)
            
            # 結果表示
            st.subheader("📊 姿勢アライメントデータ")
            d1, d2 = st.columns(2)
            with d1:
                st.metric("頭部の傾き", f"{metrics['head_tilt']:.1f}°")
                st.metric("肩の傾き", f"{metrics['shoulder_slope']:.1f}°")
            with d2:
                if posture_type == "立位 (Standing)":
                    st.metric("膝伸展角度", f"{metrics['knee_angle']:.1f}°", help="180に近いほど真っ直ぐ")
                    st.metric("体幹の前傾", f"{metrics['trunk_lean']:.1f}°")
                else:
                    st.metric("股関節角度", f"{metrics['hip_angle']:.1f}°", help="座り姿勢の深さ")
                    st.metric("膝屈曲角度", f"{metrics['knee_angle']:.1f}°")

            st.header("👨‍⚕️ AI姿勢フィードバック")
            feedbacks = generate_static_feedback(metrics, posture_type)
            for msg in feedbacks:
                if "⚠️" in msg: st.error(msg)
                elif "ℹ️" in msg: st.warning(msg)
                else: st.success(msg)

            # PDF
            pdf = create_unified_pdf("Posture Analysis", client_name, feedbacks, stat_met=metrics)
            st.download_button("📄 レポート保存", pdf, "posture_report.pdf", "application/pdf")
        else:
            st.error("人物が検出されませんでした。")

# B. 動画分析モード (Pro / Lite)
else:
    c1, c2 = st.columns(2)
    with c1: file_f = st.file_uploader("正面動画", type=['mp4', 'mov'])
    with c2: file_s = st.file_uploader("側面動画", type=['mp4', 'mov'])

    if st.button("🚀 歩行分析を実行"):
        path_f, met_f = process_video(file_f)
        path_s, met_s = process_video(file_s)
        
        main_met = met_s if met_s else met_f
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1: 
            if path_f: st.video(path_f)
        with c2: 
            if path_s: st.video(path_s)

        if main_met:
            st.metric("ケイデンス", f"{main_met['cadence']:.1f} 歩/分")
            st.success(f"検出歩数: {main_met['steps']}歩")
            
            # 簡易フィードバック (動画用)
            fb = []
            if main_met['cadence'] < 100: fb.append("ℹ️ ピッチがゆっくりです。リズムを意識しましょう。")
            else: fb.append("✅ 良好な歩行リズムです。")
            
            for msg in fb: st.info(msg)
            
            pdf = create_unified_pdf("Gait Analysis", client_name, fb, vid_met=main_met)
            st.download_button("📄 レポート保存", pdf, "gait_report.pdf", "application/pdf")
