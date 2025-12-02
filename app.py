import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# --- MediaPipe初期化 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- ページ設定 ---
# initial_sidebar_state="expanded" を追加して、PCでは最初から開くように設定
st.set_page_config(page_title="AI歩行解析アプリ", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- 画面設定：修正箇所 ---
# header {visibility: hidden;} を削除しました。これで左上の「>」ボタンが表示されます。
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            /* header {visibility: hidden;} ←ここを削除しました */
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🏃‍♂️ AI歩行ドック - Smart Gait Lab")
st.markdown("身体機能評価 × AI歩行分析 × 自動レポート生成")

# --- サイドバー：詳細な機能チェック ---
st.sidebar.header("📋 測定データ入力")

with st.sidebar.expander("1. 基本情報・問診", expanded=True):
    client_name = st.text_input("氏名", "テスト 太郎 様")
    pain_areas = st.multiselect(
        "痛み・違和感のある部位",
        ["特になし", "首", "肩", "腰", "股関節(右)", "股関節(左)", "膝(右)", "膝(左)", "足首・足部"]
    )

with st.sidebar.expander("2. 身体機能測定結果", expanded=True):
    st.caption("対象者の数値を入力してください")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**左側 (Left)**")
        grip_l = st.number_input("握力(左) kg", value=20.0)
        hip_flex_l = st.number_input("股屈曲(左) kgf/kg", value=0.9)
        one_leg_l = st.number_input("片脚立位(左) 秒", value=15.0)
        toe_grip_l = st.number_input("足趾把持(左) %", value=10.0)
    with col_s2:
        st.markdown("**右側 (Right)**")
        grip_r = st.number_input("握力(右) kg", value=25.0)
        hip_flex_r = st.number_input("股屈曲(右) kgf/kg", value=1.2)
        one_leg_r = st.number_input("片脚立位(右) 秒", value=60.0)
        toe_grip_r = st.number_input("足趾把持(右) %", value=20.0)

    st.markdown("---")
    frt = st.number_input("FRT (cm)", value=25.0)
    ffd = st.number_input("FFD (cm)", value=0.0)
    seat_step = st.number_input("座位ステップ (回/20秒)", value=30)

# --- 関数：歩行指標（ケイデンス・歩幅）の計算 ---
def analyze_gait_metrics(landmarks_history, fps):
    if not landmarks_history:
        return None

    ankle_distances = []
    shin_lengths = []

    for lms in landmarks_history:
        la = np.array([lms[27].x, lms[27].y])
        ra = np.array([lms[28].x, lms[28].y])
        lk = np.array([lms[25].x, lms[25].y])
        
        dist = np.linalg.norm(la - ra)
        ankle_distances.append(dist)
        
        shin_len = np.linalg.norm(lk - la)
        shin_lengths.append(shin_len)

    steps = 0
    peaks = []
    threshold = np.mean(ankle_distances)
    
    for i in range(1, len(ankle_distances)-1):
        prev = ankle_distances[i-1]
        curr = ankle_distances[i]
        nex = ankle_distances[i+1]
        if curr > prev and curr > nex and curr > threshold:
            steps += 1
            peaks.append(curr)

    duration_sec = len(landmarks_history) / fps
    cadence = (steps / duration_sec) * 60 if duration_sec > 0 else 0
    
    avg_step_pixel = np.mean(peaks) if peaks else 0
    avg_shin_pixel = np.mean(shin_lengths) if shin_lengths else 1
    normalized_step_length = avg_step_pixel / avg_shin_pixel

    return {
        "steps": steps,
        "duration": duration_sec,
        "cadence": cadence,
        "step_ratio": normalized_step_length
    }

# --- 関数：PDFレポート生成 ---
def create_pdf(client_name, data, feedbacks, gait_metrics):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    try:
        # 日本語フォントがあれば設定（なければ英語で代用）
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, f"Gait & Physical Analysis Report")
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, "Note: Japanese font required for full text support.")
    except:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Analysis Report")

    y = height - 100
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Name: {client_name}")
    y -= 20
    c.drawString(50, y, f"Date: 2025/12/03")
    
    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "1. Gait Analysis (AI Video)")
    y -= 20
    c.setFont("Helvetica", 11)
    if gait_metrics:
        c.drawString(60, y, f"Cadence: {gait_metrics['cadence']:.1f} steps/min")
        c.drawString(250, y, f"Step Ratio: {gait_metrics['step_ratio']:.2f} (Step/Leg Length)")
    else:
        c.drawString(60, y, "No video data analyzed.")

    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "2. Physical Functions")
    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(60, y, f"Toe Grip: L {data['toe_l']}% / R {data['toe_r']}%")
    c.drawString(250, y, f"One Leg Stand: L {data['ols_l']}s / R {data['ols_r']}s")
    y -= 20
    c.drawString(60, y, f"Hip Flexion: L {data['hip_l']} / R {data['hip_r']}")
    c.drawString(250, y, f"FRT: {data['frt']}cm  /  FFD: {data['ffd']}cm")

    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "3. AI PT Feedback")
    c.setFont("Helvetica", 10)
    y -= 20
    c.drawString(60, y, "Please refer to the app screen for detailed feedback.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --- ロジック関数：臨床推論エンジン（身体機能×歩行の統合） ---
def generate_clinical_feedback(data):
    feedback = []
    
    # データ展開
    pain = data['pain']
    toe_l, toe_r = data['toe_l'], data['toe_r']
    hip_l, hip_r = data['hip_l'], data['hip_r']
    ols_l, ols_r = data['ols_l'], data['ols_r']
    frt, ffd = data['frt'], data['ffd']
    
    # 1. 足趾機能と歩行の蹴り出し
    avg_toe = (toe_l + toe_r) / 2
    if avg_toe < 15:
        feedback.append(f"**【足指機能低下 × 歩行推進力】**\n足指の力が弱いため（平均{avg_toe:.1f}%）、歩行時に地面を蹴る力が不足しています。これが**「ペタペタ歩き」や「歩幅の減少」**に直結しています。")
    
    # 2. 股関節機能とつまずきリスク
    if hip_l < 1.0 or hip_r < 1.0:
        weak_side = "左" if hip_l < hip_r else "右"
        feedback.append(f"**【股関節筋力低下 × クリアランス】**\n{weak_side}側の腸腰筋出力が低下しています。足を振り出す際に高さが不足し、**「すり足」や「小さな段差でのつまずき」**のリスクが高い状態です。")
    
    # 3. 左右差と代償動作
    diff_hip = abs(hip_l - hip_r)
    if diff_hip > 0.2:
        weaker = "左" if hip_l < hip_r else "右"
        stronger = "右" if hip_l < hip_r else "左"
        feedback.append(f"**【機能左右差 × 歩行リズム】**\n筋力バランスの崩れにより、歩行のリズムが不均等になっています。**弱い{weaker}側を早く接地させようとするため、反対側の{stronger}側に過度な負担**がかかっています。")

    # 4. バランス機能とスウェイ
    if ols_l < 20 or ols_r < 20:
        unstable = "左" if ols_l < 20 else "右"
        feedback.append(f"**【立位バランス低下 × 重心動揺】**\n片脚立位が不安定（{unstable}側）です。歩行の立脚期に骨盤を安定させることができず、**「外側へのふらつき（ラテラルスウェイ）」**が生じています。")

    # 5. 重心移動能力
    if frt < 30:
        feedback.append(f"**【前方重心移動制限 × 後方重心】**\nFRTが{frt}cmと短縮しています。転倒への恐怖心から**「腰が引けた姿勢」**になりやすく、スムーズな体重移動が阻害されています。")

    if not feedback:
        feedback.append("✅ **素晴らしい状態です！**\n身体機能と歩行姿勢の統合分析の結果、目立った崩れは見当たりません。現在の機能を維持しましょう。")

    return feedback

# --- 動画処理関数 ---
def draw_grid_and_skeleton(image, results):
    h, w, _ = image.shape
    color_grid = (200, 200, 200)
    center_x = w // 2
    cv2.line(image, (center_x, 0), (center_x, h), (0, 255, 255), 1) 
    for x in range(0, w, w//8):
        if x != center_x: cv2.line(image, (x, 0), (x, h), color_grid, 1)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
        )
    return image

def process_video_and_analyze(uploaded_file):
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
            
            image = draw_grid_and_skeleton(image, results)
            out.write(image)
            
            if results.pose_landmarks:
                landmarks_history.append(results.pose_landmarks.landmark)

    cap.release()
    out.release()
    
    metrics = analyze_gait_metrics(landmarks_history, fps)
    return output_path, metrics

# --- メインレイアウト ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("① 正面動画")
    file_front = st.file_uploader("Front View", type=['mp4', 'mov'], key="f")
with col2:
    st.subheader("② 側面動画 (分析推奨)")
    file_side = st.file_uploader("Side View", type=['mp4', 'mov'], key="s")

if st.button("🚀 汎用分析を実行"):
    path_f, metrics_f = process_video_and_analyze(file_front)
    path_s, metrics_s = process_video_and_analyze(file_side)
    
    main_metrics = metrics_s if metrics_s else metrics_f
    
    st.markdown("---")
    
    # 1. 結果表示カラム
    res_c1, res_c2 = st.columns([2, 1])
    
    with res_c1:
        st.subheader("🎥 解析動画")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            if path_f: st.video(path_f)
        with v_col2:
            if path_s: st.video(path_s)
            
    with res_c2:
        st.subheader("📊 歩行AIメトリクス")
        if main_metrics:
            st.metric("ケイデンス (歩数/分)", f"{main_metrics['cadence']:.1f}", delta="標準: 110-120")
            st.metric("歩幅比率 (歩幅/下腿長)", f"{main_metrics['step_ratio']:.2f}", help="1.0以上が理想的")
            st.info(f"検出: {main_metrics['steps']}歩 / {main_metrics['duration']:.1f}秒")
        else:
            st.warning("動画からデータ抽出不可")

    # 2. 自動フィードバック生成（身体機能×歩行統合版）
    st.header("👨‍⚕️ AI理学療法士のフィードバック")
    
    input_data = {
        'pain': pain_areas,
        'toe_l': toe_grip_l, 'toe_r': toe_grip_r,
        'hip_l': hip_flex_l, 'hip_r': hip_flex_r,
        'ols_l': one_leg_l, 'ols_r': one_leg_r,
        'frt': frt, 'ffd': ffd, 'seat_step': seat_step
    }
    
    feedbacks = generate_clinical_feedback(input_data)
    
    for msg in feedbacks:
        st.info(msg)

    # 3. 推奨運動 & ダウンロード
    st.subheader("🏋️‍♀️ 推奨運動 & レポート")
    rec_col1, rec_col2 = st.columns([3, 1])
    
    with rec_col1:
        if (toe_grip_l + toe_grip_r)/2 < 15:
            st.markdown("- **足指強化**: タオルギャザー、足指じゃんけん")
        if hip_flex_l < 1.0 or hip_flex_r < 1.0:
            st.markdown("- **腸腰筋強化**: ニーアップ、大股歩き")
        if one_leg_l < 20 or one_leg_r < 20:
            st.markdown("- **中殿筋・バランス**: 片脚立ち保持（1分間）")
        if frt < 30:
            st.markdown("- **動的バランス**: 重心移動練習")
            
    with rec_col2:
        # PDF生成
        pdf_data = create_pdf(client_name, input_data, feedbacks, main_metrics)
        st.download_button(
            label="📄 レポートPDFをDL",
            data=pdf_data,
            file_name=f"{client_name}_Analysis_Report.pdf",
            mime="application/pdf"
        )
        
        st.markdown("---") # 区切り線

        # 2. 解析動画のDL
        if path_s: # 側面動画がある場合
            with open(path_s, 'rb') as v_file:
                st.download_button(
                    label="🎥 解析動画 (側面)を保存",
                    data=v_file,
                    file_name=f"{client_name}_SideView.mp4",
                    mime="video/mp4"
                )
        
        if path_f: # 正面動画がある場合
            with open(path_f, 'rb') as v_file:
                st.download_button(
                    label="🎥 解析動画 (正面)を保存",
                    data=v_file,
                    file_name=f"{client_name}_FrontView.mp4",
                    mime="video/mp4"
                )
