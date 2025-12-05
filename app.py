import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import io
import math
from PIL import Image

# --- PDF生成用ライブラリ ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# --- 日本語フォント登録 (PDF用) ---
# HeiseiKakuGo-W5 は多くのPDFリーダーで標準的に使える日本語フォントです
pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))

# --- MediaPipe初期化 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- ページ設定 ---
st.set_page_config(page_title="AI姿勢・歩行分析ラボ", page_icon="🏥", layout="wide")

# --- CSS設定 (メニュー非表示など) ---
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
    st.markdown("正面(アライメント) × 側面(猫背・FHP) の同時評価")

# --- サイドバー入力 ---
st.sidebar.header("📋 対象者情報")
client_name = st.sidebar.text_input("氏名", "テスト 太郎 様")

# Proモードの場合のみ詳細入力 (表示のみで現在はロジックには影響しません)
if app_mode == "動画：歩行分析 (Pro)":
    with st.sidebar.expander("1. 問診・痛み", expanded=True):
        pain_areas = st.multiselect("痛み", ["なし", "首", "肩", "腰", "股関節", "膝", "足首"])
    with st.sidebar.expander("2. 身体機能測定", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**左 (L)**")
            grip_l = st.number_input("握力L", 20.0); hip_flex_l = st.number_input("股屈曲L", 0.9)
        with c2:
            st.markdown("**右 (R)**")
            grip_r = st.number_input("握力R", 25.0); hip_flex_r = st.number_input("股屈曲R", 1.2)

# --- 幾何学計算関数 ---
def calculate_angle(a, b, c):
    """3点間の角度を算出"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def calculate_slope(a, b):
    """2点間の傾き"""
    if a is None or b is None: return 0
    return math.degrees(math.atan2(a[1]-b[1], a[0]-b[0]))

def calculate_vertical_angle(a, b):
    """垂直線に対する角度"""
    if a is None or b is None: return 0
    return math.degrees(math.atan2(b[0]-a[0], b[1]-a[1]))

# --- 静止画分析ロジック ---
def analyze_static_image(image, view, posture_type):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks: return image, None

        h, w, _ = image.shape
        lms = results.pose_landmarks.landmark
        annotated_image = image.copy()
        
        # グリッド線
        cv2.line(annotated_image, (w//2, 0), (w//2, h), (0, 255, 255), 2)
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        def get_p(idx): return [lms[idx].x * w, lms[idx].y * h]
        metrics = {}

        # --- A. 正面写真の分析 ---
        if view == "front":
            metrics['head_tilt'] = calculate_slope(get_p(7), get_p(8)) # 耳の傾き
            metrics['shoulder_slope'] = calculate_slope(get_p(11), get_p(12)) # 肩の傾き
            metrics['hip_slope'] = calculate_slope(get_p(23), get_p(24)) # 骨盤

        # --- B. 側面写真の分析 ---
        elif view == "side":
            # スマホ首 (耳7と肩11のX差を画像幅に対する%で)
            ear_x = (lms[7].x + lms[8].x) / 2 
            shoulder_x = (lms[11].x + lms[12].x) / 2
            metrics['forward_head_score'] = (ear_x - shoulder_x) * 100 
            
            # 体幹の前傾
            metrics['trunk_lean'] = calculate_vertical_angle(get_p(11), get_p(23))
            
            # 膝・股関節
            if posture_type == "立位 (Standing)":
                metrics['knee_angle'] = calculate_angle(get_p(23), get_p(25), get_p(27))
            else: # 座位
                metrics['hip_angle'] = calculate_angle(get_p(11), get_p(23), get_p(25))

        return annotated_image, metrics

# --- 静止画フィードバック生成 ---
def generate_static_feedback(f_metrics, s_metrics, posture_type):
    fb = []
    
    # 正面
    if f_metrics:
        if abs(f_metrics['head_tilt']) > 3.0: fb.append("⚠️ 【頭部の傾き】 正面から見て首が傾いています。")
        slope = f_metrics['shoulder_slope']
        if abs(slope) > 3.0: 
            side = "右" if slope > 0 else "左"
            fb.append(f"⚠️ 【肩の高さ】 {side}肩が下がっています。")
    
    # 側面
    if s_metrics:
        if abs(s_metrics['forward_head_score']) > 5.0: 
            fb.append("⚠️ 【ストレートネック傾向】 頭が肩より前に出ています（スマホ首）。")
        
        if abs(s_metrics['trunk_lean']) > 10: 
            fb.append("⚠️ 【猫背・反り腰】 上半身の軸が垂直から傾いています。")

        if posture_type == "立位 (Standing)":
            if s_metrics.get('knee_angle', 180) < 165: fb.append("ℹ️ 【膝曲がり】 膝が伸び切っていません。")
        else:
            if s_metrics.get('hip_angle', 90) > 110: fb.append("ℹ️ 【仙骨座り】 骨盤が後ろに倒れ、腰への負担が大きい座り方です。")

    if not fb: fb.append("✅ 【Good】 非常に綺麗な姿勢アライメントです。")
    return fb

# --- 動画分析関数 (修正版) ---
def analyze_video_metrics(history, fps):
    if not history: return None
    dists = []
    # 足首間距離を用いた簡易歩数カウント
    for lms in history:
        la, ra = np.array([lms[27].x, lms[27].y]), np.array([lms[28].x, lms[28].y])
        dists.append(np.linalg.norm(la - ra))
    
    steps = 0
    if len(dists) > 0:
        thresh = np.mean(dists)
        for i in range(1, len(dists)-1):
            if dists[i] > dists[i-1] and dists[i] > dists[i+1] and dists[i] > thresh: 
                steps += 1
                
    duration = len(history) / fps
    cadence = (steps / duration) * 60 if duration > 0 else 0
    return {"cadence": cadence, "steps": steps}

def process_video(file):
    """動画処理メイン関数 (色修正・数値描画追加済み)"""
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
            
            # 1. 解析用: BGR -> RGB変換
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            res = pose.process(img_rgb)
            
            # 2. 書き込み用: img (BGRのまま) を使用 ※ここが色修正のポイント
            
            # ガイドライン
            cv2.line(img, (w//2,0), (w//2,h), (0,255,255), 1)
            
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lms = res.pose_landmarks.landmark
                history.append(lms)
                
                # --- PT視点: 右膝角度のリアルタイム計算と表示 ---
                def get_c(idx): return [lms[idx].x * w, lms[idx].y * h]
                try:
                    # 右股関節(24)-右膝(26)-右足首(28)
                    knee_angle = calculate_angle(get_c(24), get_c(26), get_c(28))
                    
                    # 画面右上に角度表示パネルを描画
                    cv2.rectangle(img, (w-220, 0), (w, 60), (255, 255, 255), -1)
                    cv2.putText(img, f"R-Knee: {int(knee_angle)}", (w-200, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except:
                    pass

            out.write(img)
            
    cap.release(); out.release()
    return path, analyze_video_metrics(history, fps)

# --- PDF生成 (修正版・日本語対応) ---
def create_pdf(title, name, feedbacks, vid=None, f_stat=None, s_stat=None):
    b = io.BytesIO()
    c = canvas.Canvas(b, pagesize=A4); h = A4[1]
    
    # フォント設定 (HeiseiKakuGo-W5)
    font_name = "HeiseiKakuGo-W5"
    
    # ヘッダー
    c.setFont(font_name, 18)
    c.drawString(50, h-50, f"分析レポート: {title}")
    c.setFont(font_name, 12)
    c.drawString(50, h-80, f"氏名: {name}")
    c.setLineWidth(1)
    c.line(50, h-90, 550, h-90)
    
    y = h-130
    # --- データセクション ---
    c.setFont(font_name, 14); c.drawString(50, y, "1. 計測データ (Metrics)")
    y -= 25; c.setFont(font_name, 11)
    
    if vid:
        c.drawString(60, y, f"・ケイデンス (歩行リズム): {vid['cadence']:.1f} 歩/分")
        y -= 20
        c.drawString(60, y, f"・検出歩数: {vid['steps']} 歩")
        y -= 30
    
    if f_stat:
        c.drawString(60, y, "[正面分析結果]")
        y -= 20
        c.drawString(70, y, f"・頭部の傾き: {f_stat['head_tilt']:.1f} 度")
        c.drawString(300, y, f"・肩の傾き: {f_stat['shoulder_slope']:.1f} 度")
        y -= 30
        
    if s_stat:
        c.drawString(60, y, "[側面分析結果]")
        y -= 20
        c.drawString(70, y, f"・FHPスコア(頭の前方偏位): {s_stat['forward_head_score']:.1f}")
        c.drawString(300, y, f"・体幹前傾角度: {s_stat['trunk_lean']:.1f} 度")
        y -= 30

    y -= 20
    # --- フィードバックセクション (日本語出力対応) ---
    c.setFont(font_name, 14); c.drawString(50, y, "2. AIフィードバック & アドバイス")
    y -= 30; c.setFont(font_name, 11)
    
    for msg in feedbacks:
        # 簡易クリーニング
        clean_msg = msg.replace("**", "")
        c.drawString(60, y, clean_msg)
        y -= 25
        if y < 50: # 改ページ処理
            c.showPage()
            c.setFont(font_name, 11)
            y = h-50
            
    c.showPage(); c.save(); b.seek(0)
    return b

# --- メインロジック ---

# A. 静止画分析モード
if app_mode == "静止画：姿勢分析 (立位/座位)":
    st.info("📸 正面・側面それぞれの写真をアップロードしてください（片方のみも可）")
    posture_type = st.radio("姿勢タイプ", ["立位 (Standing)", "座位 (Sitting)"], horizontal=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("① 正面写真")
        file_f = st.file_uploader("Front Image", type=['jpg','png','jpeg'], key="sf")
    with c2:
        st.subheader("② 側面写真")
        file_s = st.file_uploader("Side Image", type=['jpg','png','jpeg'], key="ss")
    
    if st.button("🚀 姿勢分析を実行"):
        f_img, f_met, s_img, s_met = None, None, None, None
        
        # 1. 正面分析
        if file_f:
            img = np.array(Image.open(file_f))
            f_img, f_met = analyze_static_image(img, "front", posture_type)
        
        # 2. 側面分析
        if file_s:
            img = np.array(Image.open(file_s))
            s_img, s_met = analyze_static_image(img, "side", posture_type)
            
        if f_met or s_met:
            # 画像表示
            col1, col2 = st.columns(2)
            with col1:
                if f_img is not None: st.image(f_img, caption="正面解析", use_container_width=True)
            with col2:
                if s_img is not None: st.image(s_img, caption="側面解析", use_container_width=True)

            # データ表示
            st.subheader("📊 アライメント計測値")
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("##### 正面データ")
                if f_met:
                    st.metric("頭部の傾き", f"{f_met['head_tilt']:.1f}°")
                    st.metric("肩の傾き", f"{f_met['shoulder_slope']:.1f}°")
                else: st.caption("データなし")
            with d2:
                st.markdown("##### 側面データ")
                if s_met:
                    st.metric("体幹前傾", f"{s_met['trunk_lean']:.1f}°")
                    val = s_met.get('knee_angle') if posture_type == "立位 (Standing)" else s_met.get('hip_angle')
                    label = "膝伸展" if posture_type == "立位 (Standing)" else "股関節屈曲"
                    st.metric(label, f"{val:.1f}°")
                else: st.caption("データなし")
            
            # フィードバック
            st.header("👨‍⚕️ AI姿勢レポート")
            feedbacks = generate_static_feedback(f_met, s_met, posture_type)
            for msg in feedbacks:
                if "⚠️" in msg: st.error(msg)
                elif "ℹ️" in msg: st.warning(msg)
                else: st.success(msg)

            # 保存
            pdf = create_pdf("姿勢分析 (Posture Analysis)", client_name, feedbacks, f_stat=f_met, s_stat=s_met)
            st.download_button("📄 レポート保存 (PDF)", pdf, "posture_report.pdf", "application/pdf")
            
        else:
            st.warning("写真をアップロードしてください")

# B. 動画分析モード
else:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("① 正面動画")
        file_f = st.file_uploader("Front Video", type=['mp4', 'mov'], key="vf")
    with c2:
        st.subheader("② 側面動画")
        file_s = st.file_uploader("Side Video", type=['mp4', 'mov'], key="vs")

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
            st.subheader("📊 歩行データ")
            st.metric("ケイデンス (歩行率)", f"{main_met['cadence']:.1f} 歩/分")
            st.info(f"検出歩数: {main_met['steps']}歩")
            
            fb = ["✅ 解析完了。詳細はPDFをご確認ください。"]
            if main_met['cadence'] < 100: fb.append("ℹ️ 歩行ペースがややゆっくりです。")
            if main_met['cadence'] > 120: fb.append("ℹ️ 早歩き傾向、または小刻み歩行の可能性があります。")
            
            # PDF用フィードバック
            pdf_fb = fb
            
            pdf = create_pdf("歩行分析 (Gait Analysis)", client_name, pdf_fb, vid=main_met)
            st.download_button("📄 レポート保存 (PDF)", pdf, "gait_report.pdf", "application/pdf")
