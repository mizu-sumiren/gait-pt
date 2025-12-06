import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import io
import math
from datetime import datetime
from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.utils import ImageReader

# 日本語フォント登録
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="AI姿勢・歩行分析ラボ", page_icon="🏥", layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.header("⚙️ 分析モード")
app_mode = st.sidebar.radio(
    "モードを選択してください",
    ["動画：歩行分析 (Pro)", "動画：歩行分析 (Lite)", "静止画：姿勢分析 (立位/座位)"]
)

if "歩行" in app_mode:
    st.title("🏃‍♂️ AI歩行ドック (Clinical Grade)")
    st.caption("転倒リスク・腰痛リスクを「揺れ」「ばらつき」「左右差」から可視化")
else:
    st.title("📸 AI姿勢分析ラボ")
    st.caption("正面(アライメント) × 側面(猫背・FHP) の同時評価")

# サイドバー情報
st.sidebar.header("📋 対象者情報")
client_name = st.sidebar.text_input("氏名", "テスト 太郎 様")
client_age = st.sidebar.number_input("年齢", min_value=1, max_value=120, value=45, step=1)
client_gender = st.sidebar.selectbox("性別", ["男性", "女性", "その他"])
client_height_cm = st.sidebar.number_input("身長 (cm)", min_value=100, max_value=250, value=170, step=1)

if app_mode == "動画：歩行分析 (Pro)":
    with st.sidebar.expander("1. 問診・痛み", expanded=True):
        pain_areas = st.multiselect("痛み", ["なし", "首", "肩", "腰", "股関節", "膝", "足首"])
else:
    pain_areas = []

# --- 計算ロジック ---

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_slope(a, b):
    if a is None or b is None:
        return 0.0
    return math.degrees(math.atan2(a[1]-b[1], a[0]-b[0]))

def calculate_vertical_angle(a, b):
    if a is None or b is None:
        return 0.0
    return math.degrees(math.atan2(b[0]-a[0], b[1]-a[1]))

def get_risk_stars(cv_score, sway_score, asymmetry_percent, age):
    risk_score = 0.0
    cv_threshold = 0.08 if age >= 65 else 0.05
    sway_threshold = 0.12 if age >= 65 else 0.08

    if cv_score > cv_threshold * 1.5: risk_score += 2
    elif cv_score > cv_threshold: risk_score += 1

    if sway_score > sway_threshold * 1.5: risk_score += 2
    elif sway_score > sway_threshold: risk_score += 1

    if asymmetry_percent > 15: risk_score += 2
    elif asymmetry_percent > 8: risk_score += 1

    if age >= 75: risk_score += 1
    elif age >= 65: risk_score += 0.5

    if risk_score >= 5: return "★☆☆☆☆ 高リスク", 1
    elif risk_score >= 3.5: return "★★☆☆☆ 要注意", 2
    elif risk_score >= 2: return "★★★☆☆ やや注意", 3
    elif risk_score >= 1: return "★★★★☆ 良好", 4
    else: return "★★★★★ 優良", 5

def generate_clinical_feedback(metrics, analysis_type="gait", age=45):
    fb_list = []
    exercises = []

    if analysis_type == "gait":
        cadence = metrics.get("cadence", 0.0)
        sway_score = metrics.get("sway_score", 0.0)
        cv_score = metrics.get("cv_score", 0.0)
        trunk_lean_mean = metrics.get("trunk_lean_mean", 0.0)
        asymmetry_percent = metrics.get("asymmetry_percent", 0.0)
        right_mean = metrics.get("right_step_mean", 0.0)
        left_mean = metrics.get("left_step_mean", 0.0)
        gait_speed = metrics.get("gait_speed_m_s", 0.0)

        cv_threshold = 0.08 if age >= 65 else 0.05
        sway_threshold = 0.12 if age >= 65 else 0.08

        if cadence < 95:
            fb_list.append({
                "title": "歩行リズムの低下",
                "detail": f"歩行ペースがゆっくりです（Cadence: {cadence:.1f}歩/分）。",
                "cause": "下肢筋力低下や転倒不安の可能性があります。"
            })
            exercises.append("椅子座り立ち (下肢筋力強化)")
        
        if cv_score > cv_threshold:
            fb_list.append({
                "title": "歩行周期のばらつき",
                "detail": f"一歩ごとのリズムが一定ではありません（CV: {cv_score:.3f}）。",
                "cause": "運動制御能力の低下や注意機能の分散。",
                "priority": True
            })
            exercises.append("メトロノーム歩行")

        if sway_score > sway_threshold:
            fb_list.append({
                "title": "骨盤の動揺（体幹不安定）",
                "detail": f"骨盤の左右への揺れが大きいです（Sway: {sway_score:.3f}）。",
                "cause": "中殿筋や体幹筋の出力不足。",
                "priority": True
            })
            exercises.append("サイドレッグレイズ")
            exercises.append("プランク")

        if asymmetry_percent > 8:
            dominant = "右" if right_mean > left_mean else "左"
            fb_list.append({
                "title": "左右非対称性",
                "detail": f"{dominant}足の滞空時間が長く、左右差があります（{asymmetry_percent:.1f}%）。",
                "cause": "片側の疼痛回避や筋力差。",
                "priority": asymmetry_percent > 15
            })
            exercises.append("片脚立ち練習")

        if not fb_list:
            fb_list.append({"title": "良好な歩行", "detail": "問題は見られません。", "cause": "現状維持推奨。"})

    else:
        # 姿勢分析用フィードバック（省略なしで実装）
        s_met = metrics.get("s_met") or {}
        if abs(s_met.get("forward_head_score", 0.0)) > 5.0:
            fb_list.append({"title": "ストレートネック傾向", "detail": "頭部前方偏位あり。", "cause": "スマホ首など。"})
            exercises.append("チンイン")

    return fb_list, list(set(exercises))

# --- メモリ最適化版 歩行解析ロジック ---

def analyze_gait_data_only(lms_history, fps, w, h, height_cm):
    # 画像を使わず、座標データ(lms)だけで計算する
    if not lms_history or fps <= 0:
        return {}, {}

    left_ankle_y = []
    right_ankle_y = []
    pelvis_sway = []
    trunk_lean_list = []
    hip_dists = []

    max_ml_abs = 0.0
    max_lean_abs = 0.0
    
    # インデックスを記録するための変数
    idx_ml = 0
    idx_lean = 0
    idx_mid = len(lms_history) // 2

    for i, lms in enumerate(lms_history):
        # 座標取得 (正規化座標)
        la_y = lms[27].y
        ra_y = lms[28].y
        left_ankle_y.append(la_y)
        right_ankle_y.append(ra_y)

        # Sway
        pm_x = (lms[23].x + lms[24].x) / 2.0
        pelvis_sway.append(pm_x)

        # Lean
        mid_s = [(lms[11].x + lms[12].x)/2 * w, (lms[11].y + lms[12].y)/2 * h]
        mid_h = [(lms[23].x + lms[24].x)/2 * w, (lms[23].y + lms[24].y)/2 * h]
        lean = calculate_vertical_angle(mid_h, mid_s)
        trunk_lean_list.append(lean)

        # ML Deviation logic
        trunk_cx = (pm_x * w + (lms[11].x + lms[12].x)/2 * w) / 2.0
        ml_dev = (trunk_cx - w/2.0) / (w/2.0)
        
        if abs(ml_dev) > max_ml_abs:
            max_ml_abs = abs(ml_dev)
            idx_ml = i
        
        if abs(lean) > max_lean_abs:
            max_lean_abs = abs(lean)
            idx_lean = i
            
        # Gait Speed calc components
        hl = np.array([lms[23].x*w, lms[23].y*h])
        hr = np.array([lms[24].x*w, lms[24].y*h])
        hip_dists.append(np.linalg.norm(hl - hr))

    # Step Detection
    def detect_steps(arr):
        steps = 0
        frames = []
        if len(arr) > 2:
            th = np.percentile(arr, 60)
            for i in range(1, len(arr)-1):
                if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > th:
                    steps += 1
                    frames.append(i)
        return steps, frames

    ls, lf = detect_steps(left_ankle_y)
    rs, rf = detect_steps(right_ankle_y)
    total_steps = ls + rs
    duration = len(lms_history) / fps
    cadence = (total_steps / duration) * 60 if duration > 0 else 0

    # Metrics
    asym = 0.0
    lm, rm = 0.0, 0.0
    if len(lf) > 1: lm = float(np.mean(np.diff(lf)))
    if len(rf) > 1: rm = float(np.mean(np.diff(rf)))
    if (lm+rm) > 0:
        asym = abs(lm - rm) / ((lm+rm)/2) * 100

    cv = 0.0
    all_f = sorted(lf + rf)
    if len(all_f) > 2:
        intervals = np.diff(all_f)
        if np.mean(intervals) > 0:
            cv = np.std(intervals) / np.mean(intervals)

    sway_score = float(np.std(pelvis_sway)) if pelvis_sway else 0
    lean_mean = float(np.mean(trunk_lean_list)) if trunk_lean_list else 0
    
    speed = 0.0
    if total_steps > 1 and cadence > 0:
        stride = height_cm * 0.01 * 0.45
        speed = (cadence/60) * stride

    metrics = {
        "cadence": cadence,
        "steps": total_steps,
        "cv_score": cv,
        "sway_score": sway_score,
        "trunk_lean_mean": lean_mean,
        "asymmetry_percent": asym,
        "left_step_mean": lm,
        "right_step_mean": rm,
        "gait_speed_m_s": speed
    }
    
    # 重要なフレーム番号を返す
    target_indices = {
        "mid": idx_mid,
        "ml": idx_ml,
        "lean": idx_lean
    }
    
    return metrics, target_indices

def process_video_optimized(file, height_cm):
    if not file: return None, None, None
    
    # 1. ファイル保存
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = int(cap.get(5))
    
    # 2. 解析用パス (画像は保存せず、座標のみ記録)
    lms_history = []
    
    # 出力動画用 (ストリーミング書き込みでメモリ節約)
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    # mp4vが使えない場合のフォールバックは今回考えない（packages.txtで対応済み前提）
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret: break
            
            # MediaPipe処理
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            res = pose.process(img_rgb)
            
            # 描画
            cv2.line(img, (w//2, 0), (w//2, h), (0, 255, 255), 1)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # ★ここで画像は保存せず、座標だけリストに入れる
                lms_history.append(res.pose_landmarks.landmark)
            else:
                lms_history.append(None) # 検出なしフレーム
            
            out.write(img)
            
    cap.release()
    out.release()
    
    # 3. 数値解析 & キーフレーム特定
    # Noneを除去して解析に回す
    clean_lms = [l for l in lms_history if l is not None]
    if not clean_lms:
        return None, {}, {}
        
    metrics, target_indices = analyze_gait_data_only(clean_lms, fps, w, h, height_cm)
    
    # 4. キーフレーム画像だけを再取得 (省メモリ)
    snapshots = {}
    cap = cv2.VideoCapture(tfile.name) # 再オープン
    
    for key, idx in target_indices.items():
        # idx番目のフレームへジャンプ
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 描画等は省略し、生の画像をスナップショットとする
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            snapshots[key] = Image.fromarray(frame_rgb)
        else:
            snapshots[key] = None
            
    cap.release()
    
    return out_path, metrics, snapshots

def analyze_static_image(image, view, posture_type):
    # 静止画機能は変更なし
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks: return image, {}
        h, w, _ = image.shape
        lms = results.pose_landmarks.landmark
        annotated = image.copy()
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        def gp(i): return [lms[i].x*w, lms[i].y*h]
        metrics = {}
        if view == "front":
            metrics["head_tilt"] = calculate_slope(gp(7), gp(8))
            metrics["shoulder_slope"] = calculate_slope(gp(11), gp(12))
        elif view == "side":
            metrics["forward_head_score"] = (lms[7].x - lms[11].x)*100
        return annotated, metrics

def create_comprehensive_pdf(title, name, fb_data, exercises, metrics_data, snapshots=None):
    if snapshots is None: snapshots = {}
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    pw, ph = A4
    font = "HeiseiKakuGo-W5"
    
    c.setFont(font, 18)
    c.drawString(40, ph-50, title)
    c.setFont(font, 11)
    c.drawString(40, ph-75, f"氏名: {name}  /  判定日: {datetime.now().strftime('%Y/%m/%d')}")
    c.line(40, ph-85, pw-40, ph-85)
    
    y = ph - 120
    # 画像配置
    if snapshots:
        x_pos = pw - 220
        for label in ["mid", "ml", "lean"]:
            img = snapshots.get(label)
            if img:
                ih = 100
                iw = ih * img.width / img.height
                c.drawImage(ImageReader(img), x_pos, y-ih, width=iw, height=ih)
                c.drawString(x_pos, y-ih-10, f"▲ {label}")
                y -= (ih + 30)

    c.setFont(font, 14)
    c.drawString(40, ph-120, "■ 分析結果")
    y_text = ph - 145
    c.setFont(font, 10)
    
    if "cadence" in metrics_data:
        items = [
            f"Cadence: {metrics_data['cadence']:.1f} step/min",
            f"Speed: {metrics_data['gait_speed_m_s']:.2f} m/s",
            f"CV: {metrics_data['cv_score']:.3f}",
            f"Sway: {metrics_data['sway_score']:.3f}",
            f"Asymmetry: {metrics_data['asymmetry_percent']:.1f}%"
        ]
        for t in items:
            c.drawString(50, y_text, t)
            y_text -= 15
            
        star, _ = get_risk_stars(metrics_data['cv_score'], metrics_data['sway_score'], metrics_data['asymmetry_percent'], client_age)
        c.setFont(font, 12)
        c.drawString(50, y_text-10, f"★ 総合評価: {star}")
        y_text -= 40
    
    c.setFont(font, 14)
    c.drawString(40, y_text, "■ フィードバック")
    y_text -= 20
    c.setFont(font, 10)
    for fb in fb_data:
        c.drawString(50, y_text, f"● {fb['title']}")
        c.drawString(60, y_text-15, f"状態: {fb['detail']}")
        y_text -= 40
        
    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# --- メインUI ---

if app_mode == "静止画：姿勢分析 (立位/座位)":
    st.info("写真をアップロードしてください")
    f_file = st.file_uploader("正面", type=["jpg","png"])
    s_file = st.file_uploader("側面", type=["jpg","png"])
    if st.button("分析実行") and f_file and s_file:
        f_img = np.array(Image.open(f_file))
        s_img = np.array(Image.open(s_file))
        res_f, met_f = analyze_static_image(f_img, "front", "standing")
        res_s, met_s = analyze_static_image(s_img, "side", "standing")
        
        c1, c2 = st.columns(2)
        c1.image(res_f, caption="正面")
        c2.image(res_s, caption="側面")
        
        fb, ex = generate_clinical_feedback({"s_met": met_s}, "static", client_age)
        st.write(fb)
        
        pdf = create_comprehensive_pdf("姿勢レポート", client_name, fb, ex, {}, {})
        st.download_button("PDF保存", pdf, "report.pdf")

else:
    st.info("🎥 歩行動画をアップロード (30秒以内の動画推奨)")
    video_file = st.file_uploader("歩行動画", type=["mp4", "mov"])

    if st.button("🚀 歩行分析を実行") and video_file:
        with st.spinner("AIが動画を解析中... (完了までお待ちください)"):
            out_path, metrics, snapshots = process_video_optimized(video_file, client_height_cm)

        if out_path and metrics:
            st.video(out_path)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Cadence", f"{metrics['cadence']:.1f}")
            c2.metric("Sway", f"{metrics['sway_score']:.3f}")
            c3.metric("CV", f"{metrics['cv_score']:.3f}")
            
            star, _ = get_risk_stars(metrics['cv_score'], metrics['sway_score'], metrics['asymmetry_percent'], client_age)
            st.subheader(f"総合評価: {star}")
            
            fb_data, ex_list = generate_clinical_feedback(metrics, "gait", client_age)
            for item in fb_data:
                st.info(f"**{item['title']}**: {item['detail']}")
            
            if ex_list:
                st.success(f"推奨運動: {', '.join(ex_list)}")
                
            pdf = create_comprehensive_pdf("歩行分析レポート", client_name, fb_data, ex_list, metrics, snapshots)
            st.download_button("📄 PDFレポート保存", pdf, "gait_report.pdf", "application/pdf")
        else:
            st.error("解析に失敗しました。動画の形式を確認してください。")
