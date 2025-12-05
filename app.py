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

# 共通ユーティリティ
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

    if cv_score > cv_threshold * 1.5:
        risk_score += 2
    elif cv_score > cv_threshold:
        risk_score += 1

    if sway_score > sway_threshold * 1.5:
        risk_score += 2
    elif sway_score > sway_threshold:
        risk_score += 1

    if asymmetry_percent > 15:
        risk_score += 2
    elif asymmetry_percent > 8:
        risk_score += 1

    if age >= 75:
        risk_score += 1
    elif age >= 65:
        risk_score += 0.5

    if risk_score >= 5:
        return "★☆☆☆☆ 高リスク", 1
    elif risk_score >= 3.5:
        return "★★☆☆☆ 要注意", 2
    elif risk_score >= 2:
        return "★★★☆☆ やや注意", 3
    elif risk_score >= 1:
        return "★★★★☆ 良好", 4
    else:
        return "★★★★★ 優良", 5

# フィードバック生成
def generate_clinical_feedback(metrics, analysis_type="gait", age=45):
    fb_list = []
    exercises = []

    if analysis_type == "gait":
        cadence = metrics.get("cadence", 0.0)
        sway_score = metrics.get("sway_score", 0.0)
        cv_score = metrics.get("cv_score", 0.0)
        trunk_lean_mean = metrics.get("trunk_lean_mean", 0.0)
        asymmetry_percent = metrics.get("asymmetry_percent", 0.0)
        left_mean = metrics.get("left_step_mean", 0.0)
        right_mean = metrics.get("right_step_mean", 0.0)
        gait_speed = metrics.get("gait_speed_m_s", 0.0)

        cv_threshold = 0.08 if age >= 65 else 0.05
        sway_threshold = 0.12 if age >= 65 else 0.08

        if cadence < 95:
            fb_list.append({
                "title": "歩行リズムの低下",
                "detail": f"歩行ペースがゆっくりです（Cadence: {cadence:.1f}歩/分、推定速度: {gait_speed:.2f}m/s）。",
                "cause": "下肢筋力の低下や、転倒への不安感が影響している可能性があります。"
            })
            exercises.append("椅子座り立ち (下肢筋力強化)")
        elif cadence > 125:
            fb_list.append({
                "title": "小刻み歩行の傾向",
                "detail": f"歩数が多く、歩幅が狭くなっている可能性があります（Cadence: {cadence:.1f}歩/分）。",
                "cause": "股関節の柔軟性低下や、すり足気味になっていることが考えられます。"
            })
            exercises.append("大股歩き練習")

        if gait_speed > 0 and gait_speed < 1.0 and age >= 65:
            fb_list.append({
                "title": "歩行速度低下（高齢者基準）",
                "detail": f"推定歩行速度が {gait_speed:.2f}m/s と、高齢者の転倒リスク基準（<1.0m/s）を下回っています。",
                "cause": "筋力低下や心肺機能の低下が考えられます。",
                "priority": True
            })

        if cv_score > cv_threshold:
            fb_list.append({
                "title": f"歩行周期のばらつき (要注意) - {age}歳基準",
                "detail": f"一歩ごとのリズムが一定ではありません（CV: {cv_score:.3f}、基準値: {cv_threshold}）。",
                "cause": "運動制御能力の低下や、注意機能の分散が影響している可能性があります。",
                "priority": True
            })
            exercises.append("メトロノーム歩行 (一定テンポでの歩行練習)")

        if sway_score > sway_threshold:
            fb_list.append({
                "title": f"骨盤の動揺（体幹不安定） - {age}歳基準",
                "detail": f"骨盤の左右への揺れが大きくなっています（Sway: {sway_score:.3f}、基準値: {sway_threshold}）。",
                "cause": "体幹筋や中殿筋の筋力低下が疑われます。",
                "priority": True
            })
            exercises.append("サイドレッグレイズ / サイドプランク（体幹・中殿筋強化）")
            exercises.append("腕振り足踏み（姿勢制御練習）")

        if asymmetry_percent > 8:
            dominant_side = "右" if right_mean > left_mean else "左"
            other_side = "左" if dominant_side == "右" else "右"
            fb_list.append({
                "title": "左右非対称性（荷重バランス異常）",
                "detail": (
                    f"{dominant_side}足のステップ間隔が広く、{dominant_side}荷重優位です（左右差: {asymmetry_percent:.1f}%）。\n"
                    f"→ {dominant_side}側の股関節・膝への負担が増大しています。"
                ),
                "cause": f"{other_side}側の筋力低下、または{dominant_side}側への代償的荷重が疑われます。",
                "priority": asymmetry_percent > 15
            })
            exercises.append(f"{other_side}側 片脚立ち練習（バランス・筋力強化）")
            exercises.append("左右均等荷重の意識化トレーニング")

        if abs(trunk_lean_mean) > 10:
            direction = "前" if trunk_lean_mean > 0 else "後ろ"
            fb_list.append({
                "title": "体幹の傾き",
                "detail": f"平均して体幹がやや{direction}に傾いています（平均体幹前傾角度: {trunk_lean_mean:.1f}度）。",
                "cause": "胸椎後弯や股関節周囲筋のアンバランスにより負担が増えている可能性があります。"
            })
            exercises.append("股関節屈筋ストレッチ / 胸椎伸展ストレッチ")

        if not fb_list:
            fb_list.append({
                "title": "良好な歩行パターン",
                "detail": "リズム、安定性、左右バランスともに大きな問題は見られません。",
                "cause": "現在の身体機能を維持しましょう。"
            })

    else:
        f_met = metrics.get("f_met") or {}
        s_met = metrics.get("s_met") or {}

        if abs(s_met.get("forward_head_score", 0.0)) > 5.0:
            fb_list.append({
                "title": "ストレートネック傾向 (FHP)",
                "detail": "頭部が肩よりも前方に突出しています。",
                "cause": "長時間のデスクワークやスマホ操作による首・肩甲骨周囲の緊張。"
            })
            exercises.append("チンイン (顎引き運動)")

        if abs(s_met.get("trunk_lean", 0.0)) > 10.0:
            fb_list.append({
                "title": "姿勢の崩れ (猫背/反り腰)",
                "detail": "上半身の重心軸が垂直から逸脱しています。",
                "cause": "体幹深層筋の弱化、または股関節屈筋群の短縮が考えられます。"
            })
            exercises.append("股関節屈筋ストレッチ")

        if abs(f_met.get("shoulder_slope", 0.0)) > 3.0:
            side = "右" if f_met["shoulder_slope"] > 0 else "左"
            fb_list.append({
                "title": f"{side}肩の下がり",
                "detail": f"{side}肩が下がる傾向があります。",
                "cause": "片側荷重や片側でのカバン持ちなど、日常姿勢のクセが影響している可能性があります。"
            })
            exercises.append("肩甲帯周囲のストレッチとロウイング運動")

        if not fb_list:
            fb_list.append({
                "title": "Good Posture",
                "detail": "非常に綺麗な姿勢アライメントです。",
                "cause": "この状態を維持できると腰痛・肩こり予防に有利です。"
            })

    exercises = list(dict.fromkeys(exercises))
    return fb_list, exercises

# 歩行解析
def analyze_gait_from_history(history, fps, w, h, height_cm=170):
    if not history or fps <= 0:
        return None, {"ml": None, "lean": None, "mid": None}

    left_ankle_y = []
    right_ankle_y = []
    pelvis_sway_history = []
    trunk_lean_list = []
    hip_distances_px = []

    max_ml_abs = 0.0
    max_lean_abs = 0.0
    frame_ml = None
    frame_lean = None

    mid_index = len(history) // 2
    frame_mid = history[mid_index][1].copy()

    for lms, frame in history:
        la = np.array([lms[27].x, lms[27].y])
        ra = np.array([lms[28].x, lms[28].y])

        left_ankle_y.append(float(la[1]))
        right_ankle_y.append(float(ra[1]))

        pelvis_mid_x = (lms[23].x + lms[24].x) / 2.0
        pelvis_sway_history.append(pelvis_mid_x)

        mid_shoulder = [ (lms[11].x + lms[12].x) / 2 * w,
                         (lms[11].y + lms[12].y) / 2 * h ]
        mid_hip = [ (lms[23].x + lms[24].x) / 2 * w,
                    (lms[23].y + lms[24].y) / 2 * h ]
        trunk_lean = calculate_vertical_angle(mid_hip, mid_shoulder)
        trunk_lean_list.append(trunk_lean)

        hip_l = np.array([lms[23].x * w, lms[23].y * h])
        hip_r = np.array([lms[24].x * w, lms[24].y * h])
        hip_distances_px.append(np.linalg.norm(hip_l - hip_r))

        trunk_center_x = (pelvis_mid_x * w + (lms[11].x + lms[12].x) / 2 * w) / 2.0
        ml_dev = (trunk_center_x - w / 2.0) / (w / 2.0)

        if abs(ml_dev) > max_ml_abs:
            max_ml_abs = abs(ml_dev)
            frame_ml = frame.copy()

        if abs(trunk_lean) > max_lean_abs:
            max_lean_abs = abs(trunk_lean)
            frame_lean = frame.copy()

    def detect_steps(ankle_y_list):
        steps = 0
        step_frames = []
        if len(ankle_y_list) > 2:
            arr = np.array(ankle_y_list)
            threshold = np.percentile(arr, 60)
            for i in range(1, len(arr)-1):
                if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > threshold:
                    steps += 1
                    step_frames.append(i)
        return steps, step_frames

    left_steps, left_frames = detect_steps(left_ankle_y)
    right_steps, right_frames = detect_steps(right_ankle_y)
    total_steps = left_steps + right_steps

    duration = len(history) / fps
    cadence = (total_steps / duration) * 60 if duration > 0 else 0.0

    asymmetry_percent = 0.0
    left_step_mean = 0.0
    right_step_mean = 0.0
    if len(left_frames) >= 2 and len(right_frames) >= 2:
        left_intervals = np.diff(left_frames)
        right_intervals = np.diff(right_frames)
        left_step_mean = float(np.mean(left_intervals))
        right_step_mean = float(np.mean(right_intervals))
        avg_step = (left_step_mean + right_step_mean) / 2.0
        if avg_step > 0:
            asymmetry_percent = abs(left_step_mean - right_step_mean) / avg_step * 100.0

    cv_score = 0.0
    all_step_frames = sorted(left_frames + right_frames)
    if len(all_step_frames) >= 3:
        intervals = np.diff(all_step_frames)
        mean_int = float(np.mean(intervals))
        std_int = float(np.std(intervals))
        if mean_int > 0:
            cv_score = std_int / mean_int

    sway_score = float(np.std(pelvis_sway_history)) if pelvis_sway_history else 0.0
    trunk_lean_mean = float(np.mean(trunk_lean_list)) if trunk_lean_list else 0.0

    gait_speed_m_s = 0.0
    if hip_distances_px and total_steps >= 2 and cadence > 0:
        estimated_stride_m = client_height_cm * 0.01 * 0.4
        gait_speed_m_s = (cadence / 60.0) * estimated_stride_m

    metrics = {
        "cadence": float(cadence),
        "steps": int(total_steps),
        "cv_score": float(cv_score),
        "sway_score": float(sway_score),
        "trunk_lean_mean": float(trunk_lean_mean),
        "asymmetry_percent": float(asymmetry_percent),
        "left_step_mean": float(left_step_mean),
        "right_step_mean": float(right_step_mean),
        "gait_speed_m_s": float(gait_speed_m_s),
        "left_steps": int(left_steps),
        "right_steps": int(right_steps),
    }

    key_frames = {
        "ml": frame_ml,
        "lean": frame_lean,
        "mid": frame_mid,
    }

    return metrics, key_frames

def process_video_advanced(file, height_cm=170):
    if not file:
        return None, None, None

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    cap = cv2.VideoCapture(tfile.name)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    history = []
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            res = pose.process(img_rgb)

            cv2.line(img, (w//2, 0), (w//2, h), (0, 255, 255), 1)

            if res.pose_landmarks:
                mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lms = res.pose_landmarks.landmark
                history.append((lms, img.copy()))

                def get_c(idx):
                    return [lms[idx].x * w, lms[idx].y * h]
                try:
                    knee_angle = calculate_angle(get_c(24), get_c(26), get_c(28))
                    cv2.rectangle(img, (w-220, 0), (w, 60), (255, 255, 255), -1)
                    cv2.putText(img, f"R-Knee: {int(knee_angle)}",
                                (w-200, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except Exception:
                    pass

            out.write(img)

    cap.release()
    out.release()

    metrics, key_frames = analyze_gait_from_history(history, fps, w, h, height_cm)

    snapshot_dict = {}
    for k in ["ml", "lean", "mid"]:
        frame = key_frames.get(k)
        if frame is not None:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            snapshot_dict[k] = Image.fromarray(img_rgb)
        else:
            snapshot_dict[k] = None

    return out_path, metrics, snapshot_dict

def analyze_static_image(image, view, posture_type):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return image, None

        h, w, _ = image.shape
        lms = results.pose_landmarks.landmark
        annotated_image = image.copy()
        cv2.line(annotated_image, (w//2, 0), (w//2, h), (0, 255, 255), 2)
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        def get_p(idx):
            return [lms[idx].x * w, lms[idx].y * h]

        metrics = {}
        if view == "front":
            metrics["head_tilt"] = calculate_slope(get_p(7), get_p(8))
            metrics["shoulder_slope"] = calculate_slope(get_p(11), get_p(12))
            metrics["hip_slope"] = calculate_slope(get_p(23), get_p(24))
        elif view == "side":
            ear_x = (lms[7].x + lms[8].x) / 2
            shoulder_x = (lms[11].x + lms[12].x) / 2
            metrics["forward_head_score"] = (ear_x - shoulder_x) * 100
            metrics["trunk_lean"] = calculate_vertical_angle(get_p(11), get_p(23))
            if posture_type == "立位 (Standing)":
                metrics["knee_angle"] = calculate_angle(get_p(23), get_p(25), get_p(27))
            else:
                metrics["hip_angle"] = calculate_angle(get_p(11), get_p(23), get_p(25))

        return annotated_image, metrics

def create_comprehensive_pdf(title, name, fb_data, exercises, metrics_data, snapshots=None):
    if snapshots is None:
        snapshots = {}

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    pw, ph = A4
    font_name = "HeiseiKakuGo-W5"

    today_str = datetime.now().strftime("%Y/%m/%d")

    c.setFont(font_name, 18)
    c.drawString(40, ph-50, title)
    c.setFont(font_name, 11)
    c.drawString(40, ph-75, f"氏名: {name}")
    c.drawString(250, ph-75, f"判定日: {today_str}")
    c.line(40, ph-85, pw-40, ph-85)

    y = ph - 120

    if snapshots:
        x_pos = pw - 220
        for label in ["mid", "ml", "lean"]:
            img = snapshots.get(label)
            if img is not None:
                ih = 120
                iw = ih * img.width / img.height
                c.drawImage(ImageReader(img), x_pos, y-ih, width=iw, height=ih)
                c.setFont(font_name, 8)
                c.drawString(x_pos, y-ih-10, f"▲ {label} frame")
                y -= ih + 40

    c.setFont(font_name, 14)
    c.drawString(40, y, "■ 測定結果 (Metrics)")
    y -= 25
    c.setFont(font_name, 10)

    if "cadence" in metrics_data:
        c.drawString(50, y, f"・歩行リズム (Cadence): {metrics_data['cadence']:.1f} 歩/分")
        y -= 15
        c.drawString(50, y, f"・推定歩行速度: {metrics_data['gait_speed_m_s']:.2f} m/s")
        y -= 15
        c.drawString(50, y, f"・歩行周期のばらつき (CV): {metrics_data['cv_score']:.3f}")
        y -= 15
        c.drawString(50, y, f"・骨盤の左右揺れ (Sway): {metrics_data['sway_score']:.3f}")
        y -= 15
        c.drawString(50, y, f"・左右差: {metrics_data['asymmetry_percent']:.1f} %")
        y -= 25

        star_text, _ = get_risk_stars(
            metrics_data["cv_score"],
            metrics_data["sway_score"],
            metrics_data["asymmetry_percent"],
            client_age,
        )
        c.setFont(font_name, 12)
        c.drawString(50, y, f"★ 総合リスク評価: {star_text}")
        y -= 30

    c.setFont(font_name, 14)
    c.drawString(40, y, "■ 分析・評価コメント (Problem / Cause)")
    y -= 25
    c.setFont(font_name, 10)

    for fb in fb_data:
        if y < 80:
            c.showPage()
            y = ph - 50
            c.setFont(font_name, 10)

        title_str = f"● {fb['title']}"
        if fb.get("priority"):
            title_str += " 【優先改善】"
        c.drawString(50, y, title_str)
        y -= 15
        c.drawString(60, y, f"状態: {fb['detail']}")
        y -= 15
        c.drawString(60, y, f"原因: {fb['cause']}")
        y -= 20

    if y < 120:
        c.showPage()
        y = ph - 50

    c.setFont(font_name, 14)
    c.drawString(40, y, "■ あなたへの処方箋 (推奨運動)")
    y -= 25
    c.setFont(font_name, 10)
    for ex in exercises:
        if y < 60:
            c.showPage()
            y = ph - 50
            c.setFont(font_name, 10)
        c.drawString(50, y, f"□ {ex}")
        y -= 15

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# UI 部分

if app_mode == "静止画：姿勢分析 (立位/座位)":
    st.info("📸 正面・側面それぞれの写真をアップロードしてください")
    posture_type = st.radio("姿勢タイプ", ["立位 (Standing)", "座位 (Sitting)"], horizontal=True)
    col_f, col_s = st.columns(2)
    with col_f:
        file_f = st.file_uploader("正面画像", type=["jpg", "jpeg", "png"])
    with col_s:
        file_s = st.file_uploader("側面画像", type=["jpg", "jpeg", "png"])

    if st.button("🚀 姿勢分析を実行"):
        if not file_f and not file_s:
            st.error("画像をアップロードしてください")
        else:
            f_img = f_met = s_img = s_met = None
            snapshot = None

            if file_f:
                img = np.array(Image.open(file_f))
                f_img, f_met = analyze_static_image(img, "front", posture_type)
                snapshot = Image.fromarray(cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB))
            if file_s:
                img = np.array(Image.open(file_s))
                s_img, s_met = analyze_static_image(img, "side", posture_type)
                if snapshot is None and s_img is not None:
                    snapshot = Image.fromarray(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB))

            c1, c2 = st.columns(2)
            with c1:
                if f_img is not None:
                    st.image(f_img, caption="正面解析", use_container_width=True)
            with c2:
                if s_img is not None:
                    st.image(s_img, caption="側面解析", use_container_width=True)

            metrics_pack = {"f_met": f_met, "s_met": s_met}
            fb_data, ex_list = generate_clinical_feedback(metrics_pack, "static", client_age)

            st.markdown("### 👨‍⚕️ AI分析結果")
            for item in fb_data:
                if item.get("priority"):
                    st.error(f"⚠️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")
                else:
                    st.info(f"ℹ️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")

            if ex_list:
                st.markdown("#### 🧘 推奨エクササイズ")
                for ex in ex_list:
                    st.success(f"✅ {ex}")

            snapshots = {"mid": snapshot} if snapshot is not None else {}
            pdf = create_comprehensive_pdf(
                "姿勢分析レポート", client_name, fb_data, ex_list, metrics_pack, snapshots
            )
            st.download_button("📄 レポート保存 (PDF)", pdf, "posture_report.pdf", "application/pdf")

else:
    st.info("🎥 歩行動画（全身が映っているもの）をアップロードしてください")
    video_file = st.file_uploader("歩行動画", type=["mp4", "mov"])

    if st.button("🚀 歩行分析を実行") and video_file:
        out_path, metrics, snapshots = process_video_advanced(video_file, client_height_cm)

        if out_path:
            st.video(out_path)

        if not metrics:
            st.error("歩行データを取得できませんでした。")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("ケイデンス", f"{metrics['cadence']:.1f} 歩/分")
            c2.metric("体幹の安定性(Sway)", f"{metrics['sway_score']:.3f}")
            c3.metric("歩行のばらつき(CV)", f"{metrics['cv_score']:.3f}")

            star_text, star_num = get_risk_stars(
                metrics["cv_score"],
                metrics["sway_score"],
                metrics["asymmetry_percent"],
                client_age,
            )
            st.markdown(f"### ⭐ 総合リスク: {star_text}")

            fb_data, ex_list = generate_clinical_feedback(metrics, "gait", client_age)

            st.markdown("---")
            st.subheader("📝 臨床フィードバック")
            for item in fb_data:
                if item.get("priority"):
                    st.error(f"⚠️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")
                else:
                    st.info(f"ℹ️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")

            if ex_list:
                st.markdown("#### 🧘 推奨エクササイズ")
                for ex in ex_list:
                    st.success(f"✅ {ex}")

            if app_mode == "動画：歩行分析 (Pro)":
                pdf = create_comprehensive_pdf(
                    "歩行機能分析レポート", client_name, fb_data, ex_list, metrics, snapshots
                )
                st.download_button("📄 詳細レポート保存 (PDF)", pdf, "gait_report_pro.pdf", "application/pdf")
