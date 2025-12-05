import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import io
import math
from datetime import datetime
from PIL import Image

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.utils import ImageReader

# 日本語フォント登録
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))

# MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- ページ設定 ---
st.set_page_config(page_title="AI姿勢・歩行分析ラボ", page_icon="🏥", layout="wide")

# --- CSS ---
HIDE_STYLE = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(HIDE_STYLE, unsafe_allow_html=True)

# --- サイドバー モード選択 ---
st.sidebar.header("⚙️ 分析モード")
app_mode = st.sidebar.radio(
    "モードを選択してください",
    ["動画：歩行分析 (Pro)", "動画：歩行分析 (Lite)", "静止画：姿勢分析 (立位/座位)"],
)

# --- タイトル ---
if "歩行" in app_mode:
    st.title("🏃‍♂️ AI歩行ドック (Clinical Grade)")
    st.caption("転倒リスク・腰痛リスクを「揺れ」「ばらつき」「左右差」から可視化")
else:
    st.title("📸 AI姿勢分析ラボ")
    st.caption("正面(アライメント) × 側面(猫背・FHP) の同時評価")

# --- 対象者情報 ---
st.sidebar.header("📋 対象者情報")
client_name = st.sidebar.text_input("氏名", "テスト 太郎 様")
client_age = st.sidebar.number_input("年齢", min_value=1, max_value=120, value=45, step=1)
client_gender = st.sidebar.selectbox("性別", ["男性", "女性", "その他"])
client_height_cm = st.sidebar.number_input("身長 (cm)", min_value=100, max_value=250, value=170, step=1)

if app_mode == "動画：歩行分析 (Pro)":
    with st.sidebar.expander("1. 問診・痛み", expanded=True):
        pain_areas = st.multiselect("痛み", ["なし", "首", "肩", "腰", "股関節", "膝", "足首"])


# ================= 共通ユーティリティ =================

def calculate_angle(a, b, c):
    """3点間の角度"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(rad * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_slope(a, b):
    if a is None or b is None:
        return 0.0
    return math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))


def calculate_vertical_angle(a, b):
    """垂直線に対する角度（a→b の線がどれだけ前後に倒れているか）"""
    if a is None or b is None:
        return 0.0
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))


def get_risk_stars(cv_score, sway_score, asymmetry_percent, age):
    """CV / Sway / 左右差 + 年齢から★評価"""
    risk_score = 0.0
    cv_threshold = 0.08 if age >= 65 else 0.05
    sway_threshold = 0.12 if age >= 65 else 0.08

    # CV
    if cv_score > cv_threshold * 1.5:
        risk_score += 2
    elif cv_score > cv_threshold:
        risk_score += 1

    # Sway
    if sway_score > sway_threshold * 1.5:
        risk_score += 2
    elif sway_score > sway_threshold:
        risk_score += 1

    # 左右差
    if asymmetry_percent > 15:
        risk_score += 2
    elif asymmetry_percent > 8:
        risk_score += 1

    # 年齢
    if age >= 75:
        risk_score += 1
    elif age >= 65:
        risk_score += 0.5

    # ★変換
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


# ================= フィードバック生成 =================

def generate_clinical_feedback(metrics, analysis_type="gait", age=45):
    """
    gait のとき:
      metrics = {
        'cadence','steps','cv_score','sway_score','trunk_lean_mean',
        'asymmetry_percent','left_step_mean','right_step_mean',
        'gait_speed_m_s', ...
      }
    static のとき:
      metrics = {'f_met': {...}, 's_met': {...}}
    """
    fb_list = []
    exercises = []

    # ---- A. 歩行 ----
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

        # 1. リズム・速度
        if cadence < 95:
            fb_list.append({
                "title": "歩行リズムの低下",
                "detail": f"歩行ペースがゆっくりです（Cadence: {cadence:.1f}歩/分, 推定速度: {gait_speed:.2f}m/s）。",
                "cause": "下肢筋力の低下や転倒への不安感が影響している可能性があります。"
            })
            exercises.append("椅子座り立ち（下肢筋力強化）")
        elif cadence > 125:
            fb_list.append({
                "title": "小刻み歩行の傾向",
                "detail": f"歩数が多く、歩幅が狭くなっている可能性があります（Cadence: {cadence:.1f}歩/分）。",
                "cause": "股関節の柔軟性低下やすり足傾向が考えられます。"
            })
            exercises.append("大股歩き練習")

        # 高齢者の歩行速度
        if gait_speed > 0 and age >= 65 and gait_speed < 1.0:
            fb_list.append({
                "title": "歩行速度低下（高齢者基準）",
                "detail": f"推定歩行速度が {gait_speed:.2f}m/s と基準（1.0m/s）を下回っています。",
                "cause": "筋力や心肺機能の低下が考えられます。",
                "priority": True
            })

        # ばらつき（CV）
        if cv_score > cv_threshold:
            fb_list.append({
                "title": "歩行周期のばらつき（要注意）",
                "detail": f"一歩ごとのリズムが一定ではありません（CV: {cv_score:.3f} / 基準: {cv_threshold:.3f}）。",
                "cause": "運動制御能力の低下や注意力の分散が影響している可能性があります。",
                "priority": True
            })
            exercises.append("メトロノーム歩行（一定テンポでの歩行練習）")

        # 体幹・骨盤の揺れ
        if sway_score > sway_threshold:
            fb_list.append({
                "title": "骨盤の動揺（体幹不安定）",
                "detail": f"骨盤の左右への揺れが大きくなっています（Sway: {sway_score:.3f} / 基準: {sway_threshold:.3f}）。",
                "cause": "体幹筋や中殿筋の筋力低下が疑われます。",
                "priority": True
            })
            exercises.append("サイドプランク / サイドレッグレイズ（体幹・中殿筋強化）")
            exercises.append("腕振り足踏み（姿勢制御練習）")

        # 左右差
        if asymmetry_percent > 8:
            dominant_side = "右" if right_mean > left_mean else "左"
            other_side = "左" if dominant_side == "右" else "右"
            fb_list.append({
                "title": "左右非対称性（荷重バランス）",
                "detail": f"{dominant_side}脚への荷重が強い傾向があります（左右差: {asymmetry_percent:.1f}%）。",
                "cause": f"{other_side}脚の筋力低下や、{dominant_side}脚の痛み回避による代償歩行が考えられます。",
                "priority": asymmetry_percent > 15
            })
            exercises.append(f"{other_side}片脚立ち練習（バランス・筋力強化）")
            exercises.append("左右均等荷重の意識トレーニング")

        # 体幹前傾
        if abs(trunk_lean_mean) > 10:
            direction = "前" if trunk_lean_mean > 0 else "後ろ"
            fb_list.append({
                "title": "体幹の傾き",
                "detail": f"体幹が平均して{direction}に傾いています（平均: {trunk_lean_mean:.1f}度）。",
                "cause": "胸椎後弯や股関節周囲筋のアンバランスにより腰椎・股関節に負担がかかっている可能性があります。"
            })
            exercises.append("股関節屈筋ストレッチ / 胸椎伸展ストレッチ")

        if not fb_list:
            fb_list.append({
                "title": "良好な歩行パターン",
                "detail": "リズム・安定性・左右バランスともに大きな問題は見られません。",
                "cause": "現在の身体機能を維持しましょう。"
            })

    # ---- B. 静止姿勢 ----
    else:
        f_met = metrics.get("f_met") or {}
        s_met = metrics.get("s_met") or {}

        if abs(s_met.get("forward_head_score", 0.0)) > 5.0:
            fb_list.append({
                "title": "ストレートネック傾向（FHP）",
                "detail": "頭部が肩よりも前方に突出しています。",
                "cause": "長時間のスマホ・PC操作による首・肩甲帯周囲の緊張。"
            })
            exercises.append("チンイン（顎引き運動）")

        if abs(s_met.get("trunk_lean", 0.0)) > 10:
            fb_list.append({
                "title": "姿勢の崩れ（猫背/反り腰）",
                "detail": "上半身の重心軸が垂直から逸脱しています。",
                "cause": "体幹深層筋の弱化、股関節屈筋群の短縮が考えられます。"
            })
            exercises.append("股関節屈筋ストレッチ（ジャックナイフなど）")

        if abs(f_met.get("shoulder_slope", 0.0)) > 3:
            side = "右" if f_met["shoulder_slope"] > 0 else "左"
            fb_list.append({
                "title": f"{side}肩の下がり",
                "detail": f"{side}肩が下がる傾向があります。",
                "cause": "片側荷重や片側でのカバン持ちなど日常姿勢のクセが影響している可能性があります。"
            })
            exercises.append("肩甲帯ストレッチとロウイング運動")

        if not fb_list:
            fb_list.append({
                "title": "Good Posture",
                "detail": "非常に綺麗な姿勢アライメントです。",
                "cause": "この状態を維持できると腰痛・肩こり予防に有利です。"
            })

    exercises = list(dict.fromkeys(exercises))  # 重複削除
    return fb_list, exercises


# ================= 歩行解析本体 =================

def analyze_gait_from_history(history, fps, width, height, height_cm=170):
    if not history or fps <= 0:
        return None, {"ml": None, "lean": None, "mid": None}

    left_ankle_y = []
    right_ankle_y = []
    pelvis_sway_x = []
    trunk_lean_list = []
    hip_distances_px = []

    max_ml_abs = 0.0
    max_lean_abs = 0.0
    frame_ml = None
    frame_lean = None

    mid_index = len(history) // 2
    frame_mid = history[mid_index][1].copy() if history else None

    for lms, frame in history:
        la = np.array([lms[27].x, lms[27].y])
        ra = np.array([lms[28].x, lms[28].y])

        left_ankle_y.append(float(la[1]))
        right_ankle_y.append(float(ra[1]))

        pelvis_mid_x = (lms[23].x + lms[24].x) / 2
        pelvis_sway_x.append(pelvis_mid_x)

        mid_shoulder = [
            (lms[11].x + lms[12].x) / 2 * width,
            (lms[11].y + lms[12].y) / 2 * height,
        ]
        mid_hip = [
            (lms[23].x + lms[24].x) / 2 * width,
            (lms[23].y + lms[24].y) / 2 * height,
        ]
        trunk_lean = calculate_vertical_angle(mid_hip, mid_shoulder)
        trunk_lean_list.append(trunk_lean)

        hip_l = np.array([lms[23].x * width, lms[23].y * height])
        hip_r = np.array([lms[24].x * width, lms[24].y * height])
        hip_distances_px.append(np.linalg.norm(hip_l - hip_r))

        trunk_center_x = (pelvis_mid_x * width + (lms[11].x + lms[12].x) / 2 * width) / 2
        ml_dev = (trunk_center_x - width / 2) / (width / 2)

        if abs(ml_dev) > max_ml_abs:
            max_ml_abs = abs(ml_dev)
            frame_ml = frame.copy()

        if abs(trunk_lean) > max_lean_abs:
            max_lean_abs = abs(trunk_lean)
            frame_lean = frame.copy()

    # 歩数検出（左右別）
    def detect_steps(ankle_y_list):
        steps = 0
        step_frames = []
        if len(ankle_y_list) > 2:
            arr = np.array(ankle_y_list)
            threshold = np.percentile(arr, 60)  # 下側 60% を接地候補
            for i in range(1, len(arr) - 1):
                if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > threshold:
                    steps += 1
                    step_frames.append(i)
        return steps, step_frames

    left_steps, left_frames = detect_steps(left_ankle_y)
    right_steps, right_frames = detect_steps(right_ankle_y)
    total_steps = left_steps + right_steps

    duration = len(history) / fps
    cadence = (total_steps / duration) * 60 if duration > 0 else 0.0

    # 左右対称性
    asymmetry_percent = 0.0
    left_step_mean = 0.0
    right_step_mean = 0.0
    if len(left_frames) >= 2 and len(right_frames) >= 2:
        left_intervals = np.diff(left_frames)
        right_intervals = np.diff(right_frames)
        left_step_mean = float(np.mean(left_intervals))
        right_step_mean = float(np.mean(right_intervals))
        avg_step = (left_step_mean + right_step_mean) / 2
        if avg_step > 0:
            asymmetry_percent = abs(left_step_mean - right_step_mean) / avg_step * 100

    # CV
    cv_score = 0.0
    all_step_frames = sorted(left_frames + right_frames)
    if len(all_step_frames) >= 3:
        intervals = np.diff(all_step_frames)
        mean_int = float(np.mean(intervals))
        std_int = float(np.std(intervals))
        if mean_int > 0:
            cv_score = std_int / mean_int

    # 骨盤揺れ
    sway_score = float(np.std(pelvis_sway_x)) if pelvis_sway_x else 0.0
    trunk_lean_mean = float(np.mean(trunk_lean_list)) if trunk_lean_list else 0.0

    # 歩行速度（身長から簡易推定）
    gait_speed_m_s = 0.0
    if hip_distances_px and total_steps >= 2:
        estimated_stride_m = height_cm * 0.01 * 0.4  # 身長の40%
        gait_speed_m_s = (cadence / 60.0) * estimated_stride_m if cadence > 0 else 0.0

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

    key_frames = {"ml": frame_ml, "lean": frame_lean, "mid": frame_mid}
    return metrics, key_frames


def process_video_advanced(file, height_cm=170):
    """動画処理 + gait解析 + 代表フレーム抽出"""
    if not file:
        return None, None, None

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    history = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            res = pose.process(img_rgb)

            cv2.line(img, (width // 2, 0), (width // 2, height), (0, 255, 255), 1)

            if res.pose_landmarks:
                mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lms = res.pose_landmarks.landmark
                history.append((lms, img.copy()))

                def get_c(idx):
                    return [lms[idx].x * width, lms[idx].y * height]

                # 右膝角度表示
                try:
                    knee_angle = calculate_angle(get_c(24), get_c(26), get_c(28))
                    cv2.rectangle(img, (width - 220, 0), (width, 60), (255, 255, 255), -1)
                    cv2.putText(
                        img,
                        f"R-Knee: {int(knee_angle)}",
                        (width - 200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                except Exception:
                    pass

            out.write(img)

    cap.release()
    out.release()

    metrics, key_frames = analyze_gait_from_history(history, fps, width, height, height_cm)

    # BGR -> PIL 変換
    snapshot_dict = {}
    for k in ["ml", "lean", "mid"]:
        frame = key_frames.get(k)
        if frame is not None:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            snapshot_dict[k] = Image.fromarray(img_rgb)
        else:
            snapshot_dict[k] = None

    return out_path, metrics, snapshot_dict


# ================= 静止画解析 =================

def analyze_static_image(image, view, posture_type):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return image, None

        h, w, _ = image.shape
        lms = results.pose_landmarks.landmark
        annotated_image = image.copy()
        cv2.line(annotated_image, (w // 2, 0), (w // 2, h), (0, 255, 255), 2)
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


# ================= PDFレポート作成 =================

def create_comprehensive_pdf(title, name, age, gender,
                             metrics_data, feedback_data, exercises, snapshots=None):
    if snapshots is None:
        snapshots = {}

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4
    font_name = "HeiseiKakuGo-W5"

    today = datetime.now().strftime("%Y-%m-%d")

    # ヘッダー
    c.setFont(font_name, 18)
    c.drawString(40, page_h - 50, title)
    c.setFont(font_name, 11)
    c.drawString(40, page_h - 75, f"氏名: {name}")
    c.drawString(220, page_h - 75, f"年齢: {age} 歳 / 性別: {gender}")
    c.drawString(420, page_h - 75, f"判定日: {today}")
    c.line(40, page_h - 85, page_w - 40, page_h - 85)

    y = page_h - 110

    # スナップショット
    if snapshots:
        x_pos = page_w - 40 - 160
        for label in ["mid", "ml", "lean"]:
            img = snapshots.get(label)
            if img is not None:
                ratio = img.height / img.width
                w_img = 160
                h_img = int(w_img * ratio)
                c.drawImage(ImageReader(img), x_pos, y - h_img, width=w_img, height=h_img)
                c.setFont(font_name, 9)
                c.drawString(x_pos, y - h_img - 12, f"▲ {label} フレーム")
                y = min(y, y - h_img - 30)

    # 指標
    c.setFont(font_name, 13)
    c.drawString(40, y, "■ 測定結果")
    y -= 20
    c.setFont(font_name, 10)

    if "cadence" in metrics_data:
        cv_val = metrics_data.get("cv_score", 0.0)
        sway_val = metrics_data.get("sway_score", 0.0)
        asym = metrics_data.get("asymmetry_percent", 0.0)
        gait_v = metrics_data.get("gait_speed_m_s", 0.0)
        risk_label, _ = get_risk_stars(cv_val, sway_val, asym, age)

        c.drawString(50, y, f"・歩行リズム (Cadence): {metrics_data['cadence']:.1f} 歩/分")
        y -= 15
        c.drawString(50, y, f"・推定歩行速度: {gait_v:.2f} m/s")
        y -= 15
        c.drawString(50, y, f"・歩行周期のばらつき (CV): {cv_val:.3f}")
        y -= 15
        c.drawString(50, y, f"・骨盤の動揺 (Sway): {sway_val:.3f}")
        y -= 15
        c.drawString(50, y, f"・左右差: {asym:.1f} %")
        y -= 15
        c.drawString(50, y, f"・総合リスク評価: {risk_label}")
        y -= 30
    else:
        f_met = metrics_data.get("f_met") or {}
        s_met = metrics_data.get("s_met") or {}

        if f_met:
            c.drawString(50, y, f"・頭部の傾き: {f_met.get('head_tilt', 0.0):.1f} 度")
            y -= 15
            c.drawString(50, y, f"・肩の傾き: {f_met.get('shoulder_slope', 0.0):.1f} 度")
            y -= 15
        if s_met:
            c.drawString(50, y, f"・FHPスコア: {s_met.get('forward_head_score', 0.0):.1f}")
            y -= 15
            c.drawString(50, y, f"・体幹前傾角度: {s_met.get('trunk_lean', 0.0):.1f} 度")
            y -= 30

    # フィードバック
    c.setFont(font_name, 13)
    c.drawString(40, y, "■ 分析コメント")
    y -= 20
    c.setFont(font_name, 10)

    for fb in feedback_data:
        title_txt = f"● {fb['title']}"
        if fb.get("priority"):
            title_txt += " 【優先改善】"
        c.drawString(50, y, title_txt)
        y -= 13
        c.drawString(60, y, f"状態: {fb['detail']}")
        y -= 13
        c.drawString(60, y, f"原因: {fb['cause']}")
        y -= 18
        if y < 100:
            c.showPage()
            y = page_h - 50
            c.setFont(font_name, 10)

    # エクササイズ
    if y < 120:
        c.showPage()
        y = page_h - 50
    c.setFont(font_name, 13)
    c.drawString(40, y, "■ 推奨エクササイズ")
    y -= 20
    c.setFont(font_name, 10)
    for ex in exercises:
        c.drawString(50, y, f"・{ex}")
        y -= 13
        if y < 50:
            c.showPage()
            y = page_h - 50
            c.setFont(font_name, 10)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf


# ================= メインアプリ =================

# --- 静止姿勢モード ---
if app_mode == "静止画：姿勢分析 (立位/座位)":
    st.info("📸 正面・側面それぞれの写真をアップロードしてください")
    posture_type = st.radio("姿勢タイプ", ["立位 (Standing)", "座位 (Sitting)"], horizontal=True)
    col1, col2 = st.columns(2)
    with col1:
        file_front = st.file_uploader("正面画像", type=["jpg", "jpeg", "png"])
    with col2:
        file_side = st.file_uploader("側面画像", type=["jpg", "jpeg", "png"])

    if st.button("🚀 姿勢分析を実行"):
        if not file_front and not file_side:
            st.error("画像をアップロードしてください。")
        else:
            f_img = f_met = s_img = s_met = None
            snapshot = None

            if file_front:
                img = np.array(Image.open(file_front))
                f_img, f_met = analyze_static_image(img, "front", posture_type)
                snapshot = Image.fromarray(cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB))
            if file_side:
                img = np.array(Image.open(file_side))
                s_img, s_met = analyze_static_image(img, "side", posture_type)
                if snapshot is None:
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

            st.markdown("### 👨‍⚕️ AI姿勢レポート")
            for item in fb_data:
                if item.get("priority"):
                    st.error(f"⚠️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")
                else:
                    st.info(f"ℹ️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")

            st.markdown("#### 🧘 推奨エクササイズ")
            for ex in ex_list:
                st.success(f"✅ {ex}")

            pdf = create_comprehensive_pdf(
                "姿勢分析レポート",
                client_name,
                client_age,
                client_gender,
                metrics_pack,
                fb_data,
                ex_list,
                {"mid": snapshot},
            )
            st.download_button("📄 レポート保存 (PDF)", pdf, "posture_report.pdf", "application/pdf")

# --- 歩行モード（Pro / Lite共通ロジック） ---
else:
    st.info("🎥 歩行動画（全身が映っているもの）をアップロードしてください")
    file_v = st.file_uploader("歩行動画", type=["mp4", "mov"])

    if st.button("🚀 歩行分析を実行") and file_v:
        out_path, metrics, snapshots = process_video_advanced(file_v, client_height_cm)
        if not metrics:
            st.error("解析に必要なランドマークが検出できませんでした。撮影条件を見直してください。")
        else:
            st.video(out_path)
            st.markdown("### 📊 歩行ドック診断結果")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ケイデンス", f"{metrics['cadence']:.1f} 歩/分")
            with col2:
                st.metric("推定速度", f"{metrics['gait_speed_m_s']:.2f} m/s")
            with col3:
                st.metric("体幹の揺れ (Sway)", f"{metrics['sway_score']:.3f}")
            with col4:
                st.metric("歩行CV", f"{metrics['cv_score']:.3f}")

            risk_label, _ = get_risk_stars(
                metrics["cv_score"], metrics["sway_score"], metrics["asymmetry_percent"], client_age
            )
            st.info(f"総合リスク評価: **{risk_label}**")

            fb_data, ex_list = generate_clinical_feedback(metrics, "gait", client_age)

            st.markdown("---")
            st.subheader("📝 臨床フィードバック")
            for item in fb_data:
                if item.get("priority"):
                    st.error(f"⚠️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")
                else:
                    st.info(f"ℹ️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")

            st.markdown("#### 🧘 推奨プログラム")
            for ex in ex_list:
                st.success(f"✅ {ex}")

            pdf = create_comprehensive_pdf(
                "歩行機能分析レポート",
                client_name,
                client_age,
                client_gender,
                metrics,
                fb_data,
                ex_list,
                snapshots,
            )
            st.download_button("📄 詳細レポート保存 (PDF)", pdf, "gait_report.pdf", "application/pdf")
