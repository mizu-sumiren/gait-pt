import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import io
import math
from datetime import datetime
from PIL import Image

# --- PDF生成用ライブラリ ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.utils import ImageReader

# --- 日本語フォント登録 (PDF用) ---
pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))

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
    st.title("🏃‍♂️ AI歩行ドック (Clinical Grade)")
    st.caption("転倒リスク・腰痛リスクを「揺れ」「ばらつき」「左右差」から科学的に可視化（推測です）")
else:
    st.title("📸 AI姿勢分析ラボ")
    st.caption("正面(アライメント) × 側面(猫背・FHP) の同時評価")

# --- サイドバー入力（年齢・性別・身長） ---
st.sidebar.header("📋 対象者情報")
client_name = st.sidebar.text_input("氏名", "テスト 太郎 様")
client_age = st.sidebar.number_input("年齢", min_value=1, max_value=120, value=45, step=1)
client_gender = st.sidebar.selectbox("性別", ["男性", "女性", "その他"])
client_height_cm = st.sidebar.number_input("身長 (cm)", min_value=100, max_value=250, value=170, step=1)

if app_mode == "動画：歩行分析 (Pro)":
    with st.sidebar.expander("1. 問診・痛み", expanded=True):
        pain_areas = st.multiselect("痛み", ["なし", "首", "肩", "腰", "股関節", "膝", "足首"])

# ========== 共通ユーティリティ ==========

def calculate_angle(a, b, c):
    """3点間の角度を算出"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_slope(a, b):
    if a is None or b is None:
        return 0
    return math.degrees(math.atan2(a[1]-b[1], a[0]-b[0]))

def calculate_vertical_angle(a, b):
    """垂直線に対する角度（bが上、aが下）"""
    if a is None or b is None:
        return 0
    return math.degrees(math.atan2(b[0]-a[0], b[1]-a[1]))

def get_risk_stars(cv_score, sway_score, asymmetry_percent, age):
    """
    総合リスク評価を★5段階で算出（推測です）
    CV, Sway, 左右差, 年齢を考慮
    """
    risk_score = 0.0

    # 年齢補正閾値（推測です）
    cv_threshold = 0.08 if age >= 65 else 0.05
    sway_threshold = 0.12 if age >= 65 else 0.08

    # CV評価
    if cv_score > cv_threshold * 1.5:
        risk_score += 2
    elif cv_score > cv_threshold:
        risk_score += 1

    # Sway評価
    if sway_score > sway_threshold * 1.5:
        risk_score += 2
    elif sway_score > sway_threshold:
        risk_score += 1

    # 左右差評価
    if asymmetry_percent > 15:
        risk_score += 2
    elif asymmetry_percent > 8:
        risk_score += 1

    # 年齢リスク加算
    if age >= 75:
        risk_score += 1
    elif age >= 65:
        risk_score += 0.5

    # ★変換（リスク高い=★少ない）
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

# ========== フィードバック生成ロジック ==========

def generate_clinical_feedback(metrics, analysis_type="gait", age=45):
    """
    metrics: gait のときは
        {'cadence','steps','cv_score','sway_score','trunk_lean_mean',
         'asymmetry_percent','left_step_mean','right_step_mean','gait_speed_m_s',...}
    static のときは
        {'f_met': {...}, 's_met': {...}}
    """
    fb_list = []
    exercises = []

    # === A. 歩行分析フィードバック ===
    if analysis_type == "gait":
        cadence = metrics.get('cadence', 0.0)
        sway_score = metrics.get('sway_score', 0.0)
        cv_score = metrics.get('cv_score', 0.0)
        trunk_lean_mean = metrics.get('trunk_lean_mean', 0.0)
        asymmetry_percent = metrics.get('asymmetry_percent', 0.0)
        left_mean = metrics.get('left_step_mean', 0.0)
        right_mean = metrics.get('right_step_mean', 0.0)
        gait_speed = metrics.get('gait_speed_m_s', 0.0)

        # 年齢補正閾値（推測です）
        cv_threshold = 0.08 if age >= 65 else 0.05
        sway_threshold = 0.12 if age >= 65 else 0.08

        # 1. リズム・速度 (Cadence + 歩行速度)
        if cadence < 95:
            fb_list.append({
                "title": "歩行リズムの低下",
                "detail": f"歩行ペースがゆっくりです（Cadence: {cadence:.1f}歩/分, 推定速度: {gait_speed:.2f}m/s・推定値です）。推進力が低下している可能性があります。",
                "cause": "下肢筋力の低下や、転倒への不安感が影響している可能性があります（推測です）。"
            })
            exercises.append("椅子座り立ち (下肢筋力強化)")
        elif cadence > 125:
            fb_list.append({
                "title": "小刻み歩行の傾向",
                "detail": f"歩数が多く、歩幅が狭くなっている可能性があります（Cadence: {cadence:.1f}歩/分）。",
                "cause": "股関節の柔軟性低下や、すり足気味になっていることが考えられます（推測です）。"
            })
            exercises.append("大股歩き練習")

        # 歩行速度の単独評価（高齢者向け目安・推測です）
        if gait_speed > 0 and age >= 65 and gait_speed < 1.0:
            fb_list.append({
                "title": "歩行速度低下（高齢者基準）",
                "detail": f"推定歩行速度が {gait_speed:.2f}m/s と、転倒リスクが高くなる目安（1.0m/s未満・推測です）を下回っています。",
                "cause": "筋力低下や心肺機能の低下が考えられます（推測です）。",
                "priority": True
            })

        # 2. ばらつき・安定性 (CV)
        if cv_score > cv_threshold:
            fb_list.append({
                "title": f"歩行周期のばらつき (要注意) - {age}歳基準",
                "detail": f"一歩ごとのリズムが一定ではありません（CV: {cv_score:.3f}, 目安: {cv_threshold:.3f}以上で注意・推測です）。",
                "cause": "運動制御能力の低下や、注意機能の分散（考え事など）が影響している可能性があります（推測です）。",
                "priority": True
            })
            exercises.append("メトロノーム歩行 (一定テンポでの歩行練習)")

        # 3. 体幹の動揺 (sway_score: 骨盤中点)
        if sway_score > sway_threshold:
            fb_list.append({
                "title": f"骨盤の動揺（体幹不安定） - {age}歳基準",
                "detail": f"骨盤の左右への揺れが大きくなっています（Sway: {sway_score:.3f}, 目安: {sway_threshold:.3f}以上で注意・推測です）。",
                "cause": "体幹筋（腹圧）の機能不全や、中殿筋の筋力低下が疑われます（推測です）。",
                "priority": True
            })
            exercises.append("サイドレッグレイズ / サイドプランク（体幹・中殿筋強化）")
            exercises.append("腕振り足踏み（姿勢制御練習）")

        # 4. 左右対称性
        if asymmetry_percent > 8:
            dominant_side = "右" if right_mean > left_mean else "左"
            other_side = "左" if dominant_side == "右" else "右"
            fb_list.append({
                "title": "左右非対称性（荷重バランス）",
                "detail": f"{dominant_side}足のステップ間隔が広く、{dominant_side}荷重優位です（左右差: {asymmetry_percent:.1f}%・推測です）。",
                "cause": f"{other_side}側の筋力低下、または{dominant_side}側への代償的荷重が疑われます。片側性の痛みや機能障害の可能性があります（推測です）。",
                "priority": asymmetry_percent > 15
            })
            exercises.append(f"{other_side}側 片脚立ち練習（バランス・筋力強化）")
            exercises.append("左右均等荷重の意識化トレーニング")

        # 5. 体幹前傾
        if abs(trunk_lean_mean) > 10:
            direction = "前" if trunk_lean_mean > 0 else "後ろ"
            fb_list.append({
                "title": "体幹の傾き",
                "detail": f"平均して体幹がやや{direction}に傾いています（平均体幹前傾角度: {trunk_lean_mean:.1f}度・推測です）。",
                "cause": "胸椎の後弯や股関節周囲筋のアンバランスにより、腰椎・股関節への負担が増えている可能性があります（推測です）。"
            })
            exercises.append("股関節屈筋ストレッチ / 胸椎伸展ストレッチ")

        if not fb_list:
            fb_list.append({
                "title": "良好な歩行パターン",
                "detail": "リズム、安定性、左右バランスともに大きな問題は見られません。",
                "cause": "現在の身体機能を維持しましょう。"
            })

    # === B. 姿勢分析フィードバック ===
    else:
        f_met = metrics.get('f_met')
        s_met = metrics.get('s_met')

        if s_met and abs(s_met.get('forward_head_score', 0)) > 5.0:
            fb_list.append({
                "title": "ストレートネック傾向 (FHP)",
                "detail": "頭部が肩よりも前方に突出しています。",
                "cause": "長時間のデスクワークやスマホ操作による首・肩甲骨周囲の緊張（推測です）。"
            })
            exercises.append("チンイン (顎引き運動)")

        if s_met and abs(s_met.get('trunk_lean', 0)) > 10:
            fb_list.append({
                "title": "姿勢の崩れ (猫背/反り腰)",
                "detail": "上半身の重心軸が垂直から逸脱しています。",
                "cause": "体幹深層筋の弱化、または股関節屈筋群の短縮が考えられます（推測です）。"
            })
            exercises.append("股関節屈筋ストレッチ (ジャックナイフストレッチなど)")

        if f_met and abs(f_met.get('shoulder_slope', 0)) > 3.0:
            side = "右" if f_met['shoulder_slope'] > 0 else "左"
            fb_list.append({
                "title": f"{side}肩の下がり",
                "detail": f"{side}肩が下がる傾向があります。",
                "cause": "片側荷重や片側でのカバン持ちなど、日常姿勢のクセが影響している可能性があります（推測です）。"
            })
            exercises.append("肩甲帯周囲のストレッチとロウイング運動")

        if not fb_list:
            fb_list.append({
                "title": "Good Posture",
                "detail": "非常に綺麗な姿勢アライメントです。",
                "cause": "この状態を維持できると腰痛・肩こり予防に有利です（推測です）。"
            })

    # 重複エクササイズ削除
    exercises = list(dict.fromkeys(exercises))
    return fb_list, exercises

# ========== 歩行解析（履歴→メトリクス） ==========

def analyze_gait_from_history(history, fps, w, h, height_cm=170):
    """
    history: [(landmarks, frame_bgr), ...]
    戻り値: metrics(dict), key_frames(dict: 'ml','lean','mid')
    """
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

    for idx, (lms, frame) in enumerate(history):
        # 左右足首Y
        la = np.array([lms[27].x, lms[27].y])
        ra = np.array([lms[28].x, lms[28].y])
        left_ankle_y.append(float(la[1]))
        right_ankle_y.append(float(ra[1]))

        # 骨盤中点x
        pelvis_mid_x = (lms[23].x + lms[24].x) / 2
        pelvis_sway_history.append(pelvis_mid_x)

        # 体幹前傾
        mid_shoulder = [(lms[11].x + lms[12].x) / 2 * w,
                        (lms[11].y + lms[12].y) / 2 * h]
        mid_hip = [(lms[23].x + lms[24].x) / 2 * w,
                   (lms[23].y + lms[24].y) / 2 * h]
        trunk_lean = calculate_vertical_angle(mid_hip, mid_shoulder)
        trunk_lean_list.append(trunk_lean)

        # 股関節間距離（歩幅スケール推定用・推測です）
        hip_l = np.array([lms[23].x * w, lms[23].y * h])
        hip_r = np.array([lms[24].x * w, lms[24].y * h])
        hip_distances_px.append(np.linalg.norm(hip_l - hip_r))

        # 左右偏位（画面中央からのズレ）
        trunk_center_x = (pelvis_mid_x * w + (lms[11].x + lms[12].x) / 2 * w) / 2
        ml_dev = (trunk_center_x - w / 2) / (w / 2)

        if abs(ml_dev) > max_ml_abs:
            max_ml_abs = abs(ml_dev)
            frame_ml = frame.copy()

        if abs(trunk_lean) > max_lean_abs:
            max_lean_abs = abs(trunk_lean)
            frame_lean = frame.copy()

    # 足首Y極大でステップ検出（簡易・推測です）
    def detect_steps(ankle_y_list):
        steps = 0
        step_frames = []
        if len(ankle_y_list) > 2:
            arr = np.array(ankle_y_list)
            threshold = np.percentile(arr, 60)  # 下方向の60%タイル（推測です）
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

    # 全ステップ間隔からCV
    cv_score = 0.0
    all_step_frames = sorted(left_frames + right_frames)
    if len(all_step_frames) >= 3:
        intervals = np.diff(all_step_frames)
        mean_int = float(np.mean(intervals))
        std_int = float(np.std(intervals))
        if mean_int > 0:
            cv_score = std_int / mean_int

    # 骨盤中点xのSD
    sway_score = float(np.std(pelvis_sway_history)) if pelvis_sway_history else 0.0

    trunk_lean_mean = float(np.mean(trunk_lean_list)) if trunk_lean_list else 0.0

    # 歩行速度推定（かなりラフな推定です・推測です）
    gait_speed_m_s = 0.0
    if total_steps >= 2:
        estimated_stride_m = client_height_cm * 0.01 * 0.4  # 身長の40%をストライドと仮定
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

    key_frames = {
        "ml": frame_ml,
        "lean": frame_lean,
        "mid": frame_mid
    }

    return metrics, key_frames

def process_video_advanced(file, height_cm=170):
    """動画処理 + gait解析 + 代表フレーム抽出"""
    if not file:
        return None, None, None

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    cap = cv2.VideoCapture(tfile.name)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    history = []
    frame_idx = 0

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            frame_idx += 1

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            res = pose.process(img_rgb)

            cv2.line(img, (w//2, 0), (w//2, h), (0, 255, 255), 1)

            if res.pose_landmarks:
                mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lms = res.pose_landmarks.landmark
                history.append((lms, img.copy()))

                # 右膝角度表示（おまけ）
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

# ========== 静止画解析 ==========

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
            metrics['head_tilt'] = calculate_slope(get_p(7), get_p(8))
            metrics['shoulder_slope'] = calculate_slope(get_p(11), get_p(12))
            metrics['hip_slope'] = calculate_slope(get_p(23), get_p(24))
        elif view == "side":
            ear_x = (lms[7].x + lms[8].x) / 2
            shoulder_x = (lms[11].x + lms[12].x) / 2
            metrics['forward_head_score'] = (ear_x - shoulder_x) * 100  # 推測的スコア
            metrics['trunk_lean'] = calculate_vertical_angle(get_p(11), get_p(23))
            if posture_type == "立位 (Standing)":
                metrics['knee_angle'] = calculate_angle(get_p(23), get_p(25), get_p(27))
            else:
                metrics['hip_angle'] = calculate_angle(get_p(11), get_p(23), get_p(25))

        return annotated_image, metrics

# ========== PDFレポート生成 ==========

def create_comprehensive_pdf(title, name, feedback_data, exercises, metrics_data,
                             snapshot_obj=None, risk_label=None):
    b = io.BytesIO()
    c = canvas.Canvas(b, pagesize=A4)
    page_w, page_h = A4
    font_name = "HeiseiKakuGo-W5"

    today = datetime.now().strftime("%Y/%m/%d")
    c.setFont(font_name, 20)
    c.drawString(40, page_h - 50, f"{title}")
    c.setFont(font_name, 12)
    c.drawString(40, page_h - 80, f"氏名: {name} 様")
    c.drawString(400, page_h - 80, f"判定日: {today}")
    c.line(40, page_h - 90, 550, page_h - 90)

    current_y = page_h - 120

    # スナップショット画像
    if snapshot_obj:
        img_w = 200
        img_h = 150
        if isinstance(snapshot_obj, dict):
            base_y = current_y
            x1 = 330
            x2 = x1 + img_w + 10
            if snapshot_obj.get("ml"):
                buf1 = io.BytesIO()
                snapshot_obj["ml"].save(buf1, format="PNG")
                buf1.seek(0)
                c.drawImage(ImageReader(buf1), x1, base_y - img_h, width=img_w, height=img_h)
                c.drawString(x1, base_y - img_h - 12, "▲ 左右揺れが大きい場面（推測です）")
            if snapshot_obj.get("lean"):
                buf2 = io.BytesIO()
                snapshot_obj["lean"].save(buf2, format="PNG")
                buf2.seek(0)
                c.drawImage(ImageReader(buf2), x2, base_y - img_h, width=img_w, height=img_h)
                c.drawString(x2, base_y - img_h - 12, "▲ 体幹前傾が強い場面（推測です）")
            current_y = base_y - img_h - 30
        else:
            buf = io.BytesIO()
            snapshot_obj.save(buf, format="PNG")
            buf.seek(0)
            c.drawImage(ImageReader(buf), 380, current_y - img_h, width=img_w, height=img_h)
            c.drawString(380, current_y - img_h - 12, "▲ 代表スナップショット")
            current_y = current_y - img_h - 30

    # Metrics
    c.setFont(font_name, 14)
    c.drawString(40, current_y, "■ 測定結果 (Metrics)")
    current_y -= 30
    c.setFont(font_name, 11)

    if "cadence" in metrics_data:
        c.drawString(50, current_y, f"・歩行リズム (Cadence): {metrics_data['cadence']:.1f} 歩/分")
        current_y -= 18
        c.drawString(50, current_y, f"・検出歩数: {metrics_data['steps']} 歩")
        current_y -= 18

        cv_val = metrics_data.get("cv_score", 0.0)
        sway_val = metrics_data.get("sway_score", 0.0)
        lean_mean = metrics_data.get("trunk_lean_mean", 0.0)
        asym = metrics_data.get("asymmetry_percent", 0.0)
        gait_speed = metrics_data.get("gait_speed_m_s", 0.0)

        c.drawString(50, current_y, f"・歩行の変動性 (CV): {cv_val:.3f} （目安: ~0.05・推測です）")
        current_y -= 18
        c.drawString(50, current_y, f"・骨盤の動揺 (Sway): {sway_val:.3f} （目安: ~0.08・推測です）")
        current_y -= 18
        c.drawString(50, current_y, f"・左右差 (Step間隔): {asym:.1f}% （推測です）")
        current_y -= 18
        c.drawString(50, current_y, f"・推定歩行速度: {gait_speed:.2f} m/s （推定値・推測です）")
        current_y -= 18
        c.drawString(50, current_y, f"・平均体幹前傾角度: {lean_mean:.1f} 度 （推測です）")
        current_y -= 24

        if risk_label:
            c.setFont(font_name, 12)
            c.drawString(50, current_y, f"◎ 総合リスク評価: {risk_label}")
            current_y -= 26
            c.setFont(font_name, 11)

    elif "f_met" in metrics_data:
        f_met = metrics_data.get("f_met")
        s_met = metrics_data.get("s_met")
        if f_met:
            c.drawString(50, current_y, f"・頭部の傾き: {f_met['head_tilt']:.1f}°")
            current_y -= 18
            c.drawString(50, current_y, f"・肩の傾き: {f_met['shoulder_slope']:.1f}°")
            current_y -= 18
        if s_met:
            c.drawString(50, current_y, f"・FHPスコア: {s_met['forward_head_score']:.1f}（頭部前方偏位・推測です）")
            current_y -= 18
            c.drawString(50, current_y, f"・体幹前傾角度: {s_met['trunk_lean']:.1f}°")
            current_y -= 24

    # フィードバック
    c.setFont(font_name, 14)
    c.drawString(40, current_y, "■ 分析・評価コメント")
    current_y -= 30
    c.setFont(font_name, 11)

    for fb in feedback_data:
        if current_y < 80:
            c.showPage()
            current_y = page_h - 60
            c.setFont(font_name, 11)

        title_str = f"● {fb['title']}"
        if fb.get('priority'):
            title_str += " 【優先改善】"
            c.setFillColorRGB(0.7, 0, 0)
        else:
            c.setFillColorRGB(0, 0, 0)
        c.drawString(50, current_y, title_str)
        current_y -= 18

        c.setFillColorRGB(0, 0, 0)
        c.setFont(font_name, 10)
        c.drawString(60, current_y, f"状態: {fb['detail']}")
        current_y -= 15
        c.drawString(60, current_y, f"原因: {fb['cause']}")
        current_y -= 22
        c.setFont(font_name, 11)

    current_y -= 10

    # 推奨エクササイズ
    if exercises:
        if current_y < 80:
            c.showPage()
            current_y = page_h - 60
        c.setFont(font_name, 14)
        c.drawString(40, current_y, "■ あなたへの処方箋 (推奨運動)")
        current_y -= 30
        c.setFont(font_name, 11)
        for ex in exercises:
            if current_y < 60:
                c.showPage()
                current_y = page_h - 60
                c.setFont(font_name, 11)
            c.drawString(50, current_y, f"□ {ex}")
            current_y -= 18

    c.showPage()
    c.save()
    b.seek(0)
    return b

# ========== メインアプリケーション ==========

# A. 静止画モード
if app_mode == "静止画：姿勢分析 (立位/座位)":
    st.info("📸 正面・側面それぞれの写真をアップロードしてください")
    posture_type = st.radio("姿勢タイプ", ["立位 (Standing)", "座位 (Sitting)"], horizontal=True)

    col_f, col_s = st.columns(2)
    with col_f:
        file_f = st.file_uploader("正面画像", type=['jpg', 'png', 'jpeg'])
    with col_s:
        file_s = st.file_uploader("側面画像", type=['jpg', 'png', 'jpeg'])

    if st.button("🚀 姿勢分析を実行"):
        if not file_f and not file_s:
            st.error("画像をアップロードしてください。")
        else:
            f_img, f_met, s_img, s_met = None, None, None, None
            snapshot_for_pdf = None

            if file_f:
                img = np.array(Image.open(file_f))
                f_img, f_met = analyze_static_image(img, "front", posture_type)
                snapshot_for_pdf = Image.fromarray(cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB))
            if file_s:
                img = np.array(Image.open(file_s))
                s_img, s_met = analyze_static_image(img, "side", posture_type)
                if snapshot_for_pdf is None and s_img is not None:
                    snapshot_for_pdf = Image.fromarray(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB))

            c1, c2 = st.columns(2)
            with c1:
                if f_img is not None:
                    st.image(f_img, caption="正面", use_container_width=True)
            with c2:
                if s_img is not None:
                    st.image(s_img, caption="側面", use_container_width=True)

            metrics_pack = {"f_met": f_met, "s_met": s_met}
            fb_data, ex_list = generate_clinical_feedback(metrics_pack, "static", age=client_age)

            st.markdown("### 👨‍⚕️ AI姿勢レポート")
            for item in fb_data:
                if item.get('priority'):
                    st.error(f"⚠️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")
                else:
                    st.info(f"ℹ️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")

            st.markdown("#### 🧘 推奨エクササイズ")
            for ex in ex_list:
                st.success(f"✅ {ex}")

            pdf = create_comprehensive_pdf(
                "姿勢分析レポート",
                client_name,
                fb_data,
                ex_list,
                metrics_pack,
                snapshot_for_pdf,
                risk_label=None
            )
            st.download_button("📄 レポート保存 (PDF)", pdf, "posture_report.pdf", "application/pdf")

# B. 動画モード（Pro / Lite 共通ロジック）
else:
    st.info("🎥 歩行動画（全身が映っているもの）をアップロードしてください")
    file_v = st.file_uploader("Video", type=['mp4', 'mov'])

    if st.button("🚀 歩行分析を実行") and file_v:
        path_out, metrics, snapshots = process_video_advanced(file_v, height_cm=client_height_cm)

        if path_out and metrics:
            st.video(path_out)

            # 総合リスク★
            risk_label, risk_star = get_risk_stars(
                metrics.get("cv_score", 0.0),
                metrics.get("sway_score", 0.0),
                metrics.get("asymmetry_percent", 0.0),
                client_age
            )

            st.markdown("### 📊 歩行ドック診断結果")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("ケイデンス", f"{metrics['cadence']:.1f} 歩/分")
            with c2:
                st.metric("体幹の安定性 (Sway)", f"{metrics['sway_score']:.3f}")
            with c3:
                st.metric("歩行のばらつき (CV)", f"{metrics['cv_score']:.3f}")

            st.metric("総合リスク評価", risk_label)

            st.markdown("#### 代表的なシーン（推測です）")
            sc1, sc2 = st.columns(2)
            if snapshots.get("ml"):
                with sc1:
                    st.image(snapshots["ml"], caption="左右揺れが大きい場面", use_container_width=True)
            if snapshots.get("lean"):
                with sc2:
                    st.image(snapshots["lean"], caption="体幹前傾が強い場面", use_container_width=True)

            fb_data, ex_list = generate_clinical_feedback(metrics, "gait", age=client_age)

            st.markdown("---")
            st.subheader("📝 臨床フィードバック")
            for item in fb_data:
                if item.get('priority'):
                    st.error(f"⚠️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")
                else:
                    st.info(f"ℹ️ **{item['title']}**\n\n{item['detail']}\n\n💡 原因: {item['cause']}")

            st.markdown("#### 🧘 推奨プログラム")
            for ex in ex_list:
                st.success(f"✅ {ex}")

            pdf = create_comprehensive_pdf(
                "歩行機能分析レポート",
                client_name,
                fb_data,
                ex_list,
                metrics,
                snapshots,
                risk_label=risk_label
            )
            st.download_button("📄 詳細レポート保存 (PDF)", pdf, "gait_report_pro.pdf", "application/pdf")
        else:
            st.error("歩行メトリクスを算出できませんでした。撮影条件（明るさ・画角・歩数など）を調整して再度お試しください。")
