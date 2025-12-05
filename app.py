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
# HeiseiKakuGo-W5 はReportLab標準で使える日本語フォント
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

# --- サイドバー入力 ---
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
    if angle > 180.0: angle = 360 - angle
    return angle

def calculate_slope(a, b):
    if a is None or b is None: return 0
    return math.degrees(math.atan2(a[1]-b[1], a[0]-b[0]))

def calculate_vertical_angle(a, b):
    """垂直線に対する角度（bが上、aが下）"""
    if a is None or b is None: return 0
    return math.degrees(math.atan2(b[0]-a[0], b[1]-a[1]))

def get_risk_stars(cv_score, sway_score, asymmetry_percent, age):
    """総合リスク評価を★5段階で算出（推測です）"""
    risk_score = 0.0

    # 年齢補正閾値（推測）
    cv_threshold = 0.08 if age >= 65 else 0.05
    sway_threshold = 0.12 if age >= 65 else 0.08

    # CV評価
    if cv_score > cv_threshold * 1.5: risk_score += 2
    elif cv_score > cv_threshold: risk_score += 1

    # Sway評価
    if sway_score > sway_threshold * 1.5: risk_score += 2
    elif sway_score > sway_threshold: risk_score += 1

    # 左右差評価
    if asymmetry_percent > 15: risk_score += 2
    elif asymmetry_percent > 8: risk_score += 1

    # 年齢リスク加算
    if age >= 75: risk_score += 1
    elif age >= 65: risk_score += 0.5

    # ★変換
    if risk_score >= 5: return "★☆☆☆☆ 高リスク", 1
    elif risk_score >= 3.5: return "★★☆☆☆ 要注意", 2
    elif risk_score >= 2: return "★★★☆☆ やや注意", 3
    elif risk_score >= 1: return "★★★★☆ 良好", 4
    else: return "★★★★★ 優良", 5

# ========== フィードバック生成 ==========

def generate_clinical_feedback(metrics, analysis_type="gait", age=45):
    fb_list = []
    exercises = []
    
    # === A. 歩行分析 ===
    if analysis_type == "gait":
        cadence = metrics.get('cadence', 0.0)
        sway_score = metrics.get('sway_score', 0.0)
        cv_score = metrics.get('cv_score', 0.0)
        trunk_lean_mean = metrics.get('trunk_lean_mean', 0.0)
        asymmetry_percent = metrics.get('asymmetry_percent', 0.0)
        left_mean = metrics.get('left_step_mean', 0.0)
        right_mean = metrics.get('right_step_mean', 0.0)
        gait_speed = metrics.get('gait_speed_m_s', 0.0)
        
        cv_threshold = 0.08 if age >= 65 else 0.05
        sway_threshold = 0.12 if age >= 65 else 0.08

        # 1. リズム・速度
        if cadence < 95:
            fb_list.append({
                "title": "歩行リズムの低下",
                "detail": f"歩行ペースがゆっくりです（{cadence:.1f}歩/分）。推進力が低下している可能性があります。",
                "cause": "下肢筋力の低下や、転倒への不安感が影響している可能性があります（推測）。"
            })
            exercises.append("椅子座り立ち (下肢筋力強化)")
        elif cadence > 125:
            fb_list.append({
                "title": "小刻み歩行の傾向",
                "detail": f"歩数が多く、歩幅が狭くなっている可能性があります（{cadence:.1f}歩/分）。",
                "cause": "股関節の柔軟性低下や、すり足気味になっていることが考えられます（推測）。"
            })
            exercises.append("大股歩き練習")
        
        # 速度（高齢者向け）
        if gait_speed > 0 and age >= 65 and gait_speed < 1.0:
            fb_list.append({
                "title": "歩行速度低下（高齢者基準）",
                "detail": f"推定速度が {gait_speed:.2f}m/s と、転倒リスク基準（1.0m/s未満）を下回っています（推測）。",
                "cause": "筋力低下や心肺機能の低下が考えられます。",
                "priority": True
            })

        # 2. ばらつき (CV)
        if cv_score > cv_threshold:
            fb_list.append({
                "title": f"歩行周期のばらつき (要注意)",
                "detail": f"一歩ごとのリズムが一定ではありません（CV: {cv_score:.3f}）。",
                "cause": "運動制御能力の低下や、注意機能の分散（考え事など）が影響している可能性があります（推測）。",
                "priority": True
            })
            exercises.append("メトロノーム歩行")

        # 3. 体幹動揺 (Sway)
        if sway_score > sway_threshold:
            fb_list.append({
                "title": f"骨盤の動揺（体幹不安定）",
                "detail": f"骨盤の左右への揺れが大きくなっています（Sway: {sway_score:.3f}）。",
                "cause": "体幹筋（腹圧）の機能不全や、中殿筋の筋力低下が疑われます（推測）。",
                "priority": True
            })
            exercises.append("サイドレッグレイズ / サイドプランク")

        # 4. 左右差
        if asymmetry_percent > 8:
            dominant_side = "右" if right_mean > left_mean else "左"
            other_side = "左" if dominant_side == "右" else "右"
            fb_list.append({
                "title": "左右非対称性（荷重バランス）",
                "detail": f"{dominant_side}足のステップ間隔が広く、{dominant_side}荷重優位です（左右差: {asymmetry_percent:.1f}%）。",
                "cause": f"{other_side}側の筋力低下、または{dominant_side}側への代償的荷重が疑われます（推測）。",
                "priority": asymmetry_percent > 15
            })
            exercises.append(f"{other_side}側 片脚立ち練習")

        # 5. 前傾
        if abs(trunk_lean_mean) > 10:
            direction = "前" if trunk_lean_mean > 0 else "後ろ"
            fb_list.append({
                "title": "体幹の傾き",
                "detail": f"平均して体幹がやや{direction}に傾いています（{trunk_lean_mean:.1f}度）。",
                "cause": "胸椎の後弯や股関節周囲筋のアンバランスが考えられます（推測）。"
            })
            exercises.append("股関節屈筋ストレッチ / 胸椎伸展ストレッチ")

        if not fb_list:
            fb_list.append({
                "title": "良好な歩行パターン",
                "detail": "リズム、安定性、左右バランスともに大きな問題は見られません。",
                "cause": "現在の身体機能を維持しましょう。"
            })

    # === B. 姿勢分析 ===
    else:
        f_met = metrics.get('f_met')
        s_met = metrics.get('s_met')

        if s_met and abs(s_met.get('forward_head_score', 0)) > 5.0:
            fb_list.append({
                "title": "ストレートネック傾向 (FHP)",
                "detail": "頭部が肩よりも前方に突出しています。",
                "cause": "長時間のデスクワークやスマホ操作による緊張（推測）。"
            })
            exercises.append("チンイン (顎引き運動)")

        if s_met and abs(s_met.get('trunk_lean', 0)) > 10:
            fb_list.append({
                "title": "姿勢の崩れ (猫背/反り腰)",
                "detail": "上半身の重心軸が垂直から逸脱しています。",
                "cause": "体幹深層筋の弱化、または股関節屈筋群の短縮（推測）。"
            })
            exercises.append("股関節屈筋ストレッチ")

        if f_met and abs(f_met.get('shoulder_slope', 0)) > 3.0:
            side = "右" if f_met['shoulder_slope'] > 0 else "左"
            fb_list.append({
                "title": f"{side}肩の下がり",
                "detail": f"{side}肩が下がる傾向があります。",
                "cause": "片側荷重や日常姿勢のクセが影響している可能性があります（推測）。"
            })
            exercises.append("肩甲帯周囲のストレッチ")

        if not fb_list:
            fb_list.append({"title": "Good Posture", "detail": "非常に綺麗な姿勢アライメントです。", "cause": "素晴らしい状態です。"})

    return fb_list, list(dict.fromkeys(exercises))

# ========== 歩行解析 (メモリ対策版) ==========
def process_video_optimized(file, height_cm=170):
    """
    動画処理メイン関数
    【重要】全フレームを保存せず、必要なキーフレームのみ保存してメモリ不足を防ぐ
    """
    if not file: return None, None, None

    tfile = tempfile.NamedTemporaryFile(delete=False); tfile.write(file.read())
    cap = cv2.VideoCapture(tfile.name)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # データ蓄積用 (画像は保存しない)
    history_lms = [] 
    left_ankle_y = []
    right_ankle_y = []
    pelvis_sway_list = []
    trunk_lean_list = []
    hip_distances_px = []
    
    # キーフレーム用の一時変数
    max_ml_abs = 0.0
    max_lean_abs = 0.0
    frame_ml = None
    frame_lean = None
    frame_mid = None
    
    frame_count = 0
    total_est = int(cap.get(7))
    mid_idx = total_est // 2

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret: break
            frame_count += 1

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            res = pose.process(img_rgb)
            
            # 描画
            cv2.line(img, (w//2, 0), (w//2, h), (0, 255, 255), 1)

            if res.pose_landmarks:
                mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lms = res.pose_landmarks.landmark
                # history_lmsには座標のみ追加 (画像は追加しない！)
                history_lms.append(lms)
                
                # --- リアルタイム計算 ---
                # 1. 骨盤Sway
                pelvis_mid_x = (lms[23].x + lms[24].x) / 2
                pelvis_sway_list.append(pelvis_mid_x)
                
                # 2. 足首Y (ステップ検知)
                left_ankle_y.append(lms[27].y)
                right_ankle_y.append(lms[28].y)
                
                # 3. 体幹前傾 & 左右偏位
                mid_sh_x = (lms[11].x + lms[12].x) / 2 * w
                mid_hp_x = (lms[23].x + lms[24].x) / 2 * w
                trunk_center_x = (mid_sh_x + mid_hp_x) / 2
                ml_dev = (trunk_center_x - w / 2) / (w / 2)
                
                sh_pt = [mid_sh_x, (lms[11].y + lms[12].y) / 2 * h]
                hp_pt = [mid_hp_x, (lms[23].y + lms[24].y) / 2 * h]
                trunk_lean = calculate_vertical_angle(hp_pt, sh_pt)
                trunk_lean_list.append(trunk_lean)

                # 4. 股関節間距離
                hip_l = np.array([lms[23].x * w, lms[23].y * h])
                hip_r = np.array([lms[24].x * w, lms[24].y * h])
                hip_distances_px.append(np.linalg.norm(hip_l - hip_r))

                # --- キーフレーム更新 (最大値のときだけ画像をコピー保存) ---
                if abs(ml_dev) > max_ml_abs:
                    max_ml_abs = abs(ml_dev)
                    frame_ml = img.copy()
                
                if abs(trunk_lean) > max_lean_abs:
                    max_lean_abs = abs(trunk_lean)
                    frame_lean = img.copy()
                    
                if frame_count == mid_idx:
                    frame_mid = img.copy()

                # 右膝角度表示
                try:
                    def get_c(idx): return [lms[idx].x * w, lms[idx].y * h]
                    knee = calculate_angle(get_c(24), get_c(26), get_c(28))
                    cv2.rectangle(img, (w-220, 0), (w, 60), (255, 255, 255), -1)
                    cv2.putText(img, f"R-Knee: {int(knee)}", (w-200, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                except: pass

            out.write(img)

    cap.release(); out.release()

    # --- 指標計算 ---
    if fps <= 0: fps = 30
    
    # ステップ検知
    def detect_steps(y_list):
        steps = 0; frames = []
        if len(y_list) > 2:
            arr = np.array(y_list)
            thresh = np.percentile(arr, 60)
            for i in range(1, len(arr) - 1):
                if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > thresh:
                    steps += 1; frames.append(i)
        return steps, frames

    l_steps, l_frames = detect_steps(left_ankle_y)
    r_steps, r_frames = detect_steps(right_ankle_y)
    total_steps = l_steps + r_steps
    
    duration = len(history_lms) / fps
    cadence = (total_steps / duration) * 60 if duration > 0 else 0

    # 左右差
    l_mean = r_mean = 0.0
    asym = 0.0
    if len(l_frames) >= 2 and len(r_frames) >= 2:
        l_mean = float(np.mean(np.diff(l_frames)))
        r_mean = float(np.mean(np.diff(r_frames)))
        avg_step = (l_mean + r_mean) / 2
        if avg_step > 0: asym = abs(l_mean - r_mean) / avg_step * 100

    # CV
    cv_score = 0.0
    all_frames = sorted(l_frames + r_frames)
    if len(all_frames) >= 3:
        intervals = np.diff(all_frames)
        m_i = np.mean(intervals); s_i = np.std(intervals)
        if m_i > 0: cv_score = s_i / m_i

    # Sway
    sway_score = float(np.std(pelvis_sway_list)) if pelvis_sway_list else 0.0
    trunk_lean_mean = float(np.mean(trunk_lean_list)) if trunk_lean_list else 0.0
    
    # 速度推定
    speed = 0.0
    if hip_distances_px and total_steps >= 2:
        avg_hip = np.mean(hip_distances_px)
        # 股関節幅=身長*0.2と仮定
        px_per_m = avg_hip / (height_cm * 0.002) 
        est_stride = height_cm * 0.01 * 0.4 # ストライド=身長*0.4
        speed = (cadence / 60) * est_stride if cadence > 0 else 0

    metrics = {
        "cadence": cadence, "steps": total_steps,
        "cv_score": cv_score, "sway_score": sway_score,
        "trunk_lean_mean": trunk_lean_mean,
        "asymmetry_percent": asym,
        "left_step_mean": l_mean, "right_step_mean": r_mean,
        "gait_speed_m_s": speed
    }

    # 画像変換 (BGR -> RGB)
    key_images = {}
    for k, img_data in [("ml", frame_ml), ("lean", frame_lean), ("mid", frame_mid)]:
        if img_data is not None:
            key_images[k] = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
        else:
            key_images[k] = None
            
    return out_path, metrics, key_images

# ========== 静止画解析 ==========
def analyze_static_image(image, view, posture_type):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks: return image, None

        h, w, _ = image.shape
        lms = results.pose_landmarks.landmark
        annotated_image = image.copy()
        cv2.line(annotated_image, (w//2, 0), (w//2, h), (0, 255, 255), 2)
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        def get_p(idx): return [lms[idx].x * w, lms[idx].y * h]
        metrics = {}

        if view == "front":
            metrics['head_tilt'] = calculate_slope(get_p(7), get_p(8))
            metrics['shoulder_slope'] = calculate_slope(get_p(11), get_p(12))
            metrics['hip_slope'] = calculate_slope(get_p(23), get_p(24))
        elif view == "side":
            ear_x = (lms[7].x + lms[8].x) / 2
            shoulder_x = (lms[11].x + lms[12].x) / 2
            metrics['forward_head_score'] = (ear_x - shoulder_x) * 100
            metrics['trunk_lean'] = calculate_vertical_angle(get_p(11), get_p(23))
            if posture_type == "立位 (Standing)":
                metrics['knee_angle'] = calculate_angle(get_p(23), get_p(25), get_p(27))
            else:
                metrics['hip_angle'] = calculate_angle(get_p(11), get_p(23), get_p(25))

        return annotated_image, metrics

# ========== PDF生成 ==========
def create_pdf(title, name, age, gender, feedbacks, star_rating, vid=None, f_stat=None, s_stat=None, gait_images=None):
    b = io.BytesIO()
    c = canvas.Canvas(b, pagesize=A4); page_w, page_h = A4
    font_name = "HeiseiKakuGo-W5"

    today = datetime.now().strftime("%Y/%m/%d")
    c.setFont(font_name, 20); c.drawString(40, page_h - 50, f"{title}")
    c.setFont(font_name, 12); c.drawString(40, page_h - 80, f"氏名: {name} ({age}歳 {gender})")
    if star_rating: c.drawString(350, page_h - 80, f"総合評価: {star_rating}")
    c.drawString(400, page_h - 60, f"判定日: {today}")
    c.line(40, page_h - 90, 550, page_h - 90)

    y = page_h - 120

    # スナップショット
    if gait_images:
        img_w, img_h = 180, 135
        x = 50
        if gait_images.get("ml"):
            try:
                c.drawImage(ImageReader(gait_images["ml"]), x, y - img_h, width=img_w, height=img_h)
                c.setFont(font_name, 9); c.drawString(x, y - img_h - 10, "▲ 左右揺れ最大")
                x += 200
            except: pass
        if gait_images.get("lean"):
            try:
                c.drawImage(ImageReader(gait_images["lean"]), x, y - img_h, width=img_w, height=img_h)
                c.setFont(font_name, 9); c.drawString(x, y - img_h - 10, "▲ 前傾最大")
            except: pass
        y = y - img_h - 30

    # Metrics
    c.setFont(font_name, 14); c.drawString(40, y, "■ 計測データ (Metrics)")
    y -= 25; c.setFont(font_name, 11)

    if vid:
        c.drawString(50, y, f"・ケイデンス: {vid.get('cadence', 0):.1f} 歩/分"); y-=18
        c.drawString(50, y, f"・左右差: {vid.get('asymmetry_percent', 0):.1f} %"); y-=18
        c.drawString(50, y, f"・ばらつき(CV): {vid.get('cv_score', 0):.3f}"); y-=18
        c.drawString(50, y, f"・体幹揺れ(Sway): {vid.get('sway_score', 0):.3f}"); y-=18
        c.drawString(50, y, f"・推定速度: {vid.get('gait_speed_m_s', 0):.2f} m/s"); y-=25
    
    if f_stat or s_stat:
        if f_stat: c.drawString(50, y, f"[正面] 肩傾き: {f_stat['shoulder_slope']:.1f}°"); y-=18
        if s_stat: c.drawString(50, y, f"[側面] 前傾: {s_stat['trunk_lean']:.1f}° / FHP: {s_stat['forward_head_score']:.1f}"); y-=25

    # Feedback
    c.setFont(font_name, 14); c.drawString(40, y, "■ 分析コメント & 推奨運動")
    y -= 25; c.setFont(font_name, 11)

    for fb in feedbacks:
        if y < 60: c.showPage(); y = page_h - 50; c.setFont(font_name, 11)
        title = f"● {fb['title']}"
        if fb.get('priority'): 
            title += " 【優先】"
            c.setFillColorRGB(0.7, 0, 0)
        else: 
            c.setFillColorRGB(0, 0, 0)
        c.drawString(50, y, title); y-=15
        
        c.setFillColorRGB(0, 0, 0); c.setFont(font_name, 10)
        c.drawString(60, y, f"・詳細: {fb['detail']}"); y-=15
        c.drawString(60, y, f"・原因: {fb['cause']}"); y-=20
        c.setFont(font_name, 11)

    c.showPage(); c.save(); b.seek(0)
    return b

# ========== メイン UI ==========
if app_mode == "静止画：姿勢分析 (立位/座位)":
    st.info("📸 正面・側面それぞれの写真をアップロード")
    posture_type = st.radio("姿勢タイプ", ["立位", "座位"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1: f_file = st.file_uploader("正面", type=['jpg','png'])
    with c2: s_file = st.file_uploader("側面", type=['jpg','png'])
    
    if st.button("🚀 実行"):
        if f_file or s_file:
            f_img, f_met, s_img, s_met = None, None, None, None
            if f_file: f_img, f_met = analyze_static_image(np.array(Image.open(f_file)), "front", posture_type)
            if s_file: s_img, s_met = analyze_static_image(np.array(Image.open(s_file)), "side", posture_type)
            
            c1, c2 = st.columns(2)
            with c1: 
                if f_img is not None: st.image(f_img, caption="正面", use_container_width=True)
            with c2: 
                if s_img is not None: st.image(s_img, caption="側面", use_container_width=True)
            
            metrics = {"f_met": f_met, "s_met": s_met}
            fbs, exs = generate_clinical_feedback(metrics, "static", client_age)
            
            st.subheader("👨‍⚕️ 分析レポート")
            for f in fbs: st.info(f"{f['title']}: {f['detail']}")
            st.success("推奨: " + ", ".join(exs))
            
            pdf = create_pdf("姿勢分析レポート", client_name, client_age, client_gender, fbs, None, f_stat=f_met, s_stat=s_met)
            st.download_button("📄 PDF保存", pdf, "posture_report.pdf", "application/pdf")

else: # 動画モード
    c1, c2 = st.columns(2)
    with c1: vf = st.file_uploader("正面動画", type=['mp4','mov'])
    with c2: vs = st.file_uploader("側面動画", type=['mp4','mov'])
    
    if st.button("🚀 実行"):
        # メモリ最適化版を呼び出し
        pf, mf, kf = process_video_optimized(vf, client_height_cm) if vf else (None, None, None)
        ps, ms, ks = process_video_optimized(vs, client_height_cm) if vs else (None, None, None)
        
        main_m = ms if ms else mf
        main_k = ks if ks else kf
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1: 
            if pf: st.video(pf)
        with c2: 
            if ps: st.video(ps)
            
        if main_m:
            risk_label, _ = get_risk_stars(main_m['cv_score'], main_m['sway_score'], main_m['asymmetry_percent'], client_age)
            st.subheader(f"総合評価: {risk_label}")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("速度(推定)", f"{main_m['gait_speed_m_s']:.2f} m/s")
            with c2: st.metric("左右差", f"{main_m['asymmetry_percent']:.1f} %")
            with c3: st.metric("ばらつき(CV)", f"{main_m['cv_score']:.3f}")
            with c4: st.metric("揺れ(Sway)", f"{main_m['sway_score']:.3f}")
            
            # スナップショット表示
            if main_k:
                sc1, sc2 = st.columns(2)
                if main_k.get("ml"):
                    with sc1: st.image(main_k["ml"], caption="最大揺れ", use_container_width=True)
                if main_k.get("lean"):
                    with sc2: st.image(main_k["lean"], caption="最大前傾", use_container_width=True)

            st.subheader("👨‍⚕️ 臨床アドバイス")
            fbs, exs = generate_clinical_feedback(main_m, "gait", client_age)
            for f in fbs:
                if f.get('priority'): st.error(f"⚠️ {f['title']}\n{f['detail']}")
                else: st.info(f"ℹ️ {f['title']}\n{f['detail']}")
            st.success("🧘 推奨: " + ", ".join(exs))
            
            pdf = create_pdf("歩行分析レポート", client_name, client_age, client_gender, fbs, risk_label, vid=main_m, gait_images=main_k)
            st.download_button("📄 PDF保存", pdf, "gait_report.pdf", "application/pdf")
