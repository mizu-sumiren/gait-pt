import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import io
import math
from datetime import datetime
from PIL import Image

# MediaPipeのエラー回避用インポート
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.utils import ImageReader

# 日本語フォント登録
try:
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    JP_FONT = "HeiseiKakuGo-W5"
except:
    JP_FONT = "Helvetica"

st.set_page_config(page_title="AI姿勢・歩行分析ラボ", page_icon="🏥", layout="wide")

# UIデザインの調整
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- サイドバー設定 ---
st.sidebar.header("⚙️ 分析モード")
app_mode = st.sidebar.radio(
    "モードを選択してください",
    ["動画：歩行分析 (Pro)", "動画：歩行分析 (Lite)", "静止画：姿勢分析 (立位/座位)"]
)

st.sidebar.header("📋 対象者情報")
client_name = st.sidebar.text_input("氏名", "テスト 太郎 様")
client_age = st.sidebar.number_input("年齢", min_value=1, max_value=120, value=45, step=1)
client_gender = st.sidebar.selectbox("性別", ["女性", "男性", "その他"])
client_height_cm = st.sidebar.number_input("身長 (cm)", min_value=100, max_value=250, value=160, step=1)

# 女性専用モードの切り替え
is_female_mode = False
if client_gender == "女性" and "歩行" in app_mode:
    is_female_mode = st.sidebar.checkbox("👩 女性専用・詳細解析（5指標）を適用", value=True)

if "歩行" in app_mode:
    st.title("🏃‍♂️ AI歩行ドック (Clinical Grade)")
    if is_female_mode:
        st.subheader("【女性専用モード：理学療法士監修 5指標スコアリング】")
else:
    st.title("📸 AI姿勢分析ラボ")

# --- 計算共通関数 ---

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- 女性専用：5指標解析ロジック ---

def analyze_female_specific_gait(lms_history, fps, w, h, height_cm):
    if not lms_history or len(lms_history) < 10: return None

    # 接地タイミングの特定
    left_y = [l[27].y if l else 1.0 for l in lms_history]
    right_y = [l[28].y if l else 1.0 for l in lms_history]
    
    def get_peaks(arr):
        p = []
        th = np.percentile(arr, 60)
        for i in range(1, len(arr)-1):
            if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > th:
                p.append(i)
        return p

    l_p = get_peaks(left_y)
    r_p = get_peaks(right_y)
    all_p = sorted([(p, 'L') for p in l_p] + [(p, 'R') for p in r_p])

    scores = {}
    details = {}

    if len(all_p) >= 3:
        step1_range = range(0, all_p[0][0])
        step2_range = range(all_p[0][0], all_p[1][0])
        step3_range = range(all_p[1][0], all_p[2][0])

        # 1. 第1歩：股関節可動域 (30点)
        h_angs = [calculate_angle([l[11].x*w, l[11].y*h], [l[23].x*w, l[23].y*h], [l[25].x*w, l[25].y*h]) for i in step1_range if (l:=lms_history[i])]
        rom_h = max(h_angs) - min(h_angs) if h_angs else 0
        scores['股関節の伸び'] = min(30, (rom_h / 35) * 30)
        details['hip_val'] = rom_h

        # 2. 第3歩：体幹側方動揺 (30点)
        sways = [(l[23].x + l[24].x)/2 for i in step3_range if (l:=lms_history[i])]
        sway_val = np.std(sways) * 100 if sways else 0
        scores['体幹の安定性'] = max(0, 30 - (sway_val * 15))
        details['sway_val'] = sway_val

        # 3. 第2歩：体幹垂直移動 (15点)
        verts = [(l[23].y + l[24].y)/2 for i in step2_range if (l:=lms_history[i])]
        v_mov = (max(verts) - min(verts)) * height_cm if verts else 0
        scores['衝撃吸収'] = min(15, (v_mov / 5) * 15)
        details['vert_val'] = v_mov

        # 4. 第2・3歩：膝可動域 (15点)
        k_angs = [calculate_angle([l[23].x*w, l[23].y*h], [l[25].x*w, l[25].y*h], [l[27].x*w, l[27].y*h]) for i in list(step2_range)+list(step3_range) if (l:=lms_history[i])]
        rom_k = max(k_angs) - min(k_angs) if k_angs else 0
        scores['膝のクッション'] = min(15, (rom_k / 60) * 15)
        details['knee_val'] = rom_k

        # 5. 第1歩：遊脚相率 (10点)
        swing_r = (len(step1_range) / (all_p[1][0] if len(all_p)>1 else 1)) * 100
        scores['足の振り出し'] = min(10, (swing_r / 40) * 10)
        details['swing_val'] = swing_r

    total = sum(scores.values())
    return {"total": total, "details": details, "scores": scores}

# --- 動画解析コア ---

def process_video_optimized(file, height_cm, is_female):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    
    lms_history = []
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret: break
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lms_history.append(res.pose_landmarks.landmark)
            else:
                lms_history.append(None)
            out.write(img)
    cap.release()
    out.release()

    clean_lms = [l for l in lms_history if l is not None]
    if not clean_lms: return None, None, None

    female_results = None
    if is_female:
        female_results = analyze_female_specific_gait(clean_lms, fps, w, h, height_cm)

    return out_path, female_results

# --- PDF生成 ---

def create_female_pdf(name, score_dict):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont(JP_FONT, 20)
    c.drawString(50, 800, f"AI歩行分析レポート")
    c.setFont(JP_FONT, 14)
    c.drawString(50, 770, f"氏名: {name} 様")
    c.drawString(50, 750, f"分析日: {datetime.now().strftime('%Y/%m/%d')}")
    
    c.setFont(JP_FONT, 24)
    c.drawString(50, 700, f"総合スコア: {score_dict['total']:.1f} 点")
    
    y = 650
    c.setFont(JP_FONT, 12)
    c.drawString(50, y, "[ 詳細指標 ]")
    y -= 30
    for k, v in score_dict['scores'].items():
        c.drawString(70, y, f"・{k}: {v:.1f} / {30 if '安定' in k or '伸び' in k else 15 if '振り' not in k else 10} 点")
        y -= 20
    
    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# --- メインUI実行 ---

if "歩行" in app_mode:
    video_file = st.file_uploader("🎥 歩行動画をアップロード (mp4/mov)", type=["mp4", "mov"])
    if st.button("🚀 解析開始") and video_file:
        with st.spinner("PT-AIが分析中..."):
            out_path, female_res = process_video_optimized(video_file, client_height_cm, is_female_mode)
            
        if out_path:
            st.video(out_path)
            if is_female_mode and female_res:
                st.balloons()
                st.header(f"総合スコア: {female_res['total']:.1f} / 100点")
                
                cols = st.columns(5)
                for col, (lab, val) in zip(cols, female_res['scores'].items()):
                    col.metric(lab, f"{val:.1f}")

                pdf = create_female_pdf(client_name, female_res)
                st.download_button("📄 レポート(PDF)を保存", pdf, f"gait_report_{datetime.now().strftime('%Y%m%d')}.pdf")
            else:
                st.success("解析が完了しました")
