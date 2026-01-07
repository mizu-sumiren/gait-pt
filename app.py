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
try:
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    JP_FONT = "HeiseiKakuGo-W5"
except:
    JP_FONT = "Helvetica"

# MediaPipeの標準的な初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="AI姿勢・歩行分析ラボ", page_icon="🏥", layout="wide")

# --- サイドバー設定 ---
st.sidebar.header("⚙️ 分析モード")
app_mode = st.sidebar.radio("モード", ["動画：歩行分析 (Pro)", "静止画：姿勢分析"])

st.sidebar.header("📋 対象者情報")
client_name = st.sidebar.text_input("氏名", "テスト 様")
client_gender = st.sidebar.selectbox("性別", ["女性", "男性"])
client_height_cm = st.sidebar.number_input("身長 (cm)", value=160)

is_female_mode = (client_gender == "女性" and "歩行" in app_mode)

if is_female_mode:
    st.title("🏃‍♂️ AI歩行ドック (女性専用・詳細版)")
else:
    st.title("🏃‍♂️ AI歩行ドック")

# --- 女性専用：5指標解析ロジック ---
def analyze_female_specific_gait(lms_history, fps, w, h, height_cm):
    if not lms_history or len(lms_history) < 10: return None
    left_y = [l[27].y if l else 1.0 for l in lms_history]
    right_y = [l[28].y if l else 1.0 for l in lms_history]
    
    def get_peaks(arr):
        p = []
        th = np.percentile(arr, 60)
        for i in range(1, len(arr)-1):
            if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > th: p.append(i)
        return p

    l_p, r_p = get_peaks(left_y), get_peaks(right_y)
    all_p = sorted([(p, 'L') for p in l_p] + [(p, 'R') for p in r_p])
    scores = {}

    if len(all_p) >= 3:
        step1, step2, step3 = range(0, all_p[0][0]), range(all_p[0][0], all_p[1][0]), range(all_p[1][0], all_p[2][0])
        # 1. 股関節可動域 (30)
        h_angs = [calculate_angle([l[11].x*w, l[11].y*h], [l[23].x*w, l[23].y*h], [l[25].x*w, l[25].y*h]) for i in step1 if (l:=lms_history[i])]
        scores['股関節の伸び'] = min(30, ( (max(h_angs)-min(h_angs)) / 35) * 30) if h_angs else 0
        # 2. 体幹揺れ (30)
        sways = [(l[23].x + l[24].x)/2 for i in step3 if (l:=lms_history[i])]
        scores['体幹の安定性'] = max(0, 30 - (np.std(sways)*150)) if sways else 0
        # 3. 垂直移動 (15)
        verts = [(l[23].y + l[24].y)/2 for i in step2 if (l:=lms_history[i])]
        scores['衝撃吸収'] = min(15, (((max(verts)-min(verts))*height_cm) / 5) * 15) if verts else 0
        # 4. 膝可動域 (15)
        k_angs = [calculate_angle([l[23].x*w, l[23].y*h], [l[25].x*w, l[25].y*h], [l[27].x*w, l[27].y*h]) for i in list(step2)+list(step3) if (l:=lms_history[i])]
        scores['膝のクッション'] = min(15, ((max(k_angs)-min(k_angs)) / 60) * 15) if k_angs else 0
        # 5. 遊脚相率 (10)
        scores['足の振り出し'] = min(10, ((len(step1)/(all_p[1][0] if len(all_p)>1 else 1)*100) / 40) * 10)

    return {"total": sum(scores.values()), "scores": scores}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 解析実行 ---
if "歩行" in app_mode:
    video_file = st.file_uploader("🎥 動画をアップロード", type=["mp4", "mov"])
    if st.button("🚀 解析開始") and video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
        lms_history, out_path = [], tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        with mp_pose.Pose(min_detection_confidence=0.5) as pose:
            while cap.isOpened():
                ret, img = cap.read()
                if not ret: break
                res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    lms_history.append(res.pose_landmarks.landmark)
                else: lms_history.append(None)
                out.write(img)
        cap.release(); out.release()
        
        st.video(out_path)
        if is_female_mode:
            res = analyze_female_specific_gait(lms_history, fps, w, h, client_height_cm)
            if res:
                st.header(f"総合スコア: {res['total']:.1f} 点")
                cols = st.columns(5)
                for col, (lab, val) in zip(cols, res['scores'].items()):
                    col.metric(lab, f"{val:.1f}")
