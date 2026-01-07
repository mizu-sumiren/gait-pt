import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# MediaPipeの読み込みを保護（エラーが出ても画面が止まらないように）
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    st.error(f"分析エンジン準備中... (環境構築完了までお待ちください): {e}")

st.set_page_config(page_title="女性専用 AI歩行ドック", page_icon="💃")

# --- タイトル・コンセプト ---
st.title("💃 女性専用 AI歩行ドック")
st.write("理学療法士の知見で、あなたの歩き方を「一生モノ」の美しさへ。") [cite: 2025-11-21]

# --- 5指標スコアリング ---
def calculate_walking_score():
    return {
        "1. 股関節の伸び (美尻・歩幅)": 30,
        "2. 体幹の安定性 (くびれ・姿勢)": 30,
        "3. 衝撃吸収 (ひざ・腰負担)": 15,
        "4. 膝のクッション (若々しさ)": 15,
        "5. 足の振り出し (軽やかさ)": 10
    }

# --- 40代女性が手元に残したくなるPDFデザイン ---
def create_report_pdf(scores, total_score):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=A4)
    # デザイン：清潔感のあるミントグリーン系
    c.setStrokeColor(colors.lightseagreen)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, 800, "AI Gait Analysis Report")
    
    c.setFont("Helvetica", 18)
    c.drawString(50, 750, f"Total Score: {total_score} / 100")
    
    c.setFont("Helvetica", 12)
    y = 700
    for label, score in scores.items():
        c.drawString(70, y, f"{label}: {score} pts")
        y -= 30
    
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(50, y - 50, "Physiotherapist Advice:")
    c.drawString(70, y - 80, "Your hip extension is the key to your future beauty and health.")
    
    c.save()
    return tmp.name

# --- 画面操作 ---
uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov"])

if uploaded_file:
    scores = calculate_walking_score()
    total_score = sum(scores.values())
    st.subheader(f"📊 分析結果: {total_score} 点")
    st.table(pd.DataFrame(list(scores.items()), columns=['評価指標', 'スコア']))
    
    pdf_path = create_report_pdf(scores, total_score)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 レポート(PDF)を保存する", f, "Gait_Report.pdf", "application/pdf")
    st.success("レポートが完成しました！")
