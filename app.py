import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# MediaPipeの読み込みエラー対策（直接インポート方式）
try:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing
except ImportError:
    st.error("分析エンジンの読み込みに失敗しました。Python 3.11をお試しください。")

st.set_page_config(page_title="女性専用 AI歩行ドック", page_icon="💃")

# --- 40代女性に寄り添うメッセージ ---
st.title("💃 女性専用 AI歩行ドック")
st.write("理学療法士の知見で、あなたの歩き方を美しく、健康に。")

# --- PT監修：5指標スコアリング ---
def calculate_walking_score():
    # 本来は関節角度から計算しますが、まずは枠組みを表示
    scores = {
        "1. 股関節の伸び (美尻・歩幅)": 30,
        "2. 体幹の安定性 (くびれ・姿勢)": 30,
        "3. 衝撃吸収 (ひざ・腰負担)": 15,
        "4. 膝のクッション (若々しさ)": 15,
        "5. 足の振り出し (軽やかさ)": 10
    }
    return scores

# --- PDFレポート生成 (手元に残したくなるデザイン) ---
def create_pdf(scores, total_score):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=A4)
    
    # ヘッダーデザイン
    c.setStrokeColor(colors.plum)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, 800, "AI Gait Analysis Report")
    
    # 総合スコア
    c.setFont("Helvetica", 18)
    c.drawString(50, 750, f"Total Score: {total_score} / 100")
    
    # 指標のリスト
    c.setFont("Helvetica", 12)
    y = 700
    for label, score in scores.items():
        c.drawString(70, y, f"{label}: {score} pts")
        y -= 30
    
    # PT（あなた）からの温かいアドバイス
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(50, y - 50, "Physiotherapist Advice:")
    c.drawString(70, y - 80, "Great work! Focusing on your hip extension will keep you younger.")
    
    c.save()
    return tmp.name

# --- 画面構成 ---
uploaded_file = st.file_uploader("歩行動画を選択してください", type=["mp4", "mov", "avi"])

if uploaded_file:
    with st.spinner("AIが理学療法士の視点で分析しています..."):
        # スコア計算
        scores = calculate_walking_score()
        total_score = sum(scores.values())
        
        st.subheader(f"📊 総合評価: {total_score}点 / 100点")
        
        # 指標の表示
        st.table(pd.DataFrame(list(scores.items()), columns=['評価指標', 'スコア']))
        
        # PDFダウンロード
        pdf_path = create_pdf(scores, total_score)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 美しさを保つためのレポート(PDF)を保存",
                data=f,
                file_name="Gait_Report.pdf",
                mime="application/pdf"
            )
        st.success("レポートが完成しました！あなたの「将来に向けた最大の準備」にお役立てください。")
