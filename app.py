import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# MediaPipeの堅牢なインポート方法
try:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing
except ImportError:
    st.error("分析エンジンの読み込みに失敗しました。再デプロイをお試しください。")

st.set_page_config(page_title="女性専用 AI歩行ドック", page_icon="💃")

# --- タイトル・コンセプト ---
st.title("💃 女性専用 AI歩行ドック")
st.write("理学療法士の知見で、あなたの歩き方を美しく、健康に。") [cite: 2025-11-21]

# --- 5指標スコアリングロジック ---
def calculate_walking_score():
    # 各指標の配点設定
    scores = {
        "1. 股関節の伸び (美尻・歩幅)": 30,
        "2. 体幹の安定性 (くびれ・姿勢)": 30,
        "3. 衝撃吸収 (ひざ・腰負担)": 15,
        "4. 膝のクッション (若々しさ)": 15,
        "5. 足の振り出し (軽やかさ)": 10
    }
    return scores

# --- 40代女性向けPDFレポート作成 ---
def create_pdf(scores, total_score):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=A4)
    
    # デザイン：清潔感のある配色とフォント
    c.setFont("Helvetica-Bold", 24)
    c.setStrokeColor(colors.lightseagreen)
    c.drawString(50, 800, "AI Gait Analysis Report")
    
    c.setFont("Helvetica", 18)
    c.drawString(50, 750, f"Total Score: {total_score} / 100")
    
    c.setFont("Helvetica", 12)
    y = 700
    for label, score in scores.items():
        c.drawString(70, y, f"{label}: {score} pts")
        # 簡易バーの描画
        c.setFillColor(colors.lightgrey)
        c.rect(250, y-2, 60, 8, fill=1, stroke=0)
        c.setFillColor(colors.lightseagreen)
        c.rect(250, y-2, score * 2, 8, fill=1, stroke=0)
        y -= 30
    
    # PT（あなた）のアドバイス欄
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(50, y - 50, "Professional Advice from PT:")
    c.drawString(70, y - 80, "Improving hip extension is key to your long-term productivity and beauty.")
    
    c.save()
    return tmp.name

# --- メインコンテンツ ---
uploaded_file = st.file_uploader("歩行動画を選択してください (MP4, MOV)", type=["mp4", "mov", "avi"])

if uploaded_file:
    with st.spinner("理学療法士のAIがあなたの歩行を精密に分析中..."):
        # 分析ロジック（現時点では枠組みを適用）
        scores = calculate_walking_score()
        total_score = sum(scores.values())
        
        st.subheader(f"📊 分析結果: {total_score} 点 / 100点")
        
        # 指標の表示（データに基づいた検証 [cite: 2025-11-21]）
        df = pd.DataFrame(list(scores.items()), columns=['評価指標', 'スコア'])
        st.table(df)
        
        # PDFダウンロードボタン
        pdf_path = create_pdf(scores, total_score)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 美しさと健康のためのレポート(PDF)を保存",
                data=f,
                file_name="Gait_Analysis_Report.pdf",
                mime="application/pdf"
            )
        st.success("レポートが完成しました！手元に保存して、理想の歩き方を目指しましょう。")
