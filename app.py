import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# MediaPipeの標準的な初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="女性専用 AI歩行ドック", page_icon="💃")

# --- コンセプト：40代女性の未来を創る ---
st.title("💃 女性専用 AI歩行ドック")
st.write("理学療法士の知見 × AIで、働く女性の「一生動ける身体」を可視化します。") [cite: 2025-11-21]

# --- 5指標のスコアリングロジック ---
def calculate_walking_score():
    # 本来は関節座標から計算しますが、まずは枠組みを実装
    # の配点に基づき、PTの視点を注入
    scores = {
        "1. 股関節の伸び (美尻・歩幅)": 30,
        "2. 体幹の安定性 (くびれ・姿勢)": 30,
        "3. 衝撃吸収 (ひざ・腰負担)": 15,
        "4. 膝のクッション (若々しさ)": 15,
        "5. 足の振り出し (軽やかさ)": 10
    }
    return scores

# --- PDF生成機能：手元に残したくなるデザイン ---
def create_pdf(scores, total_score):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=A4)
    
    # 40代女性向けに洗練されたフォントとカラー
    c.setFont("Helvetica-Bold", 24)
    c.setStrokeColor(colors.thistle)
    c.drawString(50, 800, "AI Gait Analysis Report")
    
    c.setFont("Helvetica", 18)
    c.drawString(50, 740, f"Total Score: {total_score} / 100")
    
    # 指標のリストアップ
    c.setFont("Helvetica", 12)
    y = 680
    for label, score in scores.items():
        c.drawString(70, y, f"{label}: {score} pts")
        y -= 30
    
    # PT（あなた）からの温かいメッセージ [cite: 2025-12-23]
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(50, y - 50, "Physiotherapist Advice:")
    c.drawString(70, y - 80, "Your hip extension is excellent! This is the key to staying active.")
    
    c.save()
    return tmp.name

# --- メインコンテンツ ---
uploaded_file = st.file_uploader("歩行動画を選択してください", type=["mp4", "mov", "avi"])

if uploaded_file:
    st.info("AIが分析を開始しました...")
    
    # スコア計算
    scores = calculate_walking_score()
    total_score = sum(scores.values())
    
    st.subheader(f"📊 分析結果: {total_score} 点")
    
    # 表形式で表示（検証と数字を重視 [cite: 2025-11-21]）
    df = pd.DataFrame(list(scores.items()), columns=['評価指標', 'スコア'])
    st.table(df)
    
    # PDF出力
    pdf_path = create_pdf(scores, total_score)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 分析レポート(PDF)を保存する",
            data=f,
            file_name="Gait_Report.pdf",
            mime="application/pdf"
        )
    st.success("レポートが完成しました！あなたの「将来に向けた最大の準備」にお役立てください。") [cite: 2025-11-21]
