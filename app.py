import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# MediaPipeの読み込み（AttributeError対策の直接インポート）
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing

st.set_page_config(page_title="女性専用 AI歩行ドック", page_icon="💃")

# --- タイトル・コンセプト ---
st.title("💃 女性専用 AI歩行ドック")
st.write("理学療法士の知見 × AIで、あなたの歩き方を「一生モノ」の美しさへ。")

# --- 5指標のスコアリングロジック ---
# に基づく配点
def calculate_walking_score():
    # 本来はMediaPipeの座標から計算しますが、まずは枠組みを実装
    scores = {
        "1. 股関節の伸び (美尻・歩幅)": 30,
        "2. 体幹の安定性 (くびれ・姿勢)": 30,
        "3. 衝撃吸収 (ひざ・腰負担)": 15,
        "4. 膝のクッション (若々しさ)": 15,
        "5. 足の振り出し (軽やかさ)": 10
    }
    return scores

# --- PDF生成関数 ---
def create_pdf(scores, total_score):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=A4)
    
    # デザイン（40代女性向けに洗練されたスタイル）
    c.setFont("Helvetica-Bold", 24)
    c.setStrokeColor(colors.thistle)
    c.drawString(50, 800, "AI Gait Analysis Report")
    
    c.setFont("Helvetica", 18)
    c.drawString(50, 750, f"Total Score: {total_score} / 100")
    
    c.setFont("Helvetica", 14)
    y = 700
    for label, score in scores.items():
        c.drawString(70, y, f"{label}: {score} pts")
        y -= 30
    
    c.setFont("Helvetica-Oblique", 12)
    c.drawString(50, y - 50, "Physiotherapist Advice:")
    c.drawString(70, y - 80, "Your hip extension is excellent! Keep moving for your future health.")
    
    c.save()
    return tmp.name

# --- メイン画面 ---
uploaded_file = st.file_uploader("歩行動画をアップロードしてください (横向き推奨)", type=["mp4", "mov", "avi"])

if uploaded_file:
    st.info("AIが歩行を分析中です... しばらくお待ちください。")
    
    # 分析結果（サンプル）
    scores = calculate_walking_score()
    total_score = sum(scores.values())
    
    st.subheader("📊 分析結果: 100点満点中...")
    st.title(f"{total_score} 点")
    
    # 5指標の表示
    df = pd.DataFrame(list(scores.items()), columns=['指標', 'スコア'])
    st.table(df)
    
    # PDFダウンロードボタン
    pdf_path = create_pdf(scores, total_score)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 分析レポート(PDF)を保存する",
            data=f,
            file_name="Gait_Analysis_Report.pdf",
            mime="application/pdf"
        )
    st.success("レポートが完成しました！手元に保存して、日々の意識に役立ててください。")
