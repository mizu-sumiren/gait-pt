import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import japanize_matplotlib
import math

# --- ページ設定 ---
st.set_page_config(page_title="統合歩行分析レポート (PT Pro)", page_icon="🛡️", layout="wide")

st.title("🛡️ 統合歩行・身体機能分析レポート")
st.markdown("すべての対象者に対応：身体機能の弱点と歩行の崩れを自動リンクさせます。")

# --- サイドバー：詳細な機能チェック ---
st.sidebar.header("📋 測定データ入力")

with st.sidebar.expander("1. 問診・痛み情報", expanded=True):
    pain_areas = st.multiselect(
        "痛み・違和感のある部位",
        ["特になし", "首", "肩", "腰", "股関節(右)", "股関節(左)", "膝(右)", "膝(左)", "足首・足部"]
    )
    history = st.text_area("特記事項 (既往歴など)")

with st.sidebar.expander("2. 機能測定結果", expanded=True):
    st.caption("対象者の数値を入力してください")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**左側 (Left)**")
        grip_l = st.number_input("握力(左) kg", value=20.0)
        hip_flex_l = st.number_input("股屈曲(左) kgf/kg", value=0.9) # 試しに低くしてみる
        one_leg_l = st.number_input("片脚立位(左) 秒", value=15)    # 試しに低くしてみる
        toe_grip_l = st.number_input("足趾把持(左) %", value=10.0)  # 試しに低くしてみる
    with col_s2:
        st.markdown("**右側 (Right)**")
        grip_r = st.number_input("握力(右) kg", value=25.0)
        hip_flex_r = st.number_input("股屈曲(右) kgf/kg", value=1.2)
        one_leg_r = st.number_input("片脚立位(右) 秒", value=60)
        toe_grip_r = st.number_input("足趾把持(右) %", value=20.0)

    st.markdown("---")
    frt = st.number_input("FRT (cm)", value=25.0)
    ffd = st.number_input("FFD (cm)", value=0.0)
    seat_step = st.number_input("座位ステップ (回/20秒)", value=30)

# --- ロジック関数：臨床推論エンジン ---
def generate_clinical_feedback(data):
    """
    入力された数値データに基づき、PT視点のフィードバックを自動生成する関数
    """
    feedback = []
    
    # データ展開
    pain = data['pain']
    toe_l, toe_r = data['toe_l'], data['toe_r']
    hip_l, hip_r = data['hip_l'], data['hip_r']
    ols_l, ols_r = data['ols_l'], data['ols_r']
    frt, ffd = data['frt'], data['ffd']
    step = data['seat_step']
    
    # 1. 足趾機能 (Toe Grip)
    avg_toe = (toe_l + toe_r) / 2
    if avg_toe < 15:
        level = "機能低下" if avg_toe < 10 else "出力不足・硬さ"
        feedback.append(f"**【足指：{level} (平均{avg_toe:.1f}%)】**\n足指の力が基準を下回っています。歩行時に「後ろ足の指で地面を蹴る」動きが弱くなり、**足が外に流れる、あるいはペタペタ歩き**の原因になります。ふくらはぎの張りや、つまづきの原因になりやすいポイントです。")
    
    # 2. 股関節屈曲筋力 (Hip Flexion)
    # A. 絶対値の低さ
    if hip_l < 1.0 or hip_r < 1.0:
        weak_side = "左" if hip_l < hip_r else "右"
        feedback.append(f"**【股関節：振り出しの弱さ ({weak_side}側)】**\n股関節を引き上げる力（腸腰筋）が弱まっています。これにより**歩幅が小さくなる、すり足になる**リスクがあります。段差でのつまづきに注意が必要です。")
    
    # B. 左右差と痛みのリンク
    diff_hip = abs(hip_l - hip_r)
    if diff_hip > 0.2:
        weaker = "左" if hip_l < hip_r else "右"
        stronger = "右" if hip_l < hip_r else "左"
        mechanism = f"**【左右差：{weaker}側の弱さと代償】**\n股関節の筋力に明確な左右差があります。**弱い{weaker}脚を前に出すのが遅れるため、反対側の{stronger}脚で身体を支える時間が長くなります。**"
        
        # 痛みが強い側にある場合
        if any(stronger in p for p in pain):
            mechanism += f"\n👉 これが、現在**{stronger}側に痛みが出ている根本原因**の可能性があります（弱い方をかばって、強い方が過労状態です）。"
        
        feedback.append(mechanism)

    # 3. バランス能力 (OLS & FRT)
    if ols_l < 20 or ols_r < 20:
        unstable = "左" if ols_l < 20 else "右"
        if ols_l < 20 and ols_r < 20: unstable = "両"
        feedback.append(f"**【バランス：立脚期のふらつき ({unstable}側)】**\n片脚立ちの秒数が短くなっています。歩行中、片足に体重が乗った瞬間に**骨盤が横に逃げる（スウェイ）**動きが出現しやすく、これが腰や膝への負担（メカニカルストレス）となります。")

    if frt < 30:
        feedback.append(f"**【重心移動：前方への不安 (FRT {frt}cm)】**\nFRTが短く、身体を前に預ける能力が低下しています。転倒を怖がって**「腰が引けた姿勢」や「小刻み歩行」**になりがちです。")

    # 4. 柔軟性 (FFD)
    if ffd < -5:
        feedback.append(f"**【柔軟性：タイトネス (FFD {ffd}cm)】**\n身体の背面（ハムストリングス・腰背部）が硬いです。骨盤が後傾しやすく、**膝が曲がったまま歩く原因**になります。")
    elif ffd > 10 and (ols_l < 20 or ols_r < 20): # 柔らかいのにバランス悪い
        feedback.append(f"**【柔軟性：関節不安定性 (FFD {ffd}cm)】**\n身体は非常に柔らかいですが、それを支える筋力が不足している可能性があります（関節が緩い状態）。ストレッチよりも**「筋肉で関節を固める（安定させる）」トレーニング**が重要です。")

    # 5. アジリティ (Seat Step)
    if step < 40:
        feedback.append(f"**【俊敏性：反応の遅れ ({step}回)】**\n素早く動く能力が低下しています。とっさの時に足が出にくいため、転倒予防のためにリズム運動を取り入れましょう。")

    if not feedback:
        feedback.append("✅ **素晴らしい状態です！**\n目立った機能低下は見当たりません。現在の活動量を維持しましょう。")

    return feedback

# --- 動画処理関数（前回と同じ） ---
def draw_grid_and_skeleton(image, results):
    h, w, _ = image.shape
    color_grid = (200, 200, 200)
    center_x = w // 2
    cv2.line(image, (center_x, 0), (center_x, h), (0, 255, 255), 1) 
    for x in range(0, w, w//8):
        if x != center_x: cv2.line(image, (x, 0), (x, h), color_grid, 1)
    for y in range(0, h, h//6):
        cv2.line(image, (0, y), (w, y), color_grid, 1)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        params = [(11, 12), (23, 24), (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 31), (28, 32)]
        def get_p(idx): return int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        for s, e in params: cv2.line(image, get_p(s), get_p(e), (255, 255, 255), 3)
        keypoints = [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32]
        for k in keypoints:
            color = (0, 0, 255) if k % 2 == 0 else (255, 0, 0)
            if k == 0: color = (0, 255, 255)
            cv2.circle(image, get_p(k), 6, color, -1)
    return image

def process_video(uploaded_file):
    if uploaded_file is None: return None
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = draw_grid_and_skeleton(image, results)
            out.write(image)
    cap.release()
    out.release()
    return output_path

# --- メインレイアウト ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("① 正面動画")
    file_front = st.file_uploader("Front View", type=['mp4', 'mov'], key="f")
with col2:
    st.subheader("② 側面動画")
    file_side = st.file_uploader("Side View", type=['mp4', 'mov'], key="s")

if st.button("🚀 汎用分析を実行"):
    path_f = process_video(file_front) if file_front else None
    path_s = process_video(file_side) if file_side else None
    
    st.markdown("---")
    
    # 1. 動画表示
    c1, c2 = st.columns(2)
    with c1:
        if path_f: st.video(path_f)
    with c2:
        if path_s: st.video(path_s)

    # 2. 自動フィードバック生成
    st.header("👨‍⚕️ AI理学療法士のフィードバック")
    
    # データ辞書作成
    input_data = {
        'pain': pain_areas,
        'toe_l': toe_grip_l, 'toe_r': toe_grip_r,
        'hip_l': hip_flex_l, 'hip_r': hip_flex_r,
        'ols_l': one_leg_l, 'ols_r': one_leg_r,
        'frt': frt, 'ffd': ffd, 'seat_step': seat_step
    }
    
    # ロジック実行
    feedbacks = generate_clinical_feedback(input_data)
    
    # 表示
    for msg in feedbacks:
        st.info(msg)

    # 3. 汎用的な推奨運動
    st.subheader("🏋️‍♀️ 推奨される運動プログラム")
    
    # 弱点に応じたメニュー表示
    if (toe_grip_l + toe_grip_r)/2 < 15:
        st.markdown("- **足指強化**: タオルギャザー、足指じゃんけん（足指で蹴る感覚を養う）")
    if hip_flex_l < 1.0 or hip_flex_r < 1.0:
        st.markdown("- **腸腰筋強化**: 椅子に座っての腿上げ（ニーアップ）、大股歩き")
    if one_leg_l < 20 or one_leg_r < 20:
        st.markdown("- **中殿筋・バランス**: キッチンでの片脚立ち保持（1分間）、ヒップアブダクション")
    if frt < 30:
        st.markdown("- **動的バランス**: 重心移動練習（前後左右へのステップ）")
