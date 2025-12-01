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
st.markdown("身体機能データの「左右差」や「弱点」が、歩行や痛みにどう影響しているかを分析します。")

# --- サイドバー：詳細な機能チェック（画像の数値をデフォルト設定） ---
st.sidebar.header("📋 測定データ入力")

with st.sidebar.expander("1. 問診・痛み情報", expanded=True):
    # デフォルトで「股関節(右)」を選択状態に
    pain_areas = st.multiselect(
        "痛み・違和感のある部位",
        ["特になし", "首", "肩", "腰", "股関節(右)", "股関節(左)", "膝(右)", "膝(左)", "足首・足部"],
        default=["股関節(右)"]
    )
    history = st.text_area("特記事項", value="右股関節に硬さと痛みあり。全体的な身体機能は高い。")

with st.sidebar.expander("2. 機能測定結果 (画像データ反映)", expanded=True):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**左側 (Left)**")
        grip_l = st.number_input("握力(左) kg", value=28.6)
        hip_flex_l = st.number_input("股屈曲(左) kgf/kg", value=1.21) # 右より低い
        one_leg_l = st.number_input("片脚立位(左) 秒", value=120)
        toe_grip_l = st.number_input("足趾把持(左) %", value=11.0) # 低い
    with col_s2:
        st.markdown("**右側 (Right)**")
        grip_r = st.number_input("握力(右) kg", value=29.0)
        hip_flex_r = st.number_input("股屈曲(右) kgf/kg", value=1.36)
        one_leg_r = st.number_input("片脚立位(右) 秒", value=120)
        toe_grip_r = st.number_input("足趾把持(右) %", value=11.0) # 低い

    st.markdown("---")
    frt = st.number_input("FRT (cm)", value=42.0)
    ffd = st.number_input("FFD (cm)", value=13.6)
    seat_step = st.number_input("座位ステップ (回/20秒)", value=47)

# --- 解析用関数 ---
mp_pose = mp.solutions.pose

def draw_grid_and_skeleton(image, results):
    """グリッドと骨格を描画する関数"""
    h, w, _ = image.shape
    
    # 1. グリッド描画
    color_grid = (200, 200, 200)
    center_x = w // 2
    cv2.line(image, (center_x, 0), (center_x, h), (0, 255, 255), 1) # 黄色の正中線
    # 縦線
    for x in range(0, w, w//8):
        if x != center_x: cv2.line(image, (x, 0), (x, h), color_grid, 1)
    # 横線
    for y in range(0, h, h//6):
        cv2.line(image, (0, y), (w, y), color_grid, 1)

    # 2. 骨格描画（スッキリ版）
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 描画する関節と接続
        params = [
            (11, 12), (23, 24), (11, 23), (12, 24), # 体幹
            (23, 25), (24, 26), (25, 27), (26, 28), # 脚
            (27, 31), (28, 32) # 足
        ]
        
        def get_p(idx): return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

        # 線
        for s, e in params:
            cv2.line(image, get_p(s), get_p(e), (255, 255, 255), 3)
            
        # 点（右：赤、左：青）
        keypoints = [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32]
        for k in keypoints:
            color = (0, 0, 255) if k % 2 == 0 else (255, 0, 0) # 右偶数、左奇数
            if k == 0: color = (0, 255, 255) # 頭
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
            
            # 描画処理
            image = draw_grid_and_skeleton(image, results)
            out.write(image)
            
    cap.release()
    out.release()
    return output_path

# --- メインレイアウト ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("① 正面動画 (Front)")
    file_front = st.file_uploader("正面から撮影", type=['mp4', 'mov'], key="f")
with col2:
    st.subheader("② 側面動画 (Side)")
    file_side = st.file_uploader("横から撮影", type=['mp4', 'mov'], key="s")

if st.button("🚀 専門的分析を実行"):
    # 動画処理（ある場合のみ）
    path_f = process_video(file_front) if file_front else None
    path_s = process_video(file_side) if file_side else None
    
    st.markdown("---")
    
    # 1. 解析結果表示
    c1, c2 = st.columns(2)
    with c1:
        if path_f:
            st.video(path_f)
            st.caption("正面：骨盤の側方動揺と肩のラインを確認")
    with c2:
        if path_s:
            st.video(path_s)
            st.caption("側面：歩幅と蹴り出し（足首の動き）を確認")

    # 2. PTロジックによるフィードバック生成
    st.header("👨‍⚕️ 理学療法士AIによる統合フィードバック")
    
    # ロジック判定
    insights = []
    
    # A. 全体評価
    insights.append(f"**【全体像】**\n片脚立位が{one_leg_r}秒、FRTが{frt}cmと、バランス能力や身体の柔軟性は**非常に高いレベル**にあります。一見すると歩行も綺麗で安定しています。")
    
    # B. 足趾の機能不全について
    if toe_grip_l < 15 and toe_grip_r < 15:
        insights.append(f"**【課題1：足指の機能不全 (11%)】**\n足趾把持力が左右ともに11%と基準値を大きく下回っています。動画でも、後ろ足の踵が浮いた後に**「つま先で地面を蹴る動き」が弱く、足が流れている**ように見受けられます。\n\nこれが弱いと、ふくらはぎや殿筋を使った「前方への推進力」が得られず、**「太ももの前（大腿直筋・腸腰筋）」を使って脚を引き上げる歩き方**になりがちです。")

    # C. 股関節の左右差と痛みのリンク (ここが核心)
    if "股関節(右)" in pain_areas:
        mechanism = ""
        if hip_flex_l < hip_flex_r:
            diff = hip_flex_r - hip_flex_l
            mechanism = f"**【考察：右股関節痛の原因】**\n注目すべきは**「左の股関節屈曲筋力（{hip_flex_l}）」が右（{hip_flex_r}）よりも弱い**ことです。\n\n1. 足指の蹴り出しが弱いため、脚を前に出すには「股関節の引き上げ」が必要です。\n2. しかし、**左の引き上げる力が弱いため、左脚を前に振り出す動作が非効率**になっている可能性があります。\n3. 左脚がスムーズに出ないと、**軸足である「右脚（痛い方）」で身体を支える時間が長くなります**。\n\nつまり、**「左脚の機能不全をカバーするために、硬さのある右股関節が過重労働を強いられている」**可能性が高いです。"
        
        insights.append(mechanism)

    # D. 柔軟性のリスク
    if ffd > 10:
        insights.append(f"**【柔軟性のリスク】**\nFFDが{ffd}cmと非常に柔らかいですが、筋力（特に足指や腸腰筋）が伴っていない場合、**「柔らかすぎて関節で支えてしまう（関節不安定性）」**リスクがあります。右股関節の痛みは、筋肉で支えきれない衝撃が関節包や靭帯にかかっている痛みかもしれません。")

    # 出力
    for txt in insights:
        if txt:
            st.info(txt)

    # 3. 運動処方
    st.subheader("🏋️‍♀️ 改善のための運動プログラム")
    st.markdown(f"""
    この方の痛みを改善し、長く働ける身体を作るには、以下の優先順位をお勧めします。
    
    1.  **足指トレーニング（最優先）**
        * タオルギャザーや足指じゃんけんを徹底し、**「地面を掴んで蹴る」感覚**を取り戻します。これにより、股関節への負担を減らします。
    2.  **左腸腰筋の強化**
        * 右側だけでなく、**「左側」のニーアップ（もも上げ）**を行い、左脚をスパッと前に出せるようにします。これで右足の負担時間を減らします。
    3.  **右中殿筋の安定化**
        * 痛みのある右側は、動かすよりも「支える力」を高めるため、痛みが出ない範囲での片脚立ち保持やヒップアブダクションを行います。
    """)
