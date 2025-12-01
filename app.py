import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import math

# --- ページ設定 ---
st.set_page_config(page_title="転倒予防・身体機能分析 AI (PT Pro)", page_icon="🛡️", layout="wide")

st.title("🛡️ 転倒予防・身体機能分析レポート")
st.markdown("転倒リスク特化型：数値データと動画分析からリスクを自動判定します。")

# ==========================================
# 1. サイドバー入力（転倒リスク項目を追加）
# ==========================================
st.sidebar.header("📋 測定データ入力")

with st.sidebar.expander("1. 基本情報・リスク因子", expanded=True):
    age = st.sidebar.number_input("年齢", value=75)
    history_fall = st.sidebar.radio("過去1年間の転倒歴", ["なし", "あり (1回)", "あり (複数回)"])
    pain_areas = st.multiselect(
        "痛み・違和感のある部位",
        ["特になし", "首", "肩", "腰", "股関節(右)", "股関節(左)", "膝(右)", "膝(左)", "足首・足部"]
    )
    
    st.markdown("**歩行観察 (セラピスト評価)**")
    gait_sway = st.checkbox("歩行時の動揺・ふらつきが強い")
    gait_speed_slow = st.checkbox("歩行速度が著しく遅い")

with st.sidebar.expander("2. 機能測定結果 (数値入力)", expanded=True):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**左側 (Left)**")
        grip_l = st.number_input("握力(左) kg", value=20.0)
        hip_flex_l = st.number_input("股屈曲(左) kgf/kg", value=0.9)
        one_leg_l = st.number_input("片脚立位(左) 秒", value=4.0) # デモ用に低く設定
        toe_grip_l = st.number_input("足趾把持(左) %", value=10.0)
    with col_s2:
        st.markdown("**右側 (Right)**")
        grip_r = st.number_input("握力(右) kg", value=25.0)
        hip_flex_r = st.number_input("股屈曲(右) kgf/kg", value=1.2)
        one_leg_r = st.number_input("片脚立位(右) 秒", value=60.0)
        toe_grip_r = st.number_input("足趾把持(右) %", value=20.0)

    st.markdown("---")
    frt = st.number_input("FRT (cm)", value=25.0)
    ffd = st.number_input("FFD (cm)", value=0.0)
    seat_step = st.number_input("座位ステップ (回/20秒)", value=30)

# ==========================================
# 2. ロジック関数：スコア化とリスク判定
# ==========================================

def calculate_score_1to5(value, item_name, sex='female'):
    """
    生データを1-5点のスコアに変換する関数
    ※基準値は一般的な目安です。先生の現場の基準に合わせて調整してください。
    """
    score = 3 # デフォルト
    
    # 片脚立位 (秒) - 転倒リスクの重要指標
    if item_name == 'ols':
        if value >= 30: score = 5
        elif value >= 15: score = 4
        elif value >= 10: score = 3
        elif value >= 5: score = 2
        else: score = 1
            
    # FRT (cm)
    elif item_name == 'frt':
        if value >= 35: score = 5
        elif value >= 30: score = 4
        elif value >= 25: score = 3
        elif value >= 15: score = 2
        else: score = 1
        
    # 座位ステップ (回)
    elif item_name == 'step':
        if value >= 50: score = 5
        elif value >= 40: score = 4
        elif value >= 30: score = 3
        elif value >= 20: score = 2
        else: score = 1

    # 股関節屈曲筋力 (kgf/kg)
    elif item_name == 'hip':
        if value >= 1.5: score = 5
        elif value >= 1.2: score = 4
        elif value >= 0.9: score = 3
        elif value >= 0.6: score = 2
        else: score = 1

    # 握力 (kg) - 女性想定
    elif item_name == 'grip':
        if value >= 25: score = 5
        elif value >= 22: score = 4
        elif value >= 18: score = 3
        elif value >= 15: score = 2
        else: score = 1
        
    # 足趾把持力 (%)
    elif item_name == 'toe':
        if value >= 25: score = 5
        elif value >= 20: score = 4
        elif value >= 15: score = 3
        elif value >= 10: score = 2
        else: score = 1

    # FFD (cm) - 柔軟性は極端に低い場合のみ低スコア
    elif item_name == 'ffd':
        if value >= 5: score = 5
        elif value >= 0: score = 4
        elif value >= -5: score = 3
        elif value >= -15: score = 2
        else: score = 1

    return score

def assess_fall_risk(scores, history_fall, gait_sway):
    """
    転倒リスク判定ロジック
    条件：
    1. 転倒歴あり
    2. 歩行時の動揺あり
    3. スコア2以下の項目が存在する
    """
    reasons = []
    risk_level = "低リスク (予防)"
    alert_color = "success"

    # 条件チェック
    has_low_score = any(s <= 2 for s in scores.values())
    has_fall_history = "あり" in history_fall
    
    # 判定ロジック
    if has_fall_history or gait_sway or has_low_score:
        risk_level = "⚠️ 高リスク (要対策)"
        alert_color = "error" # 赤色
        
        if has_fall_history:
            reasons.append(f"・過去の転倒歴 ({history_fall})")
        if gait_sway:
            reasons.append("・歩行時の動揺・ふらつき")
        if has_low_score:
            low_items = [k for k, v in scores.items() if v <= 2]
            reasons.append(f"・機能低下項目あり (スコア2以下: {', '.join(low_items)})")
    
    return risk_level, alert_color, reasons

# ==========================================
# 3. 動画処理関数 (MediaPipe)
# ==========================================
mp_pose = mp.solutions.pose
def process_video_and_draw(uploaded_file):
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
            
            # グリッド線
            h, w, _ = image.shape
            cv2.line(image, (w//2, 0), (w//2, h), (0, 255, 255), 1) 
            
            # 骨格描画
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            out.write(image)
            
    cap.release()
    out.release()
    return output_path

# ==========================================
# 4. メインレイアウト & 実行ボタン
# ==========================================
col1, col2 = st.columns(2)
with col1:
    st.subheader("① 正面動画")
    file_front = st.file_uploader("Front View", type=['mp4', 'mov'], key="f")
with col2:
    st.subheader("② 側面動画")
    file_side = st.file_uploader("Side View", type=['mp4', 'mov'], key="s")

if st.button("🚀 転倒リスク分析を実行"):
    
    # --- A. スコア計算 ---
    # 左右あるものは低い方を採用してリスク管理する（または平均でも可）
    scores = {
        '握力': calculate_score_1to5(min(grip_l, grip_r), 'grip'),
        '股屈曲': calculate_score_1to5(min(hip_flex_l, hip_flex_r), 'hip'),
        '片脚立位': calculate_score_1to5(min(one_leg_l, one_leg_r), 'ols'),
        '足趾把持': calculate_score_1to5(min(toe_grip_l, toe_grip_r), 'toe'),
        'FRT': calculate_score_1to5(frt, 'frt'),
        '柔軟性(FFD)': calculate_score_1to5(ffd, 'ffd'),
        '敏捷性(Step)': calculate_score_1to5(seat_step, 'step')
    }

    # --- B. リスク判定 ---
    risk_level, color, risk_reasons = assess_fall_risk(scores, history_fall, gait_sway)
    
    # --- C. 結果表示 ---
    st.markdown("---")
    st.header("📊 分析結果レポート")

    # 1. リスク判定結果（アラート表示）
    if color == "error":
        st.error(f"## 判定: {risk_level}")
        st.markdown("**【リスク要因】**")
        for r in risk_reasons:
            st.markdown(f"- {r}")
    else:
        st.success(f"## 判定: {risk_level}")
        st.markdown("現在のところ、高い転倒リスク要因は見当たりません。")

    # 2. レーダーチャート可視化 (Matplotlib)
    # 日本語フォント問題回避のため英語ラベルを使用（必要に応じて日本語フォント設定を追加してください）
    labels = list(scores.keys())
    values = list(scores.values())
    
    # チャートを閉じるために最初のデータを最後に追加
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels) # 日本語が表示されない場合は labels_en などに変更
    ax.set_ylim(0, 5)
    st.pyplot(fig)

    # 3. 具体的なフィードバック (低スコア項目への対策)
    st.subheader("💡 改善のための運動処方")
    
    suggestions = []
    if scores['片脚立位'] <= 2:
        suggestions.append("**【バランス低下】**: 支持物ありでの片足立ち（1分間×3セット）。お尻の横（中殿筋）を意識してください。")
    if scores['股屈曲'] <= 2:
        suggestions.append("**【足の振り出し弱さ】**: 座っての足踏み運動（腸腰筋）、またぎ動作練習。")
    if scores['足趾把持'] <= 2:
        suggestions.append("**【足指の機能不全】**: タオルギャザー、足指じゃんけん。踏ん張る力を強化します。")
    if scores['敏捷性(Step)'] <= 2:
        suggestions.append("**【反応速度の低下】**: 椅子からの素早い立ち座り、前後ステップ練習。")
    
    if not suggestions:
        st.info("特定の機能低下は見られません。スクワットやウォーキングで全身持久力を維持しましょう。")
    else:
        for s in suggestions:
            st.warning(s)

    # 4. 動画処理結果の表示
    if file_front or file_side:
        st.subheader("🎥 歩行・姿勢分析")
        c1, c2 = st.columns(2)
        if file_front:
            path_f = process_video_and_draw(file_front)
            c1.video(path_f)
        if file_side:
            path_s = process_video_and_draw(file_side)
            c2.video(path_s)

