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
st.set_page_config(page_title="総合歩行・身体機能分析AI", page_icon="🛡️", layout="wide")

st.title("🛡️ 総合歩行・身体機能分析AI")
st.markdown("正面・側面の歩行動画と、詳細な身体機能データを統合し、理学療法士視点で原因を推論します。")

# --- サイドバー：詳細な機能チェック ---
st.sidebar.header("📋 身体機能・測定データ")

# 1. 基本属性・問診
with st.sidebar.expander("1. 問診・痛み", expanded=True):
    pain_areas = st.multiselect(
        "痛み・違和感のある部位",
        ["特になし", "首", "肩", "腰", "股関節(右)", "股関節(左)", "膝(右)", "膝(左)", "足首・足部"]
    )
    history = st.text_area("既往歴・特記事項", height=60)

# 2. 身体機能測定（写真の項目を網羅）
with st.sidebar.expander("2. 機能測定結果 (入力)", expanded=True):
    st.caption("測定データを入力してください")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        grip_l = st.number_input("握力(左) kg", value=28.6)
        hip_flex_l = st.number_input("股屈曲(左) kgf/kg", value=1.21)
        one_leg_l = st.number_input("片脚立位(左) 秒", value=120)
        toe_grip_l = st.number_input("足趾把持(左) %", value=11.0)
    with col_s2:
        grip_r = st.number_input("握力(右) kg", value=29.0)
        hip_flex_r = st.number_input("股屈曲(右) kgf/kg", value=1.36)
        one_leg_r = st.number_input("片脚立位(右) 秒", value=120)
        toe_grip_r = st.number_input("足趾把持(右) %", value=11.0)

    st.markdown("---")
    frt = st.number_input("FRT (cm)", value=42.0)
    ffd = st.number_input("FFD (cm)", value=13.6)
    seat_step = st.number_input("座位ステップ (回/20秒)", value=47)

# --- 解析用関数群 ---
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def calculate_vertical_angle(a, b):
    """2点間の垂直線に対する角度（体幹前傾などに使用）"""
    if a is None or b is None: return 0
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    # 垂直(0度)からの傾き。前傾プラス
    angle = math.degrees(math.atan2(dx, dy)) 
    return angle # 単純な傾き

def process_video(uploaded_file, view_type):
    """動画処理共通関数"""
    if uploaded_file is None: return None, None, None

    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    data = []
    
    # 描画設定（スッキリ版）
    # 頭部(0), 肩(11,12), 腰(23,24), 膝(25,26), 足首(27,28), かかと(29,30), つま先(31,32)
    # 耳(7,8)を追加して頭の傾きを見る
    KEYPOINTS = [0, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    CONNECTIONS = [
        (11, 12), (23, 24), (11, 23), (12, 24), # 体幹
        (23, 25), (24, 26), (25, 27), (26, 28), # 下肢
        (27, 31), (28, 32), (29, 31), (30, 32)  # 足部
    ]

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            frame_data = {"frame": frame_idx}

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h_img, w_img, _ = image.shape
                
                def get_c(idx): return [landmarks[idx].x, landmarks[idx].y]
                def get_pix(idx): return int(landmarks[idx].x * w_img), int(landmarks[idx].y * h_img)
                
                # --- 解析ロジック ---
                # 1. 共通: 膝角度
                l_knee = calculate_angle(get_c(23), get_c(25), get_c(27))
                r_knee = calculate_angle(get_c(24), get_c(26), get_c(28))
                frame_data.update({"l_knee": l_knee, "r_knee": r_knee})

                # 2. Viewごとの特化解析
                if view_type == "front":
                    # A. 頭の左右の傾き (耳の高さの差)
                    l_ear = get_c(7); r_ear = get_c(8)
                    head_tilt = (r_ear[1] - l_ear[1]) * 100 # +は右下がり(画面向かって左)
                    
                    # B. 肩の下がり
                    l_sh = get_c(11); r_sh = get_c(12)
                    shoulder_tilt = (r_sh[1] - l_sh[1]) * 100 # +は右下がり
                    
                    # C. 骨盤スウェイ (正中線からの逸脱)
                    mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2
                    # 簡易的に画面中央(0.5)または首(mid_shoulder)との差を見る
                    mid_sh_x = (landmarks[11].x + landmarks[12].x) / 2
                    sway = (mid_hip_x - mid_sh_x) * 100 # +は骨盤が右へスウェイ
                    
                    frame_data.update({
                        "head_tilt": head_tilt, 
                        "shoulder_tilt": shoulder_tilt, 
                        "sway": sway
                    })

                elif view_type == "side":
                    # A. 体幹前傾 (耳-肩-腰 のラインを見るのが正確だが、簡易的に肩-腰の垂直傾き)
                    # 側面の場合、手前側の肩と腰を使う（通常左側通行で撮影なら左、など。ここでは平均をとる）
                    mid_sh = [(landmarks[11].x+landmarks[12].x)/2, (landmarks[11].y+landmarks[12].y)/2]
                    mid_hip = [(landmarks[23].x+landmarks[24].x)/2, (landmarks[23].y+landmarks[24].y)/2]
                    trunk_angle = calculate_vertical_angle(mid_sh, mid_hip) # +は前傾ではない(座標系による)
                    # 補正: 垂直0度に対して、頭が前にあるか
                    trunk_lean = (mid_sh[0] - mid_hip[0]) * 100 # 単純なX差分
                    
                    # B. 歩幅 (両足首のX距離)
                    step_len = abs(landmarks[27].x - landmarks[28].x) * 100
                    
                    # C. 接地角度 (足首のY座標が一番低い時のつま先の上がり具合...は難しいので)
                    # つま先と踵の高さ関係を見る (背屈チェック)
                    l_toe_lift = landmarks[31].y - landmarks[29].y # -ならつま先が上(背屈)
                    
                    frame_data.update({
                        "trunk_lean": trunk_lean,
                        "step_len": step_len
                    })

                data.append(frame_data)

                # --- 描画 (スッキリ版) ---
                for start, end in CONNECTIONS:
                    cv2.line(image, get_pix(start), get_pix(end), (200, 200, 200), 2)
                
                # 左右色分け点
                for idx in KEYPOINTS:
                    color = (0, 0, 255) if idx % 2 == 0 else (255, 0, 0) # 右:赤, 左:青
                    if idx in [0, 7, 8]: color = (0, 255, 255) # 頭は黄色
                    cv2.circle(image, get_pix(idx), 5, color, -1)

            out.write(image)
            frame_idx += 1
            
    cap.release()
    out.release()
    return output_path, pd.DataFrame(data), fps

# --- メインレイアウト ---
tab1, tab2 = st.tabs(["🎥 動画解析 & 統合レポート", "📊 機能データ詳細"])

with tab1:
    col_front, col_side = st.columns(2)
    
    # --- 1. 正面動画 ---
    with col_front:
        st.subheader("① 正面 (Front View)")
        st.markdown("チェック: 頭の傾き、肩の下がり、骨盤スウェイ")
        file_front = st.file_uploader("正面動画をアップロード", type=['mp4', 'mov'], key="front")
        
    # --- 2. 側面動画 ---
    with col_side:
        st.subheader("② 側面 (Side View)")
        st.markdown("チェック: 体幹前傾、歩幅、接地・蹴り出し")
        file_side = st.file_uploader("側面動画をアップロード", type=['mp4', 'mov'], key="side")

    # 解析実行ボタン
    if file_front and file_side:
        if st.button("🚀 統合分析を実行"):
            with st.spinner("両方の動画を解析し、機能データと照合中..."):
                # 解析処理
                path_f, df_f, _ = process_video(file_front, "front")
                path_s, df_s, _ = process_video(file_side, "side")
                
                # --- 結果表示エリア ---
                st.markdown("---")
                st.header("🛡️ 統合分析レポート")
                
                # 動画表示
                c1, c2 = st.columns(2)
                with c1:
                    st.video(path_f)
                    st.caption("正面解析結果")
                    # 正面のグラフ
                    fig_f, ax_f = plt.subplots(figsize=(4, 2))
                    ax_f.plot(df_f['sway'], label='骨盤動揺(Sway)', color='purple')
                    ax_f.axhline(0, color='gray', linestyle='--')
                    ax_f.set_title("骨盤の左右動揺 (+:右へ変位)")
                    ax_f.legend(fontsize='small')
                    st.pyplot(fig_f)
                    
                with c2:
                    st.video(path_s)
                    st.caption("側面解析結果")
                    # 側面のグラフ
                    fig_s, ax_s = plt.subplots(figsize=(4, 2))
                    ax_s.plot(df_s['step_len'], label='歩幅目安', color='green')
                    ax_s.set_title("歩幅の変化 (ピークが高いほど大股)")
                    ax_s.legend(fontsize='small')
                    st.pyplot(fig_s)

                # --- 🧠 推論ロジック (PT Brain) ---
                st.subheader("👨‍⚕️ 原因推論と対策アドバイス")
                
                findings = []
                
                # 1. スウェイ × 筋力
                max_sway = df_f['sway'].abs().max()
                if max_sway > 5.0: # 閾値は仮
                    weak_glute = ""
                    if one_leg_l < 10 or one_leg_r < 10:
                        weak_glute = "片脚立位時間の短さからも、中殿筋による骨盤支持性が低下しています。"
                    elif hip_flex_l < 1.0 or hip_flex_r < 1.0:
                        weak_glute = "股関節周りの筋力不足が影響している可能性があります。"
                    
                    findings.append(f"⚠️ **骨盤の横揺れ（Trendelenburg/Duchenne様）**が見られます。{weak_glute}膝への負担が増加するリスクがあります。")

                # 2. 前傾・姿勢 × FRT/FFD
                # 平均的な前傾具合を見る
                avg_lean = df_s['trunk_lean'].mean()
                if abs(avg_lean) > 5.0:
                    posture_note = ""
                    if frt < 25:
                        posture_note = "FRT（動的バランス）も低下しており、転倒リスクが高い状態です。"
                    elif ffd < 0:
                        posture_note = "ハムストリングスや腰部の柔軟性低下が、骨盤後傾や代償動作を招いている可能性があります。"
                    findings.append(f"⚠️ **体幹の前傾（または後傾）**が目立ちます。{posture_note}")

                # 3. 歩幅・推進力 × 足趾把持・ステップ
                max_step = df_s['step_len'].max()
                if max_step < 15.0: # 仮の小股閾値
                    push_off = ""
                    if toe_grip_l < 10 or toe_grip_r < 10:
                        push_off = "「足趾把持力」が弱く、地面を蹴る力が不足しています。"
                    elif seat_step < 40:
                        push_off = "座位ステップ数が少なく、素早い動作や腸腰筋の活動が低下しています。"
                    
                    findings.append(f"⚠️ **歩幅が小さい**です。{push_off}しっかりと踵からつき、親指で蹴る意識が必要です。")

                # 4. 痛みとの統合
                if "膝(右)" in pain_areas or "膝(左)" in pain_areas:
                    findings.append("🚨 **膝の痛み**と解析結果の関連：骨盤のスウェイや足指の蹴り出し不足が、膝への回旋ストレス（ねじれ）を生んでいる可能性が高いです。膝そのものより、股関節・足部のケアが優先です。")
                
                if "腰" in pain_areas:
                    findings.append("🚨 **腰痛**と解析結果の関連：体幹の前後傾が見られる場合、腹圧の低下により腰椎で支えてしまっています。「ドローイン」などの体幹トレーニングを併用してください。")

                # 表示
                if findings:
                    for f in findings:
                        st.info(f)
                else:
                    st.success("✅ 大きな異常動作は見られません。現在の身体機能を維持しましょう！")

    elif not file_front or not file_side:
        st.warning("⚠️ 正面と側面、両方の動画をアップロードしてください。")

with tab2:
    st.markdown("### 📊 入力された機能データ詳細")
    # レーダーチャート作成
    categories = ['握力', '股屈曲', '片脚立位', 'FRT', 'FFD', '足趾把持', 'ステップ']
    # 左右平均などで簡易スコア化 (デモ用正規化)
    values = [
        (grip_l+grip_r)/2 / 30 * 5,
        (hip_flex_l+hip_flex_r)/2 / 1.5 * 5,
        (one_leg_l+one_leg_r)/2 / 60 * 5,
        frt / 40 * 5,
        (ffd + 10) / 20 * 5,
        (toe_grip_l+toe_grip_r)/2 / 15 * 5,
        seat_step / 50 * 5
    ]
    # 5点満点でクリップ
    values = [min(max(v, 0), 5) for v in values]
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig_r, ax_r = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax_r.fill(angles, values, color='blue', alpha=0.25)
    ax_r.plot(angles, values, color='blue', linewidth=2)
    ax_r.set_yticklabels([])
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(categories)
    ax_r.set_title("身体機能評価バランス (推定スコア)")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(fig_r)
    with c2:
        st.markdown(f"""
        **測定値サマリ:**
        - **FRT**: {frt} cm
        - **FFD**: {ffd} cm
        - **座位ステップ**: {seat_step} 回
        - **足趾把持力**: 右 {toe_grip_r}% / 左 {toe_grip_l}%
        
        *※レーダーチャートは入力値から簡易的に算出したスコアです。*
        """)
