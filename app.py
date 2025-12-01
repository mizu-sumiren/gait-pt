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
st.set_page_config(page_title="総合歩行・身体機能分析AI (Pro)", page_icon="🛡️", layout="wide")

st.title("🛡️ 総合歩行・身体機能分析AI (Pro)")
st.markdown("歩行の「左右差」と「機能不全」を徹底的に分析します。")

# --- サイドバー：詳細な機能チェック ---
st.sidebar.header("📋 身体機能・測定データ")

with st.sidebar.expander("1. 問診・痛み", expanded=True):
    pain_areas = st.multiselect(
        "痛み・違和感のある部位",
        ["特になし", "首", "肩", "腰", "股関節(右)", "股関節(左)", "膝(右)", "膝(左)", "足首・足部"]
    )
    history = st.text_area("既往歴・特記事項")

with st.sidebar.expander("2. 機能測定結果", expanded=True):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        grip_l = st.number_input("握力(左) kg", value=20.0)
        hip_flex_l = st.number_input("股屈曲(左) kgf/kg", value=0.8)
        one_leg_l = st.number_input("片脚立位(左) 秒", value=10)
        toe_grip_l = st.number_input("足趾把持(左) %", value=8.0)
    with col_s2:
        grip_r = st.number_input("握力(右) kg", value=29.0)
        hip_flex_r = st.number_input("股屈曲(右) kgf/kg", value=1.36)
        one_leg_r = st.number_input("片脚立位(右) 秒", value=120)
        toe_grip_r = st.number_input("足趾把持(右) %", value=11.0)

    frt = st.number_input("FRT (cm)", value=20.0)
    ffd = st.number_input("FFD (cm)", value=-5.0)
    seat_step = st.number_input("座位ステップ (回/20秒)", value=30)

# --- 解析用関数群 ---
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def draw_grid(image, interval=50):
    """姿勢評価用のグリッド線を描画"""
    h, w, _ = image.shape
    color = (200, 200, 200) 
    center_x = w // 2
    cv2.line(image, (center_x, 0), (center_x, h), (0, 255, 255), 1) 
    for x in range(0, w, interval):
        if x != center_x:
            cv2.line(image, (x, 0), (x, h), color, 1)
    for y in range(0, h, interval):
        cv2.line(image, (0, y), (w, y), color, 1)
    return image

def process_video(uploaded_file, view_type):
    if uploaded_file is None: return None, pd.DataFrame() # 空DFを返す
    
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
    
    KEYPOINTS = [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32]
    CONNECTIONS = [
        (11, 12), (23, 24), (11, 23), (12, 24), 
        (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 31), (28, 32)
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
            
            image = draw_grid(image, interval=width//10)

            # 【修正点】初期値をNaN（欠損値）にしておくことでKeyErrorを防ぐ
            frame_data = {
                "frame": frame_idx,
                "l_knee": np.nan, "r_knee": np.nan,
                "shoulder_tilt": np.nan, "hip_tilt": np.nan,
                "sway": np.nan, "step_len": np.nan
            }
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h_img, w_img, _ = image.shape
                
                def get_c(idx): return [landmarks[idx].x, landmarks[idx].y]
                def get_pix(idx): return int(landmarks[idx].x * w_img), int(landmarks[idx].y * h_img)
                
                # 計算処理
                l_knee = calculate_angle(get_c(23), get_c(25), get_c(27))
                r_knee = calculate_angle(get_c(24), get_c(26), get_c(28))
                shoulder_tilt = (landmarks[12].y - landmarks[11].y) * 100 
                hip_tilt = (landmarks[24].y - landmarks[23].y) * 100
                mid_sh_x = (landmarks[11].x + landmarks[12].x) / 2
                mid_hip_x = (landmarks[23].x + landmarks[24].x) / 2
                sway = (mid_hip_x - mid_sh_x) * 100
                step_len = abs(landmarks[27].x - landmarks[28].x) * 100
                
                # 値を上書き
                frame_data.update({
                    "l_knee": l_knee, "r_knee": r_knee,
                    "shoulder_tilt": shoulder_tilt, "hip_tilt": hip_tilt,
                    "sway": sway, "step_len": step_len
                })

                # 描画
                for start, end in CONNECTIONS:
                    cv2.line(image, get_pix(start), get_pix(end), (255, 255, 255), 2)
                for idx in KEYPOINTS:
                    color = (0, 0, 255) if idx % 2 == 0 else (255, 0, 0)
                    cv2.circle(image, get_pix(idx), 6, color, -1)
            
            data.append(frame_data)
            out.write(image)
            frame_idx += 1
            
    cap.release()
    out.release()
    return output_path, pd.DataFrame(data)

# --- メインレイアウト ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("① 正面動画 (Front)")
    file_front = st.file_uploader("正面から撮影", type=['mp4', 'mov'], key="f")
with col2:
    st.subheader("② 側面動画 (Side)")
    file_side = st.file_uploader("横から撮影", type=['mp4', 'mov'], key="s")

if file_front and file_side and st.button("🚀 解析開始"):
    with st.spinner("AIが動作の左右差とリスクを計算中..."):
        path_f, df_f = process_video(file_front, "front")
        path_s, df_s = process_video(file_side, "side")
        
        # エラーハンドリング: 解析失敗（人が映っていない等）の場合
        if df_f.empty or df_s.empty or df_f['sway'].isna().all():
            st.error("⚠️ 解析エラー: 動画から人物の骨格を検出できませんでした。全身が映っている動画を使用してください。")
        else:
            st.markdown("---")
            
            # 1. 動画と波形
            c1, c2 = st.columns(2)
            with c1:
                st.video(path_f)
                st.caption("正面：グリッドで左右のブレを確認")
                fig, ax = plt.subplots(figsize=(5, 2))
                ax.plot(df_f['sway'], color='purple', label='骨盤の横揺れ')
                ax.axhline(0, color='gray', linestyle='--')
                ax.set_title("体幹に対する骨盤の左右動揺")
                ax.legend()
                st.pyplot(fig)
                
            with c2:
                st.video(path_s)
                st.caption("側面：歩幅と姿勢を確認")
                fig2, ax2 = plt.subplots(figsize=(5, 2))
                ax2.plot(df_s['l_knee'], color='blue', label='左膝', alpha=0.7)
                ax2.plot(df_s['r_knee'], color='red', label='右膝', alpha=0.7)
                ax2.set_title("膝関節の屈曲角度 (左右差チェック)")
                ax2.legend()
                st.pyplot(fig2)

            # 2. リスク分析
            st.header("👨‍⚕️ 動作分析レポート")
            alerts = []
            
            # A. 膝の左右差
            max_l = df_s['l_knee'].max()
            max_r = df_s['r_knee'].max()
            diff_knee = abs(max_l - max_r)
            
            if diff_knee > 10:
                weak_side = "左" if max_l < max_r else "右"
                alerts.append(f"🚨 **膝の動きに大きな左右差あり (差: {diff_knee:.1f}度)**\n{weak_side}側の膝の曲がりが浅いです。痛みを避けているか、可動域制限の可能性があります。")
            
            # B. スウェイ
            sway_range = df_f['sway'].max() - df_f['sway'].min()
            if sway_range > 10: 
                reason = "中殿筋の筋力低下" if (one_leg_l < 20 or one_leg_r < 20) else "体幹機能の不安定さ"
                alerts.append(f"🚨 **歩行時の骨盤動揺（ふらつき）が大きい**です。\n{reason}が疑われます。（片脚立位: L{one_leg_l}秒 / R{one_leg_r}秒）")
            
            # C. 推進力
            step_avg = df_s['step_len'].mean()
            if step_avg < 15:
                alerts.append("⚠️ **歩幅が全体的に小さい**です。\n足趾把持力低下や、股関節の伸展制限（蹴り出し不足）が考えられます。")
            
            # D. 機能データ乖離
            if toe_grip_l < 10 or toe_grip_r < 10:
                alerts.append("⚠️ **足趾把持力が低下**しています（基準値未満）。\nこれが「蹴り出し不足」や「ふらつき」の根本原因の可能性があります。")

            if frt < 25:
                alerts.append("⚠️ **FRT(25cm未満)**：動的バランス能力が低下しており、転倒リスクが高い状態です。")

            if alerts:
                for a in alerts:
                    st.error(a)
            else:
                st.success("動作バランスは比較的良好です。引き続き左右差に注意してモニタリングしましょう。")

            st.info("※ この解析はスクリーニングです。確定診断は専門機関での評価が必要です。")
