import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# Phase 1-4のモジュールをインポート
from gait_event_detector import GaitEventDetector
from gait_parameter_calculator import GaitParameterCalculator
from integrated_gait_analyzer import IntegratedGaitAnalyzer

# --- 1. ページ設定 ---
st.set_page_config(
    page_title="歩行分析システム", 
    page_icon="🚶",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
.big-title {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
}
.risk-box {
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
}
.warning-box {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
}
.error-box {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
}
.info-box {
    background-color: #d1ecf1;
    border-left: 5px solid #17a2b8;
}
</style>
""", unsafe_allow_html=True)

# --- 2. 分析エンジンの定義 ---
@st.cache_resource
def load_pose_model():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5, 
        model_complexity=1,
        static_image_mode=False,
        smooth_landmarks=True
    )

def calculate_angle(a, b, c):
    """3点の座標から角度を算出"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

def get_line_angle(p1, p2):
    """2点間のベクトルの角度"""
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def analyze_fall_risk(cv_value: float):
    """転倒リスクを評価"""
    if cv_value < 3.0:
        return {
            "level": "低リスク",
            "color": "success",
            "message": "歩行の一定性が保たれています。",
            "icon": "✅"
        }
    elif cv_value < 5.0:
        return {
            "level": "やや注意",
            "color": "info", 
            "message": "歩行は比較的安定していますが、定期的なチェックをお勧めします。",
            "icon": "ℹ️"
        }
    elif cv_value < 10.0:
        return {
            "level": "要注意",
            "color": "warning",
            "message": "歩行にばらつきが見られます。バランストレーニングを検討してください。",
            "icon": "⚠️"
        }
    else:
        return {
            "level": "高リスク",
            "color": "error",
            "message": "歩行が不安定です。専門家への相談をお勧めします。",
            "icon": "🚨"
        }

def analyze_spine_risk(phase_diff: float):
    """脊柱リスクを評価"""
    if phase_diff >= 20.0:
        return {
            "level": "低リスク",
            "color": "success",
            "message": "体幹の協調性が良好です。",
            "advice": "しなやかな回旋が保たれています。",
            "icon": "✅"
        }
    else:
        return {
            "level": "要注意",
            "color": "warning",
            "message": "胸郭と骨盤が同調しすぎています（剛性の増加）。",
            "advice": "💡 理学療法士のアドバイス: 体幹の回旋を引き出すストレッチが有効です。",
            "icon": "⚠️"
        }

# --- 3. UI表示 ---
st.markdown('<p class="big-title">🚶 歩行分析システム</p>', unsafe_allow_html=True)
st.info("💡 動画をアップロードして、AIが自動で歩行を分析します")

# カメラアングル選択
st.subheader("📷 カメラアングルを選択")
camera_angle = st.radio(
    "動画の撮影角度",
    ["📸 側面（横から）", "📸 正面（前から）"],
    horizontal=True
)

# 動画アップロード（前のコードと同じシンプルな方式）
st.subheader("📹 動画をアップロード")

if "側面" in camera_angle:
    st.markdown("**分析内容**: 第1歩の股関節屈曲角度、歩行周期")
    uploaded_video = st.file_uploader(
        "側面動画をアップロード",
        type=["mp4", "mov", "MP4", "MOV"],
        key="side_video"
    )
else:
    st.markdown("**分析内容**: 体幹の左右動揺、歩幅の変動性")
    uploaded_video = st.file_uploader(
        "正面動画をアップロード", 
        type=["mp4", "mov", "MP4", "MOV"],
        key="front_video"
    )

# 解析変数の初期化
max_flexion_angle = 0.0
calculated_cv = 0.0
calculated_phase = 0.0

# --- 4. 解析実行 ---
if uploaded_video is not None:
    # ファイル情報表示
    file_size_mb = uploaded_video.size / (1024 * 1024)
    st.success(f"✅ {uploaded_video.name} ({file_size_mb:.1f}MB) アップロード完了")
    
    # 動画プレビュー
    with st.expander("🎬 動画プレビュー"):
        st.video(uploaded_video)
    
    if st.button("✨ アルゴリズム解析を開始", type="primary", use_container_width=True):
        with st.spinner("🔄 解析中... しばらくお待ちください"):
            try:
                # MediaPipe準備
                pose_engine = load_pose_model()
                mp_pose = mp.solutions.pose
                mp_drawing = mp.solutions.drawing_utils
                
                # 一時ファイルに保存
                file_extension = uploaded_video.name.split('.')[-1].lower()
                if file_extension == 'mov':
                    file_extension = 'mp4'
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tfile:
                    tfile.write(uploaded_video.read())
                    temp_path = tfile.name
                
                # 動画を開く
                cap = cv2.VideoCapture(temp_path)
                
                if not cap.isOpened():
                    st.error("❌ 動画を開けませんでした")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_bar = st.progress(0)
                    st.info(f"📊 総フレーム数: {total_frames}")
                    
                    if "側面" in camera_angle:
                        # --- 側面解析 ---
                        best_frame_flex = None
                        frame_count = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose_engine.process(image)
                            
                            if results.pose_landmarks:
                                lm = results.pose_landmarks.landmark
                                s = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                                     lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                                h = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, 
                                     lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                                k = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, 
                                     lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                                
                                current_angle = calculate_angle(s, h, k)
                                flex_val = np.abs(180 - current_angle)
                                
                                if flex_val > max_flexion_angle:
                                    max_flexion_angle = flex_val
                                    best_frame_flex = image.copy()
                                    mp_drawing.draw_landmarks(
                                        best_frame_flex, 
                                        results.pose_landmarks, 
                                        mp_pose.POSE_CONNECTIONS
                                    )
                            
                            frame_count += 1
                            if frame_count % 10 == 0:
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                        
                        cap.release()
                        progress_bar.progress(1.0)
                        
                        # 結果表示
                        st.success("✅ 解析完了！")
                        st.markdown("---")
                        
                        st.subheader("📊 側面分析結果")
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.metric("第1歩：最大股関節屈曲角度", f"{max_flexion_angle:.1f}°")
                            st.markdown("👉 **Sakane(2025)** 基準に基づき、振り出しの強さを評価")
                            
                            if max_flexion_angle < 30:
                                st.warning("⚠️ 股関節の振り出しが小さめです")
                            elif max_flexion_angle >= 50:
                                st.success("✅ 良好な振り出し")
                        
                        with col2:
                            if best_frame_flex is not None:
                                st.image(
                                    best_frame_flex, 
                                    caption="AIが特定した最大屈曲の瞬間",
                                    use_container_width=True
                                )
                    
                    else:
                        # --- 正面解析 ---
                        step_widths, phase_diffs = [], []
                        frame_count = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose_engine.process(image)
                            
                            if results.pose_landmarks:
                                lm = results.pose_landmarks.landmark
                                
                                # 肩と腰の傾き
                                ls = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                                      lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                                rs = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                                      lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                                lh = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, 
                                      lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                                rh = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, 
                                      lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                                
                                phase_diffs.append(
                                    abs(get_line_angle(ls, rs) - get_line_angle(lh, rh))
                                )
                                
                                # ステップ幅
                                step_widths.append(
                                    abs(lm[mp_pose.PoseLandmark.LEFT_HEEL].x - 
                                        lm[mp_pose.PoseLandmark.RIGHT_HEEL].x)
                                )
                            
                            frame_count += 1
                            if frame_count % 10 == 0:
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                        
                        cap.release()
                        progress_bar.progress(1.0)
                        
                        # CV値計算
                        if step_widths and np.mean(step_widths) != 0:
                            calculated_cv = (np.std(step_widths) / np.mean(step_widths)) * 100
                        
                        calculated_phase = np.mean(phase_diffs) if phase_diffs else 0
                        
                        # 結果表示
                        st.success("✅ 解析完了！")
                        st.markdown("---")
                        
                        st.subheader("📊 正面分析結果")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "歩幅CV値（変動性）", 
                                f"{calculated_cv:.1f}%",
                                delta=f"{calculated_cv-21.7:.1f}% vs 閾値",
                                delta_color="inverse"
                            )
                        with col2:
                            st.metric(
                                "脊柱協調性(位相差)", 
                                f"{calculated_phase:.1f}°",
                                delta=f"{calculated_phase-20:.1f}° vs 閾値"
                            )
                        
                        # 総合判定
                        st.markdown("---")
                        st.header("📋 総合リスク判定")
                        
                        r1, r2 = st.columns(2)
                        
                        with r1:
                            st.subheader("🚨 転倒リスク評価")
                            fall_risk = analyze_fall_risk(calculated_cv)
                            
                            risk_class = f"{fall_risk['color']}-box"
                            st.markdown(f"""
                            <div class="risk-box {risk_class}">
                                <h3>{fall_risk['icon']} 【{fall_risk['level']}】</h3>
                                <p>{fall_risk['message']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with r2:
                            st.subheader("🦴 脊柱・腰痛リスク評価")
                            spine_risk = analyze_spine_risk(calculated_phase)
                            
                            risk_class = f"{spine_risk['color']}-box"
                            st.markdown(f"""
                            <div class="risk-box {risk_class}">
                                <h3>{spine_risk['icon']} 【{spine_risk['level']}】</h3>
                                <p>{spine_risk['message']}</p>
                                <p><strong>{spine_risk['advice']}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # 一時ファイル削除
                os.unlink(temp_path)
            
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")
                import traceback
                with st.expander("詳細なエラー情報"):
                    st.code(traceback.format_exc())

# アルゴリズムの根拠
with st.expander("📚 アルゴリズムの根拠（PT用）"):
    st.markdown("""
    ### 転倒リスク評価の根拠
    - **変動係数 (CV) < 3%**: 正常な歩行パターン
    - **CV 3-5%**: 軽度の不安定性
    - **CV 5-10%**: 中等度の不安定性
    - **CV > 10%**: 高度な不安定性（要介入）
    
    **参考文献**: 
    - Sakane (2025): 第1歩の股関節屈曲と転倒リスク
    - Park (2025): ステップ幅CV値のカットオフ 21.7%
    
    ### 脊柱リスク評価の根拠
    - 相対位相差 **20度未満** を剛性増加の指標とする
    - 体幹の協調性と腰痛の関連
    
    **参考文献**:
    - Lamoth et al. (2002): Pelvis-thorax coordination
    - Smith/Xu: 体幹の同調と腰痛リスク
    """)

if __name__ == "__main__":
    pass
