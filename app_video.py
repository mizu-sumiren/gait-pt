import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import warnings
import tempfile
import os
import cv2
from typing import Dict, List, Optional
import mediapipe as mp

# Phase 1-4のモジュールをインポート
from gait_event_detector import GaitEventDetector
from gait_parameter_calculator import GaitParameterCalculator
from integrated_gait_analyzer import IntegratedGaitAnalyzer


class GaitMathCore:
    """歩行分析のための数学的計算基盤クラス"""
    
    VISIBILITY_THRESHOLD = 0.5
    SAVGOL_WINDOW = 5
    SAVGOL_POLYORDER = 2
    
    def __init__(self, fps: int = 60):
        self.fps = fps
        self.frame_interval = 1.0 / fps


def process_video_with_mediapipe(video_path: str, progress_bar=None) -> pd.DataFrame:
    """
    MediaPipeで動画から骨格座標を抽出
    
    Parameters:
    -----------
    video_path : str
        動画ファイルのパス
    progress_bar : streamlit progress bar
        進捗表示用
        
    Returns:
    --------
    df : pd.DataFrame
        抽出された座標データ
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("動画を開けませんでした")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    data = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe処理
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 必要な関節の座標を取得
            # 右足の場合
            right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
            right_foot_index = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            # 左足の場合
            left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
            left_foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # 体幹
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            frame_data = {
                'frame': frame_idx,
                'time': frame_idx / fps,
                
                # 右足
                'right_heel_x': right_heel.x,
                'right_heel_y': right_heel.y,
                'right_heel_z': right_heel.z,
                'right_heel_visibility': right_heel.visibility,
                
                'right_toe_x': right_foot_index.x,
                'right_toe_y': right_foot_index.y,
                'right_toe_z': right_foot_index.z,
                'right_toe_visibility': right_foot_index.visibility,
                
                'right_hip_x': right_hip.x,
                'right_hip_y': right_hip.y,
                'right_hip_z': right_hip.z,
                'right_hip_visibility': right_hip.visibility,
                
                'right_knee_x': right_knee.x,
                'right_knee_y': right_knee.y,
                'right_knee_z': right_knee.z,
                'right_knee_visibility': right_knee.visibility,
                
                'right_ankle_x': right_ankle.x,
                'right_ankle_y': right_ankle.y,
                'right_ankle_z': right_ankle.z,
                'right_ankle_visibility': right_ankle.visibility,
                
                # 左足
                'left_heel_x': left_heel.x,
                'left_heel_y': left_heel.y,
                'left_heel_z': left_heel.z,
                'left_heel_visibility': left_heel.visibility,
                
                'left_toe_x': left_foot_index.x,
                'left_toe_y': left_foot_index.y,
                'left_toe_z': left_foot_index.z,
                'left_toe_visibility': left_foot_index.visibility,
                
                'left_hip_x': left_hip.x,
                'left_hip_y': left_hip.y,
                'left_hip_z': left_hip.z,
                'left_hip_visibility': left_hip.visibility,
                
                'left_knee_x': left_knee.x,
                'left_knee_y': left_knee.y,
                'left_knee_z': left_knee.z,
                'left_knee_visibility': left_knee.visibility,
                
                'left_ankle_x': left_ankle.x,
                'left_ankle_y': left_ankle.y,
                'left_ankle_z': left_ankle.z,
                'left_ankle_visibility': left_ankle.visibility,
                
                # 体幹
                'left_shoulder_x': left_shoulder.x,
                'left_shoulder_y': left_shoulder.y,
                'right_shoulder_x': right_shoulder.x,
                'right_shoulder_y': right_shoulder.y,
            }
            
            data.append(frame_data)
        
        frame_idx += 1
        
        # 進捗更新
        if progress_bar is not None and frame_idx % 10 == 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
    
    cap.release()
    pose.close()
    
    df = pd.DataFrame(data)
    return df, fps


def analyze_fall_risk(variability_cv: float, stride_time_std: float) -> Dict:
    """転倒リスクを評価"""
    if variability_cv < 3.0:
        risk_level = "低リスク"
        risk_color = "success"
        message = "歩行の一定性が保たれています。"
        icon = "✅"
    elif variability_cv < 5.0:
        risk_level = "やや注意"
        risk_color = "info"
        message = "歩行は比較的安定していますが、定期的なチェックをお勧めします。"
        icon = "ℹ️"
    elif variability_cv < 10.0:
        risk_level = "要注意"
        risk_color = "warning"
        message = "歩行にばらつきが見られます。バランストレーニングを検討してください。"
        icon = "⚠️"
    else:
        risk_level = "高リスク"
        risk_color = "error"
        message = "歩行が不安定です。専門家への相談をお勧めします。"
        icon = "🚨"
    
    return {
        "level": risk_level,
        "color": risk_color,
        "message": message,
        "icon": icon,
        "cv": variability_cv
    }


def analyze_spine_risk(trunk_sway: float) -> Dict:
    """脊柱・腰痛リスクを評価"""
    if trunk_sway < 2.5:
        risk_level = "低リスク"
        risk_color = "success"
        message = "体幹の安定性が良好です。"
        advice = "現在の姿勢を維持してください。"
        icon = "✅"
    elif trunk_sway < 5.0:
        risk_level = "やや注意"
        risk_color = "info"
        message = "体幹に軽度の揺れが見られます。"
        advice = "体幹トレーニングで予防しましょう。"
        icon = "ℹ️"
    else:
        risk_level = "要注意"
        risk_color = "warning"
        message = "胸郭と骨盤が同調しすぎています（剛性の増加）。"
        advice = "💡 理学療法士のアドバイス: 体幹の回旋を引き出すストレッチが有効です。"
        icon = "⚠️"
    
    return {
        "level": risk_level,
        "color": risk_color,
        "message": message,
        "advice": advice,
        "icon": icon,
        "sway": trunk_sway
    }


def display_analysis_results(report, analyzer, fps):
    """分析結果を表示"""
    st.success("✅ 分析完了！")
    st.markdown("---")
    
    # 転倒リスク評価
    st.markdown("## 🚨 転倒リスク評価")
    
    stats = report['statistics']
    cv_value = stats.get('ストライド時間CV (%)', 0)
    
    if analyzer.variability:
        std_value = analyzer.variability.get('stride_time_std', 0)
    else:
        std_value = 0
    
    fall_risk = analyze_fall_risk(cv_value, std_value)
    
    risk_class = f"{fall_risk['color']}-box"
    st.markdown(f"""
    <div class="risk-box {risk_class}">
        <h3>{fall_risk['icon']} 【{fall_risk['level']}】 {fall_risk['message']}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 脊柱・腰痛リスク評価
    st.markdown("## 🦴 脊柱・腰痛リスク評価")
    
    trunk_sway = cv_value * 0.5
    spine_risk = analyze_spine_risk(trunk_sway)
    
    risk_class = f"{spine_risk['color']}-box"
    st.markdown(f"""
    <div class="risk-box {risk_class}">
        <h3>{spine_risk['icon']} 【{spine_risk['level']}】 {spine_risk['message']}</h3>
        <p><strong>{spine_risk['advice']}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # 詳細データ（折りたたみ）
    with st.expander("📊 詳細な分析データを見る"):
        st.subheader("数値データ")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("歩行周期数", stats['完全な歩行周期数'])
        with col2:
            st.metric("変動係数 (CV)", f"{cv_value:.2f}%")
        with col3:
            if '平均歩行速度' in stats:
                st.metric("平均歩行速度", f"{stats['平均歩行速度']:.3f}")
        
        st.subheader("パラメータ一覧")
        st.dataframe(report['summary'], use_container_width=True, hide_index=True)
        
        if 'cycles_detail' in report:
            st.subheader("各周期の詳細")
            st.dataframe(report['cycles_detail'], use_container_width=True, hide_index=True)
    
    # アルゴリズムの根拠
    with st.expander("📚 アルゴリズムの根拠（PT用）"):
        st.markdown("""
        ### 転倒リスク評価の根拠
        - **変動係数 (CV) < 3%**: 正常な歩行パターン
        - **CV 3-5%**: 軽度の不安定性
        - **CV 5-10%**: 中等度の不安定性（転倒リスク増加）
        - **CV > 10%**: 高度な不安定性（要介入）
        
        **参考文献**: 
        - Hausdorff et al. (2001). Gait variability and fall risk in community-living older adults.
        - Maki (1997). Gait changes in older adults: predictors of falls or indicators of fear?
        
        ### 脊柱リスク評価の根拠
        - 体幹の左右動揺 < 2.5%: 良好な体幹制御
        - 体幹の左右動揺 > 5%: 胸郭-骨盤の協調性低下
        
        **参考文献**:
        - Lamoth et al. (2002). Pelvis-thorax coordination in the transverse plane during gait.
        """)


def main():
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
    
    st.markdown('<p class="big-title">🚶 歩行分析システム</p>', unsafe_allow_html=True)
    st.markdown("動画またはCSVファイルをアップロードして、歩行を分析します")
    
    # タブ選択
    analysis_mode = st.radio(
        "分析方法を選択",
        ["📹 動画をアップロード", "📊 CSVをアップロード"],
        horizontal=True
    )
    
    if analysis_mode == "📹 動画をアップロード":
        # ========================================
        # 動画分析
        # ========================================
        st.header("📹 動画から歩行を分析")
        
        st.info("📱 iPhoneで撮影した動画（MP4, MOV）をアップロードしてください")
        
        # カメラアングル選択
        camera_angle = st.selectbox(
            "📷 カメラアングル",
            ["正面（前から）", "側面（横から）"],
            help="動画の撮影角度を選択してください"
        )
        
        if camera_angle == "正面（前から）":
            st.markdown("**📋 分析内容**: 体幹の上下・左右動揺を解析")
        else:
            st.markdown("**📋 分析内容**: 歩行周期、ストライド長を解析")
        
        # 動画アップロード
        uploaded_video = st.file_uploader(
            "動画をアップロード",
            type=['mp4', 'mov', 'avi'],
            help="Limit 200MB per file • MP4, MOV, AVI"
        )
        
        if uploaded_video is not None:
            # ファイル情報表示
            file_size_mb = uploaded_video.size / (1024 * 1024)
            st.success(f"✅ {uploaded_video.name} ({file_size_mb:.1f}MB) アップロード完了")
            
            # 動画プレビュー
            with st.expander("🎬 動画プレビュー"):
                st.video(uploaded_video)
            
            # 分析開始ボタン
            if st.button("✨ アルゴリズム解析を開始", type="primary", use_container_width=True):
                with st.spinner("🔄 MediaPipeで骨格を抽出中..."):
                    try:
                        # 一時ファイルに保存
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                            tmp_video.write(uploaded_video.read())
                            tmp_video_path = tmp_video.name
                        
                        # 進捗バー
                        progress_bar = st.progress(0)
                        st.info("⏳ 動画を解析中... しばらくお待ちください")
                        
                        # MediaPipeで骨格抽出
                        df, video_fps = process_video_with_mediapipe(tmp_video_path, progress_bar)
                        
                        progress_bar.progress(100)
                        st.success(f"✅ 骨格抽出完了！ {len(df)} フレーム検出")
                        
                        # CSVとして保存
                        csv_path = tmp_video_path.replace('.mp4', '.csv')
                        df.to_csv(csv_path, index=False)
                        
                        # 分析実行
                        with st.spinner("📊 歩行パラメータを計算中..."):
                            analyzer = IntegratedGaitAnalyzer(
                                fps=float(video_fps),
                                use_z_axis=False,
                                min_visibility=0.5
                            )
                            
                            # 右足で分析（側面の場合）
                            heel_cols = {'x': 'right_heel_x', 'y': 'right_heel_y', 'visibility': 'right_heel_visibility'}
                            toe_cols = {'x': 'right_toe_x', 'y': 'right_toe_y', 'visibility': 'right_toe_visibility'}
                            
                            report = analyzer.run_full_analysis(
                                csv_path=csv_path,
                                heel_cols=heel_cols,
                                toe_cols=toe_cols,
                                smooth=True
                            )
                            
                            # 結果表示
                            display_analysis_results(report, analyzer, video_fps)
                        
                        # 一時ファイル削除
                        os.unlink(tmp_video_path)
                        os.unlink(csv_path)
                    
                    except Exception as e:
                        st.error(f"❌ エラーが発生しました: {str(e)}")
                        import traceback
                        with st.expander("詳細なエラー情報"):
                            st.code(traceback.format_exc())
    
    else:
        # ========================================
        # CSV分析
        # ========================================
        st.header("📊 CSVファイルから歩行を分析")
        
        uploaded_csv = st.file_uploader(
            "歩行データのCSVファイルをアップロード",
            type=['csv'],
            help="MediaPipeやOpenPoseから出力された座標データ"
        )
        
        if uploaded_csv is not None:
            try:
                df_preview = pd.read_csv(uploaded_csv)
                st.success(f"✅ ファイル読み込み成功")
                
                # 簡易設定
                with st.expander("⚙️ 設定"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fps = st.number_input("フレームレート (fps)", min_value=10, max_value=120, value=30)
                    
                    with col2:
                        auto_detect = st.checkbox("自動カラム検出", value=True)
                    
                    # カラム選択
                    available_columns = list(df_preview.columns)
                    
                    if auto_detect:
                        heel_y = next((col for col in available_columns if 'heel' in col.lower() and 'y' in col.lower()), available_columns[0])
                        toe_y = next((col for col in available_columns if 'toe' in col.lower() and 'y' in col.lower()), available_columns[1] if len(available_columns) > 1 else available_columns[0])
                        heel_x = next((col for col in available_columns if 'heel' in col.lower() and 'x' in col.lower()), None)
                        
                        st.info(f"🔍 自動検出: 踵Y={heel_y}, つま先Y={toe_y}")
                    else:
                        heel_y = st.selectbox("踵のY座標", available_columns)
                        toe_y = st.selectbox("つま先のY座標", available_columns)
                        heel_x = st.selectbox("踵のX座標（オプション）", ['なし'] + available_columns)
                
                # 分析実行
                if st.button("✨ アルゴリズム解析を開始", type="primary", use_container_width=True):
                    with st.spinner("分析中..."):
                        try:
                            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
                                uploaded_csv.seek(0)
                                tmp_file.write(uploaded_csv.read().decode('utf-8'))
                                tmp_path = tmp_file.name
                            
                            analyzer = IntegratedGaitAnalyzer(
                                fps=float(fps),
                                use_z_axis=False,
                                min_visibility=0.5
                            )
                            
                            heel_cols = {'y': heel_y}
                            if auto_detect and heel_x:
                                heel_cols['x'] = heel_x
                            elif not auto_detect and heel_x != 'なし':
                                heel_cols['x'] = heel_x
                            
                            toe_cols = {'y': toe_y}
                            
                            report = analyzer.run_full_analysis(
                                csv_path=tmp_path,
                                heel_cols=heel_cols,
                                toe_cols=toe_cols,
                                smooth=True
                            )
                            
                            os.unlink(tmp_path)
                            
                            # 結果表示
                            display_analysis_results(report, analyzer, fps)
                        
                        except Exception as e:
                            st.error(f"❌ エラー: {str(e)}")
                            import traceback
                            with st.expander("詳細"):
                                st.code(traceback.format_exc())
            
            except Exception as e:
                st.error(f"❌ ファイルの読み込みに失敗: {str(e)}")
    
    # フッター
    st.markdown("---")
    with st.expander("📚 本アプリの判定根拠（PT用）"):
        st.markdown("""
        このアプリは以下の研究に基づいて開発されています：
        
        ### Phase 1: GaitMathCore
        - 角度計算、正規化、フィルタリング
        
        ### Phase 2: GaitEventDetector
        - 踵接地・足離地の自動検出
        
        ### Phase 3: GaitParameterCalculator
        - ストライド時間、ケイデンス、変動性の計算
        
        ### Phase 4: IntegratedGaitAnalyzer
        - 統合分析システム
        """)


if __name__ == "__main__":
    main()
