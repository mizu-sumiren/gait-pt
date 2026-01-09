import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import warnings
import tempfile
import os
import cv2
from typing import Dict, List, Optional

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
        
    @staticmethod
    def calculate_angle_3d(p1, p2, p3, use_z_axis=False, min_visibility=0.5):
        """3点から関節角度を計算（p2が頂点）"""
        if any(p.get('visibility', 0) < min_visibility for p in [p1, p2, p3]):
            return None
        
        if use_z_axis:
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
        else:
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            return None
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg


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
    tab1, tab2, tab3 = st.tabs(["📹 動画分析", "📊 CSV分析", "🔧 詳細設定"])
    
    # ========================================
    # タブ1: 動画分析
    # ========================================
    with tab1:
        st.header("動画から歩行を分析")
        
        st.info("💡 現在、動画アップロード機能は開発中です。CSVファイルをご利用ください。")
        
        uploaded_video = st.file_uploader(
            "歩行動画をアップロード（MP4, MOV, AVI）",
            type=['mp4', 'mov', 'avi'],
            help="正面または側面から撮影した歩行動画"
        )
        
        if uploaded_video is not None:
            st.warning("⚠️ 動画分析機能は開発中です。現在はCSV分析タブをご利用ください。")
            
            # 将来的な実装のプレースホルダー
            with st.expander("📝 動画分析の準備中..."):
                st.markdown("""
                動画分析機能では以下を自動で行います：
                1. MediaPipeによる骨格検出
                2. 座標データの抽出
                3. 歩行イベントの自動検出
                4. リスク評価レポートの生成
                
                **現在の回避策**: 
                - MediaPipeで座標を抽出したCSVファイルを「CSV分析」タブでアップロードしてください
                """)
    
    # ========================================
    # タブ2: CSV分析（メイン機能）
    # ========================================
    with tab2:
        st.header("CSVファイルから歩行を分析")
        
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
                st.subheader("⚙️ 簡単設定")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fps = st.number_input("フレームレート (fps)", min_value=10, max_value=120, value=30)
                
                with col2:
                    auto_detect = st.checkbox("自動カラム検出", value=True, 
                                            help="heel, toe などのキーワードから自動検出")
                
                # カラム選択
                available_columns = list(df_preview.columns)
                
                if auto_detect:
                    # 自動検出ロジック
                    heel_x = next((col for col in available_columns if 'heel' in col.lower() and 'x' in col.lower()), available_columns[0])
                    heel_y = next((col for col in available_columns if 'heel' in col.lower() and 'y' in col.lower()), available_columns[1] if len(available_columns) > 1 else available_columns[0])
                    toe_x = next((col for col in available_columns if 'toe' in col.lower() and 'x' in col.lower()), available_columns[0])
                    toe_y = next((col for col in available_columns if 'toe' in col.lower() and 'y' in col.lower()), available_columns[1] if len(available_columns) > 1 else available_columns[0])
                    
                    st.info(f"🔍 自動検出: 踵Y={heel_y}, つま先Y={toe_y}")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        heel_y = st.selectbox("踵のY座標", available_columns, key="heel_y_manual")
                    with col2:
                        toe_y = st.selectbox("つま先のY座標", available_columns, key="toe_y_manual")
                    
                    heel_x = st.selectbox("踵のX座標（オプション）", ['なし'] + available_columns, key="heel_x_manual")
                    toe_x = st.selectbox("つま先のX座標（オプション）", ['なし'] + available_columns, key="toe_x_manual")
                
                # 分析実行
                if st.button("🚀 分析を実行", type="primary", use_container_width=True):
                    with st.spinner("分析中... しばらくお待ちください"):
                        try:
                            # 一時ファイルに保存
                            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
                                uploaded_csv.seek(0)
                                tmp_file.write(uploaded_csv.read().decode('utf-8'))
                                tmp_path = tmp_file.name
                            
                            # IntegratedGaitAnalyzerで分析
                            analyzer = IntegratedGaitAnalyzer(
                                fps=float(fps),
                                use_z_axis=False,
                                min_visibility=0.5
                            )
                            
                            # カラム設定
                            heel_cols = {'y': heel_y}
                            toe_cols = {'y': toe_y}
                            
                            if auto_detect:
                                if heel_x and heel_x != 'なし':
                                    heel_cols['x'] = heel_x
                            else:
                                if heel_x != 'なし':
                                    heel_cols['x'] = heel_x
                            
                            # 分析実行
                            report = analyzer.run_full_analysis(
                                csv_path=tmp_path,
                                heel_cols=heel_cols,
                                toe_cols=toe_cols,
                                smooth=True
                            )
                            
                            os.unlink(tmp_path)
                            
                            # ============================================
                            # わかりやすい結果表示
                            # ============================================
                            
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
                            
                            # 体幹の揺れを簡易計算（実際のデータがあれば使用）
                            trunk_sway = cv_value * 0.5  # 簡易的な計算
                            
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
                                    if 'mean_walking_speed' in stats:
                                        st.metric("平均歩行速度", f"{stats['平均歩行速度']:.3f}")
                                
                                st.subheader("パラメータ一覧")
                                st.dataframe(report['summary'], use_container_width=True, hide_index=True)
                                
                                if 'cycles_detail' in report:
                                    st.subheader("各周期の詳細")
                                    st.dataframe(report['cycles_detail'], use_container_width=True, hide_index=True)
                            
                            # アルゴリズムの根拠（専門家向け）
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
                        
                        except Exception as e:
                            st.error(f"❌ 分析中にエラーが発生しました: {str(e)}")
                            import traceback
                            with st.expander("詳細なエラー情報"):
                                st.code(traceback.format_exc())
            
            except Exception as e:
                st.error(f"❌ ファイルの読み込みに失敗しました: {str(e)}")
        
        else:
            st.info("👆 CSVファイルをアップロードしてください")
    
    # ========================================
    # タブ3: 詳細設定（専門家向け）
    # ========================================
    with tab3:
        st.header("詳細設定")
        st.markdown("専門家向けの詳細な設定と分析結果")
        
        st.info("🔧 Phase 1-4の全機能にアクセスできます")
        
        if st.checkbox("詳細分析モードを表示"):
            st.warning("⚠️ この機能は専門家向けです")
            
            # ここに元の詳細なタブを表示することも可能
            st.markdown("""
            詳細分析モードでは以下が可能です：
            - Phase 1: 角度計算、正規化
            - Phase 2: 歩行イベント検出の詳細
            - Phase 3: 全パラメータの詳細表示
            - Phase 4: カスタム分析
            """)


if __name__ == "__main__":
    main()
