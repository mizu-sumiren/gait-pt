import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from typing import Dict, List, Tuple, Optional, Union
import warnings
import matplotlib.pyplot as plt

# Phase 2の機能をインポート
from gait_event_detector import GaitEventDetector

# Phase 3の機能をインポート
from gait_parameter_calculator import GaitParameterCalculator

# Phase 4の機能をインポート
from integrated_gait_analyzer import IntegratedGaitAnalyzer

# ========================================
# GaitMathCore クラス（変更なし）
# ========================================

class GaitMathCore:
    """
    歩行分析のための数学的計算基盤クラス
    """
    
    VISIBILITY_THRESHOLD = 0.5
    SAVGOL_WINDOW = 5
    SAVGOL_POLYORDER = 2
    
    def __init__(self, fps: int = 60):
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
    @staticmethod
    def calculate_angle_3d(
        p1: Dict[str, float], 
        p2: Dict[str, float], 
        p3: Dict[str, float],
        use_z_axis: bool = False,
        min_visibility: float = 0.5
    ) -> Optional[float]:
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
    
    @staticmethod
    def calculate_segment_length_3d(
        p1: Dict[str, float],
        p2: Dict[str, float],
        use_z_axis: bool = False,
        min_visibility: float = 0.5
    ) -> Optional[float]:
        """2点間の距離を計算"""
        if any(p.get('visibility', 0) < min_visibility for p in [p1, p2]):
            return None
        
        if use_z_axis:
            distance = np.sqrt(
                (p1['x'] - p2['x'])**2 +
                (p1['y'] - p2['y'])**2 +
                (p1['z'] - p2['z'])**2
            )
        else:
            distance = np.sqrt(
                (p1['x'] - p2['x'])**2 +
                (p1['y'] - p2['y'])**2
            )
        
        return distance
    
    @staticmethod
    def savitzky_golay_filter(
        data: Union[List[float], np.ndarray],
        window_length: int = 5,
        polyorder: int = 2,
        handle_nan: bool = True
    ) -> np.ndarray:
        """Savitzky-Golayフィルタによるノイズ除去"""
        data_array = np.array(data, dtype=float)
        
        if handle_nan and np.any(np.isnan(data_array)):
            valid_idx = ~np.isnan(data_array)
            if np.sum(valid_idx) < 2:
                return data_array
            
            x_valid = np.where(valid_idx)[0]
            y_valid = data_array[valid_idx]
            x_all = np.arange(len(data_array))
            data_array = np.interp(x_all, x_valid, y_valid)
        
        if len(data_array) < window_length:
            return data_array
        
        if window_length % 2 == 0:
            window_length += 1
        
        try:
            filtered = savgol_filter(data_array, window_length, polyorder)
        except Exception:
            return data_array
        
        return filtered
    
    @staticmethod
    def normalize_by_segment_length(
        value: float,
        segment_length: float,
        segment_name: str = "大腿骨長"
    ) -> Optional[float]:
        """身体比率による正規化"""
        if segment_length <= 0 or np.isnan(segment_length):
            return None
        
        normalized = value / segment_length
        return normalized


# ========================================
# Streamlit アプリケーション
# ========================================

def main():
    st.set_page_config(
        page_title="歩行分析エンジン - Phase 1 & 2",
        page_icon="🚶",
        layout="wide"
    )
    
    st.title("🚶 歩行分析エンジン - GaitMathCore + GaitEventDetector")
    st.markdown("---")
    
    # サイドバー
    st.sidebar.header("⚙️ 設定")
    fps = st.sidebar.slider("フレームレート (fps)", 30, 120, 60, 10)
    use_z_axis = st.sidebar.checkbox("Z軸を使用（3D計算）", value=False)
    
    # GaitMathCore 初期化
    math_core = GaitMathCore(fps=fps)
    
    # タブ分け（Phase 2-4のタブを追加）
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📐 角度計算テスト", 
        "📏 セグメント長計算", 
        "🔄 正規化テスト",
        "📊 フィルタリングテスト",
        "🦶 歩行イベント検出（Phase 2）",
        "📈 歩行パラメータ計算（Phase 3）",
        "📂 CSVデータ分析（Phase 4）"
    ])
    
    # ========================================
    # タブ1: 角度計算
    # ========================================
    with tab1:
        st.header("関節角度計算のテスト")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("点1（例: 股関節）")
            p1_x = st.number_input("X座標", value=0.5, key="p1_x")
            p1_y = st.number_input("Y座標", value=0.5, key="p1_y")
            p1_z = st.number_input("Z座標", value=0.0, key="p1_z")
            p1_vis = st.slider("信頼度", 0.0, 1.0, 0.9, key="p1_vis")
        
        with col2:
            st.subheader("点2（例: 膝関節）")
            p2_x = st.number_input("X座標", value=0.5, key="p2_x")
            p2_y = st.number_input("Y座標", value=0.3, key="p2_y")
            p2_z = st.number_input("Z座標", value=0.0, key="p2_z")
            p2_vis = st.slider("信頼度", 0.0, 1.0, 0.9, key="p2_vis")
        
        with col3:
            st.subheader("点3（例: 足関節）")
            p3_x = st.number_input("X座標", value=0.7, key="p3_x")
            p3_y = st.number_input("Y座標", value=0.3, key="p3_y")
            p3_z = st.number_input("Z座標", value=0.0, key="p3_z")
            p3_vis = st.slider("信頼度", 0.0, 1.0, 0.9, key="p3_vis")
        
        if st.button("角度を計算", type="primary", key="calc_angle"):
            p1 = {'x': p1_x, 'y': p1_y, 'z': p1_z, 'visibility': p1_vis}
            p2 = {'x': p2_x, 'y': p2_y, 'z': p2_z, 'visibility': p2_vis}
            p3 = {'x': p3_x, 'y': p3_y, 'z': p3_z, 'visibility': p3_vis}
            
            angle = math_core.calculate_angle_3d(p1, p2, p3, use_z_axis=use_z_axis)
            
            if angle is not None:
                st.success(f"### 計算結果: {angle:.2f}°")
                
                # 角度の評価
                if 170 <= angle <= 180:
                    st.info("✓ 完全伸展位")
                elif 90 <= angle < 170:
                    st.info("✓ 軽度屈曲位")
                elif angle < 90:
                    st.info("✓ 屈曲位")
            else:
                st.error("⚠️ 計算できませんでした（信頼度不足またはゼロベクトル）")
    
    # ========================================
    # タブ2: セグメント長計算
    # ========================================
    with tab2:
        st.header("セグメント長（例: 大腿骨長）の計算")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("始点（例: 股関節）")
            seg_p1_x = st.number_input("X座標", value=0.5, key="seg_p1_x")
            seg_p1_y = st.number_input("Y座標", value=0.5, key="seg_p1_y")
            seg_p1_vis = st.slider("信頼度", 0.0, 1.0, 0.9, key="seg_p1_vis")
        
        with col2:
            st.subheader("終点（例: 膝関節）")
            seg_p2_x = st.number_input("X座標", value=0.5, key="seg_p2_x")
            seg_p2_y = st.number_input("Y座標", value=0.3, key="seg_p2_y")
            seg_p2_vis = st.slider("信頼度", 0.0, 1.0, 0.9, key="seg_p2_vis")
        
        if st.button("セグメント長を計算", type="primary", key="calc_seg"):
            seg_p1 = {'x': seg_p1_x, 'y': seg_p1_y, 'z': 0.0, 'visibility': seg_p1_vis}
            seg_p2 = {'x': seg_p2_x, 'y': seg_p2_y, 'z': 0.0, 'visibility': seg_p2_vis}
            
            length = math_core.calculate_segment_length_3d(seg_p1, seg_p2, use_z_axis=False)
            
            if length is not None:
                st.success(f"### セグメント長: {length:.4f}")
                st.info(f"これを基準単位として正規化に使用します")
            else:
                st.error("⚠️ 計算できませんでした")
    
    # ========================================
    # タブ3: 正規化テスト
    # ========================================
    with tab3:
        st.header("身体比率による正規化")
        
        value_to_normalize = st.number_input(
            "正規化したい値（例: 体幹の上下移動量 [pixel]）",
            value=50.0,
            step=1.0
        )
        
        segment_length = st.number_input(
            "基準セグメント長（例: 大腿骨長 [pixel]）",
            value=200.0,
            step=1.0
        )
        
        if st.button("正規化", type="primary", key="normalize"):
            normalized = math_core.normalize_by_segment_length(
                value_to_normalize, segment_length, "大腿骨長"
            )
            
            if normalized is not None:
                st.success(f"### 正規化値: {normalized:.4f}")
                st.info(f"体幹移動は大腿骨長の {normalized*100:.2f}% に相当")
            else:
                st.error("⚠️ 計算できませんでした")
    
    # ========================================
    # タブ4: フィルタリングテスト
    # ========================================
    with tab4:
        st.header("Savitzky-Golayフィルタのテスト")
        
        # サンプルデータ生成
        n_samples = st.slider("サンプル数", 50, 200, 100)
        noise_level = st.slider("ノイズレベル", 0.0, 0.5, 0.1, 0.01)
        
        # ノイズ付き正弦波
        t = np.linspace(0, 4*np.pi, n_samples)
        clean_signal = np.sin(t)
        noisy_signal = clean_signal + np.random.normal(0, noise_level, n_samples)
        
        # フィルタリング
        filtered_signal = math_core.savitzky_golay_filter(
            noisy_signal,
            window_length=math_core.SAVGOL_WINDOW,
            polyorder=math_core.SAVGOL_POLYORDER
        )
        
        # グラフ表示
        df_plot = pd.DataFrame({
            'フレーム': range(n_samples),
            '元信号': clean_signal,
            'ノイズあり': noisy_signal,
            'フィルタ後': filtered_signal
        })
        
        st.line_chart(df_plot.set_index('フレーム'))
        
        # 統計情報
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ノイズあり標準偏差", f"{np.std(noisy_signal):.4f}")
        with col2:
            st.metric("フィルタ後標準偏差", f"{np.std(filtered_signal):.4f}")
        with col3:
            improvement = (1 - np.std(filtered_signal)/np.std(noisy_signal)) * 100
            st.metric("改善率", f"{improvement:.1f}%")
    
    # ========================================
    # タブ5: Phase 2 - 歩行イベント検出
    # ========================================
    with tab5:
        st.header("🦶 Phase 2: 歩行イベント検出")
        st.markdown("踵接地（Heel Strike）と足離地（Toe Off）を自動検出します")
        
        # GaitEventDetector 初期化
        detector = GaitEventDetector(sampling_rate=float(fps))
        
        # サンプルデータ生成オプション
        st.subheader("テストデータ設定")
        
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("データの長さ（秒）", 5, 20, 10)
            stride_frequency = st.slider("歩行頻度 (Hz)", 0.5, 2.0, 1.0, 0.1)
        
        with col2:
            noise_level_gait = st.slider("ノイズレベル", 0.0, 10.0, 2.0, 0.5)
            amplitude = st.slider("振幅 (pixel)", 10.0, 50.0, 30.0, 5.0)
        
        if st.button("サンプルデータを生成して検出", type="primary", key="detect_events"):
            # テストデータ生成
            n_frames = int(duration * fps)
            t = np.linspace(0, duration, n_frames)
            
            # 模擬的な踵とつま先のY座標
            # 踵：周期的に上下（地面に近づく＝Y座標が小さくなる）
            heel_y = -50 + amplitude * np.sin(2 * np.pi * stride_frequency * t)
            heel_y += np.random.normal(0, noise_level_gait, n_frames)
            
            # つま先：踵より少し位相がずれる
            toe_y = -40 + (amplitude * 0.8) * np.sin(2 * np.pi * stride_frequency * t + np.pi/6)
            toe_y += np.random.normal(0, noise_level_gait, n_frames)
            
            # イベント検出
            events = detector.detect_events(heel_y, toe_y)
            
            # 歩行周期の計算
            cycles = detector.calculate_gait_cycles(events)
            
            # 結果表示
            st.success("✅ 検出完了！")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("踵接地検出数", len(events['heel_strikes']))
            with col2:
                st.metric("足離地検出数", len(events['toe_offs']))
            with col3:
                st.metric("歩行周期数", len(cycles))
            
            # グラフ表示
            st.subheader("検出結果の可視化")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 踵とつま先のY座標をプロット
            ax.plot(t, heel_y, label='踵 Y座標', linewidth=1.5, alpha=0.7)
            ax.plot(t, toe_y, label='つま先 Y座標', linewidth=1.5, alpha=0.7)
            
            # 踵接地をマーク
            for hs_frame in events['heel_strikes']:
                ax.axvline(x=t[hs_frame], color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax.plot(t[hs_frame], heel_y[hs_frame], 'ro', markersize=8, label='踵接地' if hs_frame == events['heel_strikes'][0] else '')
            
            # 足離地をマーク
            for to_frame in events['toe_offs']:
                ax.axvline(x=t[to_frame], color='blue', linestyle='--', alpha=0.5, linewidth=1)
                ax.plot(t[to_frame], toe_y[to_frame], 'bs', markersize=8, label='足離地' if to_frame == events['toe_offs'][0] else '')
            
            ax.set_xlabel('時間 (秒)', fontsize=12)
            ax.set_ylabel('Y座標 (pixel)', fontsize=12)
            ax.set_title('歩行イベント検出結果', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # 歩行周期の詳細
            if len(cycles) > 0:
                st.subheader("歩行周期の詳細")
                
                cycles_df = pd.DataFrame(cycles)
                cycles_df['開始時刻 (秒)'] = cycles_df['start_frame'] / fps
                cycles_df['立脚期 (秒)'] = cycles_df['stance_duration'] / fps
                cycles_df['遊脚期 (秒)'] = cycles_df['swing_duration'] / fps
                cycles_df['ストライド時間 (秒)'] = cycles_df['stride_duration'] / fps
                
                display_df = cycles_df[[
                    '開始時刻 (秒)', 
                    '立脚期 (秒)', 
                    '遊脚期 (秒)', 
                    'ストライド時間 (秒)',
                    'stance_percentage'
                ]].copy()
                display_df.columns = [
                    '開始時刻', 
                    '立脚期', 
                    '遊脚期', 
                    'ストライド時間',
                    '立脚期割合 (%)'
                ]
                
                st.dataframe(display_df, use_container_width=True)
                
                # 平均値
                st.subheader("平均値")
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_stance = cycles_df['立脚期 (秒)'].mean()
                    st.metric("平均立脚期", f"{avg_stance:.3f} 秒")
                with col2:
                    avg_swing = cycles_df['遊脚期 (秒)'].mean()
                    st.metric("平均遊脚期", f"{avg_swing:.3f} 秒")
                with col3:
                    avg_stride = cycles_df['ストライド時間 (秒)'].mean()
                    st.metric("平均ストライド時間", f"{avg_stride:.3f} 秒")
    
    # ========================================
    # タブ6: Phase 3 - 歩行パラメータ計算
    # ========================================
    with tab6:
        st.header("📈 Phase 3: 歩行パラメータ計算")
        st.markdown("検出された歩行イベントから詳細な歩行パラメータを計算します")
        
        # GaitParameterCalculator 初期化
        st.subheader("設定")
        
        col1, col2 = st.columns(2)
        with col1:
            use_pixel_conversion = st.checkbox("ピクセルをメートルに変換", value=False)
            pixel_to_meter = None
            if use_pixel_conversion:
                pixel_to_meter = st.number_input(
                    "変換係数 (例: 100pixel=1mなら0.01)",
                    value=0.01,
                    format="%.4f",
                    min_value=0.0001,
                    max_value=1.0
                )
        
        with col2:
            normalize_spatial = st.checkbox("空間パラメータを正規化", value=False)
            normalization_length = None
            if normalize_spatial:
                normalization_length = st.number_input(
                    "正規化用の長さ (例: 大腿骨長 [pixel])",
                    value=200.0,
                    min_value=1.0
                )
        
        calculator = GaitParameterCalculator(
            sampling_rate=float(fps),
            pixel_to_meter=pixel_to_meter
        )
        
        # サンプルデータ生成
        st.subheader("テストデータ生成")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            duration_p3 = st.slider("データの長さ（秒）", 5, 20, 10, key="duration_p3")
            stride_freq_p3 = st.slider("歩行頻度 (Hz)", 0.5, 2.0, 1.0, 0.1, key="stride_freq_p3")
        
        with col2:
            noise_level_p3 = st.slider("ノイズレベル", 0.0, 10.0, 2.0, 0.5, key="noise_p3")
            amplitude_p3 = st.slider("振幅 (pixel)", 10.0, 50.0, 30.0, 5.0, key="amp_p3")
        
        with col3:
            walking_distance = st.slider("歩行距離 (pixel)", 100.0, 500.0, 300.0, 50.0)
        
        if st.button("パラメータを計算", type="primary", key="calc_params"):
            with st.spinner("計算中..."):
                # テストデータ生成
                n_frames = int(duration_p3 * fps)
                t = np.linspace(0, duration_p3, n_frames)
                
                # 踵とつま先のY座標
                heel_y = -50 + amplitude_p3 * np.sin(2 * np.pi * stride_freq_p3 * t)
                heel_y += np.random.normal(0, noise_level_p3, n_frames)
                
                toe_y = -40 + (amplitude_p3 * 0.8) * np.sin(2 * np.pi * stride_freq_p3 * t + np.pi/6)
                toe_y += np.random.normal(0, noise_level_p3, n_frames)
                
                # 踵の前方移動（X座標）を模擬
                heel_x = np.linspace(0, walking_distance, n_frames)
                heel_positions = np.column_stack([heel_x, heel_y])
                
                # GaitEventDetectorでイベント検出
                detector = GaitEventDetector(sampling_rate=float(fps))
                events = detector.detect_events(heel_y, toe_y)
                cycles = detector.calculate_gait_cycles(events)
                
                if len(cycles) == 0:
                    st.error("⚠️ 歩行周期が検出されませんでした")
                else:
                    # Phase 3: パラメータ計算
                    
                    # 1. 基本パラメータ
                    parameters = calculator.calculate_stride_parameters(cycles)
                    
                    # 2. 空間パラメータ
                    spatial_params = calculator.calculate_spatial_parameters(
                        heel_positions[:, 0],  # X座標のみ使用
                        events,
                        normalize_by=normalization_length if normalize_spatial else None
                    )
                    
                    # 3. 速度パラメータ
                    stride_times = [p.stride_time for p in parameters]
                    stride_lengths = spatial_params.get('stride_lengths', [])
                    
                    if len(stride_lengths) > 0:
                        speed_params = calculator.calculate_walking_speed(
                            stride_times[:len(stride_lengths)],
                            stride_lengths
                        )
                    else:
                        speed_params = {}
                    
                    # 4. 変動性
                    variability = calculator.calculate_variability(parameters)
                    
                    # 結果表示
                    st.success("✅ 計算完了！")
                    
                    # サマリー表示
                    st.subheader("📊 サマリー統計")
                    
                    summary_df = calculator.generate_summary_report(
                        parameters,
                        spatial_params=spatial_params,
                        speed_params=speed_params,
                        variability=variability
                    )
                    
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # 詳細な指標
                    st.subheader("🔍 詳細な指標")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_stride_time = np.mean([p.stride_time for p in parameters])
                        st.metric("平均ストライド時間", f"{avg_stride_time:.3f} 秒")
                    
                    with col2:
                        avg_cadence = np.mean([p.cadence for p in parameters if p.cadence])
                        st.metric("平均ケイデンス", f"{avg_cadence:.1f} steps/min")
                    
                    with col3:
                        if spatial_params and 'mean_stride_length' in spatial_params:
                            unit = 'm' if pixel_to_meter else 'pixel'
                            st.metric(f"平均ストライド長", f"{spatial_params['mean_stride_length']:.3f} {unit}")
                        else:
                            st.metric("平均ストライド長", "N/A")
                    
                    with col4:
                        if speed_params and 'mean_walking_speed' in speed_params:
                            unit = 'm/s' if pixel_to_meter else 'pixel/s'
                            st.metric(f"平均歩行速度", f"{speed_params['mean_walking_speed']:.3f} {unit}")
                        else:
                            st.metric("平均歩行速度", "N/A")
                    
                    # 変動性の表示
                    if variability:
                        st.subheader("📉 変動性")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "ストライド時間 変動係数 (CV)",
                                f"{variability.get('stride_time_cv', 0):.2f} %",
                                help="低いほど安定した歩行"
                            )
                        
                        with col2:
                            st.metric(
                                "ストライド時間 標準偏差",
                                f"{variability.get('stride_time_std', 0):.4f} 秒"
                            )
                        
                        # CVの評価
                        cv_value = variability.get('stride_time_cv', 0)
                        if cv_value < 3:
                            st.success("✓ 非常に安定した歩行パターン")
                        elif cv_value < 5:
                            st.info("✓ 安定した歩行パターン")
                        elif cv_value < 10:
                            st.warning("⚠ やや不安定な歩行パターン")
                        else:
                            st.error("⚠ 不安定な歩行パターン")
                    
                    # 各周期の詳細データ
                    st.subheader("📋 各周期の詳細")
                    
                    detail_data = []
                    for i, (param, cycle) in enumerate(zip(parameters, cycles)):
                        row = {
                            '周期': i + 1,
                            'ストライド時間 (秒)': f"{param.stride_time:.3f}",
                            '立脚期 (秒)': f"{param.stance_time:.3f}",
                            '遊脚期 (秒)': f"{param.swing_time:.3f}",
                            '立脚期割合 (%)': f"{param.stance_percentage:.1f}",
                            'ケイデンス (steps/min)': f"{param.cadence:.1f}" if param.cadence else "N/A"
                        }
                        
                        if i < len(stride_lengths):
                            unit = 'm' if pixel_to_meter else 'pixel'
                            row[f'ストライド長 ({unit})'] = f"{stride_lengths[i]:.3f}"
                        
                        detail_data.append(row)
                    
                    detail_df = pd.DataFrame(detail_data)
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                    
                    # グラフ表示
                    st.subheader("📊 時系列グラフ")
                    
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    
                    # 1. ストライド時間の推移
                    ax1 = axes[0, 0]
                    stride_times_plot = [p.stride_time for p in parameters]
                    ax1.plot(range(1, len(stride_times_plot) + 1), stride_times_plot, 'o-', linewidth=2, markersize=8)
                    ax1.axhline(y=np.mean(stride_times_plot), color='r', linestyle='--', label='平均')
                    ax1.set_xlabel('周期', fontsize=11)
                    ax1.set_ylabel('ストライド時間 (秒)', fontsize=11)
                    ax1.set_title('ストライド時間の推移', fontsize=12, fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # 2. 立脚期・遊脚期の割合
                    ax2 = axes[0, 1]
                    stance_pcts = [p.stance_percentage for p in parameters]
                    swing_pcts = [p.swing_percentage for p in parameters]
                    x_pos = range(1, len(parameters) + 1)
                    ax2.bar(x_pos, stance_pcts, label='立脚期', alpha=0.7)
                    ax2.bar(x_pos, swing_pcts, bottom=stance_pcts, label='遊脚期', alpha=0.7)
                    ax2.set_xlabel('周期', fontsize=11)
                    ax2.set_ylabel('割合 (%)', fontsize=11)
                    ax2.set_title('立脚期・遊脚期の割合', fontsize=12, fontweight='bold')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3, axis='y')
                    
                    # 3. ケイデンスの推移
                    ax3 = axes[1, 0]
                    cadences_plot = [p.cadence for p in parameters if p.cadence]
                    if len(cadences_plot) > 0:
                        ax3.plot(range(1, len(cadences_plot) + 1), cadences_plot, 's-', linewidth=2, markersize=8, color='green')
                        ax3.axhline(y=np.mean(cadences_plot), color='r', linestyle='--', label='平均')
                        ax3.set_xlabel('周期', fontsize=11)
                        ax3.set_ylabel('ケイデンス (steps/min)', fontsize=11)
                        ax3.set_title('ケイデンスの推移', fontsize=12, fontweight='bold')
                        ax3.legend()
                        ax3.grid(True, alpha=0.3)
                    
                    # 4. ストライド長の推移
                    ax4 = axes[1, 1]
                    if len(stride_lengths) > 0:
                        ax4.plot(range(1, len(stride_lengths) + 1), stride_lengths, '^-', linewidth=2, markersize=8, color='purple')
                        ax4.axhline(y=np.mean(stride_lengths), color='r', linestyle='--', label='平均')
                        unit = 'm' if pixel_to_meter else 'pixel'
                        ax4.set_xlabel('周期', fontsize=11)
                        ax4.set_ylabel(f'ストライド長 ({unit})', fontsize=11)
                        ax4.set_title('ストライド長の推移', fontsize=12, fontweight='bold')
                        ax4.legend()
                        ax4.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    # ========================================
    # タブ7: Phase 4 - CSVデータ分析
    # ========================================
    with tab7:
        st.header("📂 Phase 4: CSVデータ分析（統合システム）")
        st.markdown("実際のCSVデータを読み込んで、Phase 1-3の全機能を使った完全な分析を実行します")
        
        # ファイルアップロード
        st.subheader("1. CSVファイルのアップロード")
        uploaded_file = st.file_uploader(
            "歩行データのCSVファイルを選択してください",
            type=['csv'],
            help="MediaPipeやOpenPoseなどから出力されたCSVファイル"
        )
        
        if uploaded_file is not None:
            # データのプレビュー
            try:
                df_preview = pd.read_csv(uploaded_file)
                st.success(f"✅ ファイル読み込み成功: {len(df_preview)} 行 × {len(df_preview.columns)} 列")
                
                with st.expander("データプレビュー（先頭10行）"):
                    st.dataframe(df_preview.head(10), use_container_width=True)
                
                # カラム名の取得
                available_columns = list(df_preview.columns)
                
                # カラム選択
                st.subheader("2. カラムの選択")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**踵（Heel）のカラム**")
                    heel_x_col = st.selectbox("X座標", available_columns, key="heel_x")
                    heel_y_col = st.selectbox("Y座標", available_columns, key="heel_y")
                    heel_vis_col = st.selectbox("信頼度（オプション）", ['なし'] + available_columns, key="heel_vis")
                
                with col2:
                    st.markdown("**つま先（Toe）のカラム**")
                    toe_x_col = st.selectbox("X座標", available_columns, key="toe_x")
                    toe_y_col = st.selectbox("Y座標", available_columns, key="toe_y")
                    toe_vis_col = st.selectbox("信頼度（オプション）", ['なし'] + available_columns, key="toe_vis")
                
                # 分析設定
                st.subheader("3. 分析設定")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    apply_smoothing = st.checkbox("平滑化フィルタを適用", value=True)
                    if apply_smoothing:
                        smooth_window = st.slider("窓長", 3, 15, 5, 2)
                
                with col2:
                    use_normalization = st.checkbox("正規化を使用", value=False)
                    norm_length = None
                    if use_normalization:
                        norm_length = st.number_input("基準長（pixel）", value=200.0, min_value=1.0)
                
                with col3:
                    use_conversion = st.checkbox("ピクセル→メートル変換", value=False)
                    conversion_factor = None
                    if use_conversion:
                        conversion_factor = st.number_input(
                            "変換係数",
                            value=0.01,
                            format="%.4f",
                            min_value=0.0001
                        )
                
                # 分析実行ボタン
                if st.button("🚀 完全分析を実行", type="primary", key="run_full_analysis"):
                    with st.spinner("分析中... しばらくお待ちください"):
                        try:
                            # IntegratedGaitAnalyzerの初期化
                            analyzer = IntegratedGaitAnalyzer(
                                fps=float(fps),
                                use_z_axis=False,
                                min_visibility=0.5,
                                pixel_to_meter=conversion_factor
                            )
                            
                            # ファイルを一時保存
                            import tempfile
                            import os
                            
                            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
                                uploaded_file.seek(0)
                                tmp_file.write(uploaded_file.read().decode('utf-8'))
                                tmp_path = tmp_file.name
                            
                            # カラム名の辞書を作成
                            heel_cols = {'x': heel_x_col, 'y': heel_y_col}
                            if heel_vis_col != 'なし':
                                heel_cols['visibility'] = heel_vis_col
                            
                            toe_cols = {'x': toe_x_col, 'y': toe_y_col}
                            if toe_vis_col != 'なし':
                                toe_cols['visibility'] = toe_vis_col
                            
                            # 完全分析の実行
                            report = analyzer.run_full_analysis(
                                csv_path=tmp_path,
                                heel_cols=heel_cols,
                                toe_cols=toe_cols,
                                normalize_by=norm_length,
                                smooth=apply_smoothing
                            )
                            
                            # 一時ファイルを削除
                            os.unlink(tmp_path)
                            
                            # 結果の表示
                            st.success("🎉 分析完了！")
                            
                            # 統計サマリー
                            st.subheader("📊 統計サマリー")
                            
                            stats = report['statistics']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("総フレーム数", stats['総フレーム数'])
                            with col2:
                                st.metric("総時間", f"{stats['総時間 (秒)']:.1f} 秒")
                            with col3:
                                st.metric("歩行周期数", stats['完全な歩行周期数'])
                            with col4:
                                if 'ストライド時間CV (%)' in stats:
                                    st.metric("変動係数", f"{stats['ストライド時間CV (%)']:.2f}%")
                            
                            # サマリーテーブル
                            st.subheader("📋 パラメータサマリー")
                            st.dataframe(report['summary'], use_container_width=True, hide_index=True)
                            
                            # イベント情報
                            if 'events' in report and len(report['events']) > 0:
                                st.subheader("🦶 検出されたイベント")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    heel_strikes = report['events'][report['events']['event_type'] == 'heel_strike']
                                    st.write(f"**踵接地: {len(heel_strikes)}回**")
                                    if len(heel_strikes) > 0:
                                        st.dataframe(
                                            heel_strikes[['frame', 'time']].head(10),
                                            use_container_width=True,
                                            hide_index=True
                                        )
                                
                                with col2:
                                    toe_offs = report['events'][report['events']['event_type'] == 'toe_off']
                                    st.write(f"**足離地: {len(toe_offs)}回**")
                                    if len(toe_offs) > 0:
                                        st.dataframe(
                                            toe_offs[['frame', 'time']].head(10),
                                            use_container_width=True,
                                            hide_index=True
                                        )
                            
                            # 周期詳細
                            if 'cycles_detail' in report and len(report['cycles_detail']) > 0:
                                st.subheader("🔄 歩行周期の詳細")
                                st.dataframe(
                                    report['cycles_detail'],
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            # グラフ表示
                            st.subheader("📈 可視化")
                            
                            if analyzer.processed_data is not None and analyzer.events is not None:
                                fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                                
                                # 時間軸の作成
                                n_frames = len(analyzer.processed_data)
                                time_axis = np.arange(n_frames) / fps
                                
                                # 上段: 踵とつま先のY座標 + イベント
                                ax1 = axes[0]
                                heel_y_data = analyzer.processed_data[heel_y_col].values
                                toe_y_data = analyzer.processed_data[toe_y_col].values
                                
                                ax1.plot(time_axis, heel_y_data, label='踵 Y座標', linewidth=1.5, alpha=0.7)
                                ax1.plot(time_axis, toe_y_data, label='つま先 Y座標', linewidth=1.5, alpha=0.7)
                                
                                # 踵接地をマーク
                                for hs_frame in analyzer.events['heel_strikes']:
                                    ax1.axvline(x=time_axis[hs_frame], color='red', linestyle='--', alpha=0.3)
                                    if hs_frame == analyzer.events['heel_strikes'][0]:
                                        ax1.plot(time_axis[hs_frame], heel_y_data[hs_frame], 'ro', 
                                               markersize=8, label='踵接地')
                                    else:
                                        ax1.plot(time_axis[hs_frame], heel_y_data[hs_frame], 'ro', markersize=8)
                                
                                # 足離地をマーク
                                for to_frame in analyzer.events['toe_offs']:
                                    ax1.axvline(x=time_axis[to_frame], color='blue', linestyle='--', alpha=0.3)
                                    if to_frame == analyzer.events['toe_offs'][0]:
                                        ax1.plot(time_axis[to_frame], toe_y_data[to_frame], 'bs', 
                                               markersize=8, label='足離地')
                                    else:
                                        ax1.plot(time_axis[to_frame], toe_y_data[to_frame], 'bs', markersize=8)
                                
                                ax1.set_xlabel('時間 (秒)', fontsize=12)
                                ax1.set_ylabel('Y座標 (pixel)', fontsize=12)
                                ax1.set_title('歩行イベント検出結果', fontsize=14, fontweight='bold')
                                ax1.legend(loc='best')
                                ax1.grid(True, alpha=0.3)
                                
                                # 下段: ストライド時間の推移
                                ax2 = axes[1]
                                if analyzer.parameters and len(analyzer.parameters) > 0:
                                    stride_times = [p.stride_time for p in analyzer.parameters]
                                    cycle_numbers = range(1, len(stride_times) + 1)
                                    
                                    ax2.plot(cycle_numbers, stride_times, 'o-', linewidth=2, markersize=8)
                                    ax2.axhline(y=np.mean(stride_times), color='r', linestyle='--', 
                                              linewidth=2, label=f'平均: {np.mean(stride_times):.3f}秒')
                                    
                                    ax2.set_xlabel('周期番号', fontsize=12)
                                    ax2.set_ylabel('ストライド時間 (秒)', fontsize=12)
                                    ax2.set_title('ストライド時間の推移', fontsize=14, fontweight='bold')
                                    ax2.legend()
                                    ax2.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            # ダウンロードボタン
                            st.subheader("💾 レポートのダウンロード")
                            
                            # CSVとしてダウンロード
                            if 'cycles_detail' in report:
                                csv_data = report['cycles_detail'].to_csv(index=False)
                                st.download_button(
                                    label="📥 周期詳細をCSVでダウンロード",
                                    data=csv_data,
                                    file_name="gait_cycles_detail.csv",
                                    mime="text/csv"
                                )
                        
                        except Exception as e:
                            st.error(f"❌ 分析中にエラーが発生しました: {str(e)}")
                            import traceback
                            with st.expander("詳細なエラー情報"):
                                st.code(traceback.format_exc())
            
            except Exception as e:
                st.error(f"❌ ファイルの読み込みに失敗しました: {str(e)}")
        
        else:
            st.info("👆 CSVファイルをアップロードして分析を開始してください")
            
            # サンプルデータの説明
            with st.expander("📖 必要なCSVフォーマット"):
                st.markdown("""
                CSVファイルには以下のカラムが必要です：
                
                - **踵のX座標**: 例 `heel_x`, `right_heel_x`
                - **踵のY座標**: 例 `heel_y`, `right_heel_y`
                - **つま先のX座標**: 例 `toe_x`, `right_toe_x`
                - **つま先のY座標**: 例 `toe_y`, `right_toe_y`
                - **信頼度（オプション）**: 例 `heel_visibility`, `toe_visibility`
                
                サンプル:
                ```
                frame,heel_x,heel_y,heel_visibility,toe_x,toe_y,toe_visibility
                0,100.5,200.3,0.95,120.2,205.1,0.92
                1,101.2,198.7,0.94,121.1,203.5,0.93
                ...
                ```
                """)
    
    # ========================================
    # フッター
    # ========================================
    st.markdown("---")
    st.markdown("""
    ### ✅ Phase 1 チェックリスト
    - ✓ 角度計算は180°表記（度数法）
    - ✓ Z軸の扱いを選択可能
    - ✓ 信頼度による欠損値検出
    - ✓ Savitzky-Golayフィルタによる平滑化
    - ✓ 大腿骨長による正規化
    
    ### ✅ Phase 2 チェックリスト
    - ✓ 踵接地（Heel Strike）の自動検出
    - ✓ 足離地（Toe Off）の自動検出
    - ✓ 歩行周期の計算
    - ✓ 立脚期・遊脚期の分析
    
    ### ✅ Phase 3 チェックリスト
    - ✓ ストライド時間・立脚期・遊脚期の計算
    - ✓ ストライド長・ステップ長の計算
    - ✓ 歩行速度・ケイデンスの計算
    - ✓ 変動性（CV）の計算
    - ✓ サマリーレポートの生成
    
    ### ✅ Phase 4 チェックリスト
    - ✓ CSVファイルの読み込み
    - ✓ データ前処理と平滑化
    - ✓ Phase 1-3の統合
    - ✓ 完全な分析レポート生成
    - ✓ 結果の可視化とCSVエクスポート
    
    **🎉 全フェーズ完成！実データで歩行分析が可能になりました！**
    """)


if __name__ == "__main__":
    main()
