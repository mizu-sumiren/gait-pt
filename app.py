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
    
    # タブ分け（Phase 2のタブを追加）
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📐 角度計算テスト", 
        "📏 セグメント長計算", 
        "🔄 正規化テスト",
        "📊 フィルタリングテスト",
        "🦶 歩行イベント検出（Phase 2）"
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
    
    **次のステップ**: Phase 3（GaitParameterCalculator）の実装へ
    """)


if __name__ == "__main__":
    main()
