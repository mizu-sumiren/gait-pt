import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from typing import Dict, List, Tuple, Optional, Union
import warnings

class GaitMathCore:
    """
    歩行分析のための数学的計算基盤クラス
    
    設計原則:
    - 全ての角度は度数法(degrees)で出力
    - 欠損値に対する安全策を実装
    - 60fps撮影を標準とした時系列処理
    - 大腿骨長による正規化を標準搭載
    
    Author: Based on PT clinical requirements
    Reference: Sakane (2025), Rancho Los Amigos Gait Analysis
    """
    
    # 信頼度閾値（MediaPipe Visibility）
    VISIBILITY_THRESHOLD = 0.5
    
    # 60fps用のフィルタパラメータ（約83ms窓）
    SAVGOL_WINDOW = 5
    SAVGOL_POLYORDER = 2
    
    def __init__(self, fps: int = 60):
        """
        Parameters:
        -----------
        fps : int
            動画のフレームレート（デフォルト60fps）
        """
        self.fps = fps
        self.frame_interval = 1.0 / fps  # 秒
        
    @staticmethod
    def calculate_angle_3d(
        p1: Dict[str, float], 
        p2: Dict[str, float], 
        p3: Dict[str, float],
        use_z_axis: bool = False,
        min_visibility: float = VISIBILITY_THRESHOLD
    ) -> Optional[float]:
        """
        3点から関節角度を計算（p2が頂点）
        
        Parameters:
        -----------
        p1, p2, p3 : dict
            {'x': float, 'y': float, 'z': float, 'visibility': float}
            p2が関節点（例: 膝）、p1とp3が隣接点（例: 股関節と足首）
        use_z_axis : bool
            True: 3次元ベクトルで計算
            False: XY平面（側面分析）のみ使用
        min_visibility : float
            最低信頼度（デフォルト0.5）
            
        Returns:
        --------
        float : 関節角度（度数法、0-180°）
            内角を返す（伸展0°、屈曲180°方向）
        None : いずれかの点の信頼度が閾値未満の場合
        
        Notes:
        ------
        - ベクトルの内積を用いた計算: cos(θ) = (v1·v2) / (|v1||v2|)
        - 180°表記を保証（np.arccos → np.degrees）
        - Z軸を無視する場合、側面撮影時の奥行き誤差を排除
        """
        # 信頼度チェック
        if any(p.get('visibility', 0) < min_visibility for p in [p1, p2, p3]):
            return None
        
        # ベクトル構築
        if use_z_axis:
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
        else:
            # XY平面のみ（側面分析）
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        # ゼロベクトルチェック
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            warnings.warn("ベクトルの長さがほぼゼロです。座標が重複している可能性があります。")
            return None
        
        # 内積計算
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        
        # 数値誤差対策（cos_angleが[-1, 1]を超えないようクリップ）
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # ラジアン → 度数法変換（必須）
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    @staticmethod
    def savitzky_golay_filter(
        data: Union[List[float], np.ndarray],
        window_length: int = SAVGOL_WINDOW,
        polyorder: int = SAVGOL_POLYORDER,
        handle_nan: bool = True
    ) -> np.ndarray:
        """
        Savitzky-Golayフィルタによるノイズ除去
        
        Parameters:
        -----------
        data : array-like
            時系列データ（例: 関節のY座標列）
        window_length : int
            窓の幅（奇数、60fpsでは5フレーム≈83ms）
        polyorder : int
            多項式の次数（2次が標準）
        handle_nan : bool
            NaNを含む場合、線形補間してからフィルタリング
            
        Returns:
        --------
        np.ndarray : 平滑化されたデータ
        
        Notes:
        ------
        - window_lengthはデータ長より小さく、奇数である必要がある
        - 欠損値がある場合は事前に補間処理を推奨
        """
        data_array = np.array(data, dtype=float)
        
        # NaN処理
        if handle_nan and np.any(np.isnan(data_array)):
            # NaNのインデックスを取得
            valid_idx = ~np.isnan(data_array)
            if np.sum(valid_idx) < 2:
                warnings.warn("有効なデータ点が2点未満のため、フィルタリングをスキップします。")
                return data_array
            
            # 線形補間
            x_valid = np.where(valid_idx)[0]
            y_valid = data_array[valid_idx]
            x_all = np.arange(len(data_array))
            data_array = np.interp(x_all, x_valid, y_valid)
        
        # データ長チェック
        if len(data_array) < window_length:
            warnings.warn(f"データ長({len(data_array)})が窓幅({window_length})未満です。フィルタリングをスキップします。")
            return data_array
        
        # 窓幅を奇数に調整
        if window_length % 2 == 0:
            window_length += 1
        
        # フィルタ適用
        try:
            filtered = savgol_filter(data_array, window_length, polyorder)
        except Exception as e:
            warnings.warn(f"Savitzky-Golayフィルタ適用エラー: {e}")
            return data_array
        
        return filtered
    
    @staticmethod
    def spline_interpolate(
        time_points: np.ndarray,
        data: np.ndarray,
        missing_mask: np.ndarray,
        smoothing_factor: float = 0.0
    ) -> np.ndarray:
        """
        スプライン補間による欠損値補完
        
        Parameters:
        -----------
        time_points : np.ndarray
            時間軸（フレーム番号またはタイムスタンプ）
        data : np.ndarray
            元データ（NaNまたは欠損値を含む）
        missing_mask : np.ndarray (bool)
            Trueが欠損位置
        smoothing_factor : float
            スプラインの平滑化係数（0で厳密補間）
            
        Returns:
        --------
        np.ndarray : 補間されたデータ
        
        Notes:
        ------
        - Visibility < 0.5の点を欠損として扱う想定
        - 有効点が3点未満の場合は線形補間にフォールバック
        """
        valid_mask = ~missing_mask
        valid_points = time_points[valid_mask]
        valid_data = data[valid_mask]
        
        if len(valid_points) < 3:
            # スプライン補間には最低3点必要
            warnings.warn("有効点が3点未満のため、線形補間を使用します。")
            return np.interp(time_points, valid_points, valid_data)
        
        try:
            # UnivariateSpline（3次スプライン）
            spline = UnivariateSpline(valid_points, valid_data, s=smoothing_factor, k=3)
            interpolated = spline(time_points)
        except Exception as e:
            warnings.warn(f"スプライン補間エラー: {e}。線形補間にフォールバック。")
            interpolated = np.interp(time_points, valid_points, valid_data)
        
        return interpolated
    
    @staticmethod
    def calculate_segment_length_3d(
        p1: Dict[str, float],
        p2: Dict[str, float],
        use_z_axis: bool = False,
        min_visibility: float = VISIBILITY_THRESHOLD
    ) -> Optional[float]:
        """
        2点間の距離（セグメント長）を計算
        
        Parameters:
        -----------
        p1, p2 : dict
            座標と信頼度を含む辞書
        use_z_axis : bool
            3次元距離か2次元距離か
        min_visibility : float
            最低信頼度
            
        Returns:
        --------
        float : ユークリッド距離
        None : 信頼度不足の場合
        
        Notes:
        ------
        - 大腿骨長（HIP→KNEE）の算出に使用
        - 正規化の基準単位として利用
        """
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
    def normalize_by_segment_length(
        value: float,
        segment_length: float,
        segment_name: str = "大腿骨長"
    ) -> Optional[float]:
        """
        身体比率による正規化
        
        Parameters:
        -----------
        value : float
            正規化したい値（例: 体幹の上下移動量 [pixel]）
        segment_length : float
            基準となるセグメント長（例: 大腿骨長 [pixel]）
        segment_name : str
            セグメント名（エラーメッセージ用）
            
        Returns:
        --------
        float : 正規化された値（無次元比率）
        None : セグメント長が不正な場合
        
        Notes:
        ------
        - カメラ距離の影響を除去
        - 体格差を吸収した比較が可能
        
        Example:
        --------
        体幹の上下移動が50ピクセル、大腿骨長が200ピクセル
        → 正規化値 = 50/200 = 0.25（体幹長の25%相当）
        """
        if segment_length <= 0 or np.isnan(segment_length):
            warnings.warn(f"{segment_name}が不正な値です: {segment_length}")
            return None
        
        normalized = value / segment_length
        return normalized
    
    @staticmethod
    def calculate_velocity(
        position_series: np.ndarray,
        fps: int = 60
    ) -> np.ndarray:
        """
        位置データから速度を算出（中心差分）
        
        Parameters:
        -----------
        position_series : np.ndarray
            時系列の位置データ（例: 踵のY座標）
        fps : int
            フレームレート
            
        Returns:
        --------
        np.ndarray : 速度 [単位/秒]
        
        Notes:
        ------
        - Initial Contact検出に使用（速度≈0）
        - 中心差分: v[i] = (pos[i+1] - pos[i-1]) / (2Δt)
        """
        dt = 1.0 / fps
        velocity = np.gradient(position_series, dt)
        return velocity
    
    def preprocess_landmark_timeseries(
        self,
        df: pd.DataFrame,
        coord_columns: List[str],
        visibility_column: str = 'visibility',
        apply_filter: bool = True
    ) -> pd.DataFrame:
        """
        ランドマーク時系列データの前処理パイプライン
        
        Parameters:
        -----------
        df : pd.DataFrame
            フレームごとのランドマーク座標
            必須列: coord_columns + [visibility_column]
        coord_columns : list of str
            処理対象の座標列（例: ['x', 'y', 'z']）
        visibility_column : str
            信頼度列の名前
        apply_filter : bool
            Savitzky-Golayフィルタを適用するか
            
        Returns:
        --------
        pd.DataFrame : 前処理済みデータ
            - 欠損値補間済み
            - （オプション）平滑化済み
            
        Processing Steps:
        -----------------
        1. Visibility < 0.5 の点を欠損としてマーク
        2. スプライン補間で補完
        3. Savitzky-Golayフィルタで平滑化
        
        Notes:
        ------
        - このメソッドは各ランドマーク（例: 右膝）ごとに呼び出す
        - 入力dfは単一ランドマークの時系列を想定
        """
        df_processed = df.copy()
        
        # 欠損マスク作成
        missing_mask = df_processed[visibility_column] < self.VISIBILITY_THRESHOLD
        
        # 時間軸（フレーム番号）
        time_points = np.arange(len(df_processed))
        
        # 各座標軸を補間
        for col in coord_columns:
            if col not in df_processed.columns:
                continue
            
            data = df_processed[col].values
            
            # スプライン補間
            interpolated = self.spline_interpolate(
                time_points, data, missing_mask, smoothing_factor=0.0
            )
            
            df_processed[col] = interpolated
        
        # Savitzky-Golayフィルタ適用
        if apply_filter:
            for col in coord_columns:
                if col not in df_processed.columns:
                    continue
                
                filtered = self.savitzky_golay_filter(
                    df_processed[col].values,
                    window_length=self.SAVGOL_WINDOW,
                    polyorder=self.SAVGOL_POLYORDER
                )
                
                df_processed[col] = filtered
        
        return df_processed


# ========================================
# 使用例（テストケース）
# ========================================

if __name__ == "__main__":
    # 初期化
    math_core = GaitMathCore(fps=60)
    
    # テスト1: 角度計算（膝関節90度屈曲を想定）
    hip = {'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 0.9}
    knee = {'x': 0.5, 'y': 0.3, 'z': 0.0, 'visibility': 0.9}
    ankle = {'x': 0.7, 'y': 0.3, 'z': 0.0, 'visibility': 0.9}
    
    angle = math_core.calculate_angle_3d(hip, knee, ankle, use_z_axis=False)
    print(f"膝関節角度: {angle:.2f}°")  # 期待値: 約90°
    
    # テスト2: セグメント長計算
    femur_length = math_core.calculate_segment_length_3d(hip, knee, use_z_axis=False)
    print(f"大腿骨長（2D）: {femur_length:.4f}")
    
    # テスト3: 正規化
    trunk_movement = 0.05  # 仮の上下移動量
    normalized_movement = math_core.normalize_by_segment_length(
        trunk_movement, femur_length, "大腿骨長"
    )
    print(f"正規化された体幹移動: {normalized_movement:.4f}")
    
    # テスト4: フィルタリング
    noisy_data = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
    filtered_data = math_core.savitzky_golay_filter(noisy_data)
    print(f"フィルタ前の標準偏差: {np.std(noisy_data):.4f}")
    print(f"フィルタ後の標準偏差: {np.std(filtered_data):.4f}")
    
    print("\n✓ GaitMathCore の基本機能テスト完了")
