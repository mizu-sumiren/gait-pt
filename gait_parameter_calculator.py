"""
歩行パラメータ計算器 (GaitParameterCalculator)
Phase 3: 歩行速度、ストライド長、ケイデンス等の計算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class GaitParameters:
    """歩行パラメータを格納するデータクラス"""
    
    # 時間パラメータ
    stride_time: float  # ストライド時間 [秒]
    stance_time: float  # 立脚期時間 [秒]
    swing_time: float   # 遊脚期時間 [秒]
    double_support_time: Optional[float] = None  # 両脚支持期時間 [秒]
    
    # 時間比率
    stance_percentage: float = 0.0  # 立脚期の割合 [%]
    swing_percentage: float = 0.0   # 遊脚期の割合 [%]
    
    # 空間パラメータ
    stride_length: Optional[float] = None  # ストライド長 [pixel or m]
    step_length: Optional[float] = None    # ステップ長 [pixel or m]
    step_width: Optional[float] = None     # 歩幅（左右） [pixel or m]
    
    # 速度パラメータ
    walking_speed: Optional[float] = None  # 歩行速度 [pixel/s or m/s]
    cadence: Optional[float] = None        # ケイデンス [steps/min]
    
    # 変動性
    stride_time_variability: Optional[float] = None  # ストライド時間の変動係数


class GaitParameterCalculator:
    """
    歩行パラメータを計算するクラス
    """
    
    def __init__(self, 
                 sampling_rate: float = 30.0,
                 pixel_to_meter: Optional[float] = None):
        """
        Parameters:
        -----------
        sampling_rate : float
            サンプリングレート [Hz]
        pixel_to_meter : float, optional
            ピクセルからメートルへの変換係数
            例: 100 pixel = 1 meter なら 0.01
        """
        self.sampling_rate = sampling_rate
        self.pixel_to_meter = pixel_to_meter
    
    def calculate_stride_parameters(self,
                                    cycles: List[Dict]) -> List[GaitParameters]:
        """
        各歩行周期の基本パラメータを計算
        
        Parameters:
        -----------
        cycles : List[Dict]
            GaitEventDetectorから得られた歩行周期のリスト
            
        Returns:
        --------
        parameters : List[GaitParameters]
            各周期の歩行パラメータ
        """
        parameters = []
        
        for cycle in cycles:
            # 時間パラメータ（フレームから秒に変換）
            stride_time = cycle['stride_duration'] / self.sampling_rate
            stance_time = cycle['stance_duration'] / self.sampling_rate
            swing_time = cycle['swing_duration'] / self.sampling_rate
            
            # 時間比率
            stance_pct = cycle['stance_percentage']
            swing_pct = 100.0 - stance_pct
            
            # ケイデンス（1分あたりのステップ数）
            # 1ストライド = 2ステップなので、2倍する
            cadence = (60.0 / stride_time) * 2 if stride_time > 0 else None
            
            params = GaitParameters(
                stride_time=stride_time,
                stance_time=stance_time,
                swing_time=swing_time,
                stance_percentage=stance_pct,
                swing_percentage=swing_pct,
                cadence=cadence
            )
            
            parameters.append(params)
        
        return parameters
    
    def calculate_spatial_parameters(self,
                                     heel_positions: np.ndarray,
                                     events: Dict[str, List[int]],
                                     normalize_by: Optional[float] = None) -> Dict[str, float]:
        """
        空間的パラメータ（ストライド長、ステップ長など）を計算
        
        Parameters:
        -----------
        heel_positions : np.ndarray
            踵の位置の時系列データ (N×2 または N×3: x, y, [z])
        events : Dict[str, List[int]]
            検出された歩行イベント
        normalize_by : float, optional
            正規化用の長さ（例: 大腿骨長）
            
        Returns:
        --------
        spatial_params : Dict[str, float]
            空間パラメータの辞書
            - 'mean_stride_length': 平均ストライド長
            - 'mean_step_length': 平均ステップ長
            - 'mean_step_width': 平均ステップ幅
        """
        heel_strikes = events['heel_strikes']
        
        if len(heel_strikes) < 2:
            return {}
        
        stride_lengths = []
        
        # ストライド長の計算（連続する2つの踵接地間の距離）
        for i in range(len(heel_strikes) - 1):
            hs1 = heel_strikes[i]
            hs2 = heel_strikes[i + 1]
            
            # X方向（進行方向）の距離
            if heel_positions.ndim == 1:
                # 1次元データの場合
                stride_length = abs(heel_positions[hs2] - heel_positions[hs1])
            else:
                # 2次元以上のデータの場合
                pos1 = heel_positions[hs1]
                pos2 = heel_positions[hs2]
                
                # 主に進行方向（X軸）の変位を使用
                stride_length = np.linalg.norm(pos2[:2] - pos1[:2])
            
            stride_lengths.append(stride_length)
        
        mean_stride_length = np.mean(stride_lengths) if len(stride_lengths) > 0 else 0.0
        
        # 正規化
        if normalize_by is not None and normalize_by > 0:
            mean_stride_length = mean_stride_length / normalize_by
        
        # メートル変換
        if self.pixel_to_meter is not None:
            mean_stride_length = mean_stride_length * self.pixel_to_meter
        
        spatial_params = {
            'mean_stride_length': mean_stride_length,
            'stride_lengths': stride_lengths,
            # ステップ長は通常ストライド長の半分
            'mean_step_length': mean_stride_length / 2.0
        }
        
        return spatial_params
    
    def calculate_walking_speed(self,
                               stride_times: List[float],
                               stride_lengths: List[float]) -> Dict[str, float]:
        """
        歩行速度を計算
        
        Parameters:
        -----------
        stride_times : List[float]
            ストライド時間のリスト [秒]
        stride_lengths : List[float]
            ストライド長のリスト [pixel or m]
            
        Returns:
        --------
        speed_params : Dict[str, float]
            速度パラメータの辞書
        """
        if len(stride_times) != len(stride_lengths) or len(stride_times) == 0:
            return {'mean_walking_speed': 0.0, 'walking_speeds': []}
        
        # 各ストライドの速度
        walking_speeds = [
            length / time if time > 0 else 0.0
            for length, time in zip(stride_lengths, stride_times)
        ]
        
        mean_speed = np.mean(walking_speeds)
        
        return {
            'mean_walking_speed': mean_speed,
            'walking_speeds': walking_speeds,
            'speed_variability': np.std(walking_speeds) if len(walking_speeds) > 1 else 0.0
        }
    
    def calculate_variability(self,
                             parameters: List[GaitParameters]) -> Dict[str, float]:
        """
        歩行パラメータの変動性を計算
        
        Parameters:
        -----------
        parameters : List[GaitParameters]
            歩行パラメータのリスト
            
        Returns:
        --------
        variability : Dict[str, float]
            変動性の指標
            - 'stride_time_cv': ストライド時間の変動係数 (CV)
            - 'stride_time_std': ストライド時間の標準偏差
        """
        if len(parameters) < 2:
            return {}
        
        stride_times = [p.stride_time for p in parameters]
        
        mean_stride_time = np.mean(stride_times)
        std_stride_time = np.std(stride_times)
        
        # 変動係数 (Coefficient of Variation)
        cv = (std_stride_time / mean_stride_time * 100) if mean_stride_time > 0 else 0.0
        
        return {
            'stride_time_cv': cv,
            'stride_time_std': std_stride_time,
            'mean_stride_time': mean_stride_time
        }
    
    def calculate_symmetry(self,
                          left_params: List[GaitParameters],
                          right_params: List[GaitParameters]) -> Dict[str, float]:
        """
        左右の対称性を計算
        
        Parameters:
        -----------
        left_params : List[GaitParameters]
            左足の歩行パラメータ
        right_params : List[GaitParameters]
            右足の歩行パラメータ
            
        Returns:
        --------
        symmetry : Dict[str, float]
            対称性指標
            - 'stride_time_symmetry': ストライド時間の対称性比
            - 'stance_time_symmetry': 立脚期時間の対称性比
        """
        if len(left_params) == 0 or len(right_params) == 0:
            return {}
        
        left_stride_times = [p.stride_time for p in left_params]
        right_stride_times = [p.stride_time for p in right_params]
        
        mean_left_stride = np.mean(left_stride_times)
        mean_right_stride = np.mean(right_stride_times)
        
        # 対称性比（1.0が完全に対称）
        stride_symmetry = min(mean_left_stride, mean_right_stride) / max(mean_left_stride, mean_right_stride)
        
        # 立脚期の対称性
        left_stance_times = [p.stance_time for p in left_params]
        right_stance_times = [p.stance_time for p in right_params]
        
        mean_left_stance = np.mean(left_stance_times)
        mean_right_stance = np.mean(right_stance_times)
        
        stance_symmetry = min(mean_left_stance, mean_right_stance) / max(mean_left_stance, mean_right_stance)
        
        return {
            'stride_time_symmetry': stride_symmetry,
            'stance_time_symmetry': stance_symmetry,
            'left_mean_stride_time': mean_left_stride,
            'right_mean_stride_time': mean_right_stride
        }
    
    def generate_summary_report(self,
                               parameters: List[GaitParameters],
                               spatial_params: Optional[Dict] = None,
                               speed_params: Optional[Dict] = None,
                               variability: Optional[Dict] = None) -> pd.DataFrame:
        """
        サマリーレポートを生成
        
        Parameters:
        -----------
        parameters : List[GaitParameters]
            歩行パラメータのリスト
        spatial_params : Dict, optional
            空間パラメータ
        speed_params : Dict, optional
            速度パラメータ
        variability : Dict, optional
            変動性パラメータ
            
        Returns:
        --------
        summary_df : pd.DataFrame
            サマリーレポート
        """
        if len(parameters) == 0:
            return pd.DataFrame()
        
        # 基本統計
        stride_times = [p.stride_time for p in parameters]
        stance_times = [p.stance_time for p in parameters]
        swing_times = [p.swing_time for p in parameters]
        cadences = [p.cadence for p in parameters if p.cadence is not None]
        
        summary_data = {
            'パラメータ': [],
            '平均': [],
            '標準偏差': [],
            '最小': [],
            '最大': []
        }
        
        # ストライド時間
        summary_data['パラメータ'].append('ストライド時間 (秒)')
        summary_data['平均'].append(f"{np.mean(stride_times):.3f}")
        summary_data['標準偏差'].append(f"{np.std(stride_times):.3f}")
        summary_data['最小'].append(f"{np.min(stride_times):.3f}")
        summary_data['最大'].append(f"{np.max(stride_times):.3f}")
        
        # 立脚期時間
        summary_data['パラメータ'].append('立脚期時間 (秒)')
        summary_data['平均'].append(f"{np.mean(stance_times):.3f}")
        summary_data['標準偏差'].append(f"{np.std(stance_times):.3f}")
        summary_data['最小'].append(f"{np.min(stance_times):.3f}")
        summary_data['最大'].append(f"{np.max(stance_times):.3f}")
        
        # 遊脚期時間
        summary_data['パラメータ'].append('遊脚期時間 (秒)')
        summary_data['平均'].append(f"{np.mean(swing_times):.3f}")
        summary_data['標準偏差'].append(f"{np.std(swing_times):.3f}")
        summary_data['最小'].append(f"{np.min(swing_times):.3f}")
        summary_data['最大'].append(f"{np.max(swing_times):.3f}")
        
        # ケイデンス
        if len(cadences) > 0:
            summary_data['パラメータ'].append('ケイデンス (steps/min)')
            summary_data['平均'].append(f"{np.mean(cadences):.1f}")
            summary_data['標準偏差'].append(f"{np.std(cadences):.1f}")
            summary_data['最小'].append(f"{np.min(cadences):.1f}")
            summary_data['最大'].append(f"{np.max(cadences):.1f}")
        
        # 空間パラメータ
        if spatial_params and 'mean_stride_length' in spatial_params:
            stride_lengths = spatial_params.get('stride_lengths', [])
            if len(stride_lengths) > 0:
                unit = 'm' if self.pixel_to_meter else 'pixel'
                summary_data['パラメータ'].append(f'ストライド長 ({unit})')
                summary_data['平均'].append(f"{np.mean(stride_lengths):.3f}")
                summary_data['標準偏差'].append(f"{np.std(stride_lengths):.3f}")
                summary_data['最小'].append(f"{np.min(stride_lengths):.3f}")
                summary_data['最大'].append(f"{np.max(stride_lengths):.3f}")
        
        # 速度パラメータ
        if speed_params and 'mean_walking_speed' in speed_params:
            unit = 'm/s' if self.pixel_to_meter else 'pixel/s'
            summary_data['パラメータ'].append(f'歩行速度 ({unit})')
            summary_data['平均'].append(f"{speed_params['mean_walking_speed']:.3f}")
            summary_data['標準偏差'].append(f"{speed_params.get('speed_variability', 0):.3f}")
            summary_data['最小'].append('-')
            summary_data['最大'].append('-')
        
        # 変動性
        if variability and 'stride_time_cv' in variability:
            summary_data['パラメータ'].append('ストライド時間変動係数 (%)')
            summary_data['平均'].append(f"{variability['stride_time_cv']:.2f}")
            summary_data['標準偏差'].append('-')
            summary_data['最小'].append('-')
            summary_data['最大'].append('-')
        
        summary_df = pd.DataFrame(summary_data)
        
        return summary_df


# テスト用
if __name__ == "__main__":
    # サンプルデータ
    sample_cycles = [
        {
            'start_frame': 0,
            'end_frame': 30,
            'toe_off_frame': 18,
            'stance_duration': 18,
            'swing_duration': 12,
            'stride_duration': 30,
            'stance_percentage': 60.0
        },
        {
            'start_frame': 30,
            'end_frame': 60,
            'toe_off_frame': 48,
            'stance_duration': 18,
            'swing_duration': 12,
            'stride_duration': 30,
            'stance_percentage': 60.0
        }
    ]
    
    # 計算器の作成
    calculator = GaitParameterCalculator(sampling_rate=30.0)
    
    # パラメータ計算
    parameters = calculator.calculate_stride_parameters(sample_cycles)
    
    print("=== 歩行パラメータ ===")
    for i, param in enumerate(parameters):
        print(f"\n周期 {i+1}:")
        print(f"  ストライド時間: {param.stride_time:.3f} 秒")
        print(f"  立脚期時間: {param.stance_time:.3f} 秒")
        print(f"  遊脚期時間: {param.swing_time:.3f} 秒")
        print(f"  立脚期割合: {param.stance_percentage:.1f} %")
        print(f"  ケイデンス: {param.cadence:.1f} steps/min")
    
    # 変動性
    variability = calculator.calculate_variability(parameters)
    print(f"\n=== 変動性 ===")
    print(f"ストライド時間CV: {variability.get('stride_time_cv', 0):.2f} %")
    
    # サマリーレポート
    summary = calculator.generate_summary_report(parameters, variability=variability)
    print(f"\n=== サマリーレポート ===")
    print(summary.to_string(index=False))
