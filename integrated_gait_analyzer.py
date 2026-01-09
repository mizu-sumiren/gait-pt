"""
統合歩行分析システム (IntegratedGaitAnalyzer)
Phase 4: Phase 1-3を統合し、実データ（CSV）の読み込みと完全な分析を実行
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Phase 1-3のモジュールをインポート
from gait_event_detector import GaitEventDetector
from gait_parameter_calculator import GaitParameterCalculator, GaitParameters


class IntegratedGaitAnalyzer:
    """
    歩行分析の統合システム
    CSV読み込みから最終レポート生成までを一貫して実行
    """
    
    def __init__(self,
                 fps: float = 30.0,
                 use_z_axis: bool = False,
                 min_visibility: float = 0.5,
                 pixel_to_meter: Optional[float] = None):
        """
        Parameters:
        -----------
        fps : float
            フレームレート [Hz]
        use_z_axis : bool
            3D計算を使用するか
        min_visibility : float
            最小信頼度閾値
        pixel_to_meter : float, optional
            ピクセルからメートルへの変換係数
        """
        self.fps = fps
        self.use_z_axis = use_z_axis
        self.min_visibility = min_visibility
        self.pixel_to_meter = pixel_to_meter
        
        # 各Phaseのコンポーネントを初期化
        self.event_detector = GaitEventDetector(sampling_rate=fps)
        self.param_calculator = GaitParameterCalculator(
            sampling_rate=fps,
            pixel_to_meter=pixel_to_meter
        )
        
        # 分析結果を保存
        self.raw_data = None
        self.processed_data = None
        self.events = None
        self.cycles = None
        self.parameters = None
        self.spatial_params = None
        self.speed_params = None
        self.variability = None
    
    def load_csv_data(self,
                     csv_path: str,
                     landmark_columns: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        CSVデータを読み込む
        
        Parameters:
        -----------
        csv_path : str
            CSVファイルのパス
        landmark_columns : Dict[str, List[str]], optional
            ランドマークごとのカラム名の辞書
            例: {'heel': ['heel_x', 'heel_y', 'heel_z', 'heel_visibility']}
            
        Returns:
        --------
        df : pd.DataFrame
            読み込まれたデータ
        """
        try:
            df = pd.read_csv(csv_path)
            self.raw_data = df
            
            # データの基本情報
            print(f"データ読み込み成功: {len(df)} フレーム")
            print(f"カラム: {list(df.columns)}")
            
            return df
        
        except Exception as e:
            raise ValueError(f"CSVの読み込みに失敗しました: {e}")
    
    def preprocess_data(self,
                       df: pd.DataFrame,
                       heel_cols: Dict[str, str],
                       toe_cols: Optional[Dict[str, str]] = None,
                       smooth: bool = True,
                       window_length: int = 5) -> pd.DataFrame:
        """
        データの前処理（欠損値補完、平滑化）
        
        Parameters:
        -----------
        df : pd.DataFrame
            生データ
        heel_cols : Dict[str, str]
            踵のカラム名 {'x': 'col_name', 'y': 'col_name', ...}
        toe_cols : Dict[str, str], optional
            つま先のカラム名
        smooth : bool
            Savitzky-Golayフィルタを適用するか
        window_length : int
            フィルタの窓長
            
        Returns:
        --------
        processed_df : pd.DataFrame
            前処理後のデータ
        """
        processed_df = df.copy()
        
        # 信頼度による欠損値検出
        if 'visibility' in heel_cols:
            vis_col = heel_cols['visibility']
            if vis_col in processed_df.columns:
                low_conf_mask = processed_df[vis_col] < self.min_visibility
                for coord in ['x', 'y', 'z']:
                    if coord in heel_cols and heel_cols[coord] in processed_df.columns:
                        processed_df.loc[low_conf_mask, heel_cols[coord]] = np.nan
        
        # 平滑化
        if smooth:
            from scipy.signal import savgol_filter
            
            for coord in ['x', 'y', 'z']:
                if coord in heel_cols and heel_cols[coord] in processed_df.columns:
                    col_name = heel_cols[coord]
                    data = processed_df[col_name].values
                    
                    # NaNを補完
                    valid_mask = ~np.isnan(data)
                    if np.sum(valid_mask) > window_length:
                        x_valid = np.where(valid_mask)[0]
                        y_valid = data[valid_mask]
                        x_all = np.arange(len(data))
                        data_interp = np.interp(x_all, x_valid, y_valid)
                        
                        # 平滑化
                        if window_length % 2 == 0:
                            window_length += 1
                        
                        try:
                            data_smooth = savgol_filter(data_interp, window_length, 2)
                            processed_df[col_name] = data_smooth
                        except:
                            processed_df[col_name] = data_interp
                
                # つま先も同様に処理
                if toe_cols and coord in toe_cols and toe_cols[coord] in processed_df.columns:
                    col_name = toe_cols[coord]
                    data = processed_df[col_name].values
                    
                    valid_mask = ~np.isnan(data)
                    if np.sum(valid_mask) > window_length:
                        x_valid = np.where(valid_mask)[0]
                        y_valid = data[valid_mask]
                        x_all = np.arange(len(data))
                        data_interp = np.interp(x_all, x_valid, y_valid)
                        
                        if window_length % 2 == 0:
                            window_length += 1
                        
                        try:
                            data_smooth = savgol_filter(data_interp, window_length, 2)
                            processed_df[col_name] = data_smooth
                        except:
                            processed_df[col_name] = data_interp
        
        self.processed_data = processed_df
        
        return processed_df
    
    def detect_gait_events(self,
                          heel_y: np.ndarray,
                          toe_y: np.ndarray,
                          threshold_percentile: float = 10.0) -> Dict[str, List[int]]:
        """
        歩行イベントを検出（Phase 2）
        
        Parameters:
        -----------
        heel_y : np.ndarray
            踵のY座標
        toe_y : np.ndarray
            つま先のY座標
        threshold_percentile : float
            検出閾値
            
        Returns:
        --------
        events : Dict[str, List[int]]
            検出されたイベント
        """
        self.events = self.event_detector.detect_events(
            heel_y, toe_y, threshold_percentile
        )
        
        self.cycles = self.event_detector.calculate_gait_cycles(self.events)
        
        return self.events
    
    def calculate_parameters(self,
                           heel_x: Optional[np.ndarray] = None,
                           normalize_by: Optional[float] = None) -> List[GaitParameters]:
        """
        歩行パラメータを計算（Phase 3）
        
        Parameters:
        -----------
        heel_x : np.ndarray, optional
            踵のX座標（空間パラメータ計算用）
        normalize_by : float, optional
            正規化用の長さ
            
        Returns:
        --------
        parameters : List[GaitParameters]
            歩行パラメータ
        """
        if self.cycles is None or len(self.cycles) == 0:
            raise ValueError("先にdetect_gait_events()を実行してください")
        
        # 基本パラメータ
        self.parameters = self.param_calculator.calculate_stride_parameters(self.cycles)
        
        # 空間パラメータ
        if heel_x is not None:
            self.spatial_params = self.param_calculator.calculate_spatial_parameters(
                heel_x,
                self.events,
                normalize_by=normalize_by
            )
        
        # 速度パラメータ
        if self.spatial_params and 'stride_lengths' in self.spatial_params:
            stride_times = [p.stride_time for p in self.parameters]
            stride_lengths = self.spatial_params['stride_lengths']
            
            self.speed_params = self.param_calculator.calculate_walking_speed(
                stride_times[:len(stride_lengths)],
                stride_lengths
            )
        
        # 変動性
        self.variability = self.param_calculator.calculate_variability(self.parameters)
        
        return self.parameters
    
    def generate_report(self) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        総合レポートを生成
        
        Returns:
        --------
        report : Dict
            各種レポートを含む辞書
        """
        if self.parameters is None:
            raise ValueError("先にcalculate_parameters()を実行してください")
        
        report = {}
        
        # サマリー統計
        summary_df = self.param_calculator.generate_summary_report(
            self.parameters,
            spatial_params=self.spatial_params,
            speed_params=self.speed_params,
            variability=self.variability
        )
        report['summary'] = summary_df
        
        # イベント情報
        if self.events:
            events_df = self.event_detector.get_events_dataframe(self.events)
            report['events'] = events_df
        
        # 周期詳細
        cycles_detail = []
        for i, (param, cycle) in enumerate(zip(self.parameters, self.cycles)):
            detail = {
                '周期': i + 1,
                '開始フレーム': cycle['start_frame'],
                '終了フレーム': cycle['end_frame'],
                'ストライド時間 (秒)': param.stride_time,
                '立脚期 (秒)': param.stance_time,
                '遊脚期 (秒)': param.swing_time,
                '立脚期割合 (%)': param.stance_percentage,
                'ケイデンス (steps/min)': param.cadence if param.cadence else np.nan
            }
            cycles_detail.append(detail)
        
        report['cycles_detail'] = pd.DataFrame(cycles_detail)
        
        # 統計サマリー
        report['statistics'] = {
            '総フレーム数': len(self.processed_data) if self.processed_data is not None else 0,
            '総時間 (秒)': len(self.processed_data) / self.fps if self.processed_data is not None else 0,
            '検出された踵接地数': len(self.events['heel_strikes']) if self.events else 0,
            '検出された足離地数': len(self.events['toe_offs']) if self.events else 0,
            '完全な歩行周期数': len(self.cycles) if self.cycles else 0,
        }
        
        if self.variability:
            report['statistics']['ストライド時間CV (%)'] = self.variability.get('stride_time_cv', np.nan)
        
        if self.speed_params:
            report['statistics']['平均歩行速度'] = self.speed_params.get('mean_walking_speed', np.nan)
        
        return report
    
    def run_full_analysis(self,
                         csv_path: str,
                         heel_cols: Dict[str, str],
                         toe_cols: Dict[str, str],
                         normalize_by: Optional[float] = None,
                         smooth: bool = True) -> Dict:
        """
        完全な分析を一括実行
        
        Parameters:
        -----------
        csv_path : str
            CSVファイルのパス
        heel_cols : Dict[str, str]
            踵のカラム名
        toe_cols : Dict[str, str]
            つま先のカラム名
        normalize_by : float, optional
            正規化用の長さ
        smooth : bool
            平滑化するか
            
        Returns:
        --------
        report : Dict
            分析レポート
        """
        print("=== 統合歩行分析システム ===")
        print("Step 1: データ読み込み中...")
        df = self.load_csv_data(csv_path)
        
        print("Step 2: データ前処理中...")
        processed_df = self.preprocess_data(df, heel_cols, toe_cols, smooth=smooth)
        
        print("Step 3: 歩行イベント検出中...")
        heel_y = processed_df[heel_cols['y']].values
        toe_y = processed_df[toe_cols['y']].values
        
        events = self.detect_gait_events(heel_y, toe_y)
        
        print(f"  - 踵接地: {len(events['heel_strikes'])}回")
        print(f"  - 足離地: {len(events['toe_offs'])}回")
        print(f"  - 歩行周期: {len(self.cycles)}周期")
        
        print("Step 4: 歩行パラメータ計算中...")
        heel_x = processed_df[heel_cols['x']].values if 'x' in heel_cols else None
        
        parameters = self.calculate_parameters(heel_x, normalize_by=normalize_by)
        
        print("Step 5: レポート生成中...")
        report = self.generate_report()
        
        print("✅ 分析完了！")
        
        return report


# テスト用
if __name__ == "__main__":
    print("IntegratedGaitAnalyzer - テストモード")
    print("実際のCSVデータでテストするには、")
    print("analyzer.run_full_analysis()を使用してください")
