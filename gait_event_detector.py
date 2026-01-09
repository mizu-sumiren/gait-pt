"""
歩行イベント検出器 (GaitEventDetector)
Phase 2: 踵接地（Heel Strike）と足離地（Toe Off）の自動検出
"""

import numpy as np
from scipy import signal
from typing import List, Dict, Tuple, Optional
import pandas as pd


class GaitEventDetector:
    """
    歩行イベント（踵接地・足離地）を自動検出するクラス
    """
    
    def __init__(self, 
                 sampling_rate: float = 30.0,
                 min_stride_time: float = 0.5,
                 max_stride_time: float = 2.0):
        """
        Parameters:
        -----------
        sampling_rate : float
            サンプリングレート [Hz]
        min_stride_time : float
            最小ストライド時間 [秒]（これより短い周期は無視）
        max_stride_time : float
            最大ストライド時間 [秒]（これより長い周期は無視）
        """
        self.sampling_rate = sampling_rate
        self.min_stride_time = min_stride_time
        self.max_stride_time = max_stride_time
        
        # 最小・最大ストライド間隔（フレーム数）
        self.min_stride_frames = int(min_stride_time * sampling_rate)
        self.max_stride_frames = int(max_stride_time * sampling_rate)
    
    def detect_heel_strikes(self, 
                           heel_y: np.ndarray,
                           toe_y: np.ndarray,
                           threshold_percentile: float = 10.0) -> List[int]:
        """
        踵接地（Heel Strike）を検出
        
        踵のY座標が極小値（地面に近い）になるポイントを検出
        
        Parameters:
        -----------
        heel_y : np.ndarray
            踵のY座標の時系列データ（上が正、下が負）
        toe_y : np.ndarray
            つま先のY座標の時系列データ
        threshold_percentile : float
            閾値パーセンタイル（下位何%を踵接地の候補とするか）
            
        Returns:
        --------
        heel_strikes : List[int]
            踵接地のフレームインデックスのリスト
        """
        # 欠損値の処理
        valid_mask = ~np.isnan(heel_y)
        if not np.any(valid_mask):
            return []
        
        # 閾値の計算（低い位置にある点を抽出）
        threshold = np.percentile(heel_y[valid_mask], threshold_percentile)
        
        # 極小値の検出
        # distance: 最小ストライド間隔以上離れた極小値のみ検出
        peaks, properties = signal.find_peaks(
            -heel_y,  # 極小値を検出するため符号反転
            distance=self.min_stride_frames,
            prominence=np.nanstd(heel_y) * 0.3  # ノイズを避けるため
        )
        
        # 閾値以下のもののみを選択
        heel_strikes = [p for p in peaks if heel_y[p] <= threshold]
        
        return sorted(heel_strikes)
    
    def detect_toe_offs(self,
                       heel_y: np.ndarray,
                       toe_y: np.ndarray,
                       heel_strikes: List[int],
                       search_window: float = 0.5) -> List[int]:
        """
        足離地（Toe Off）を検出
        
        各踵接地の前に、つま先のY座標が急上昇するポイントを検出
        
        Parameters:
        -----------
        heel_y : np.ndarray
            踵のY座標の時系列データ
        toe_y : np.ndarray
            つま先のY座標の時系列データ
        heel_strikes : List[int]
            踵接地のフレームインデックス
        search_window : float
            探索窓の時間幅 [秒]（踵接地の何秒前まで探すか）
            
        Returns:
        --------
        toe_offs : List[int]
            足離地のフレームインデックスのリスト
        """
        toe_offs = []
        search_frames = int(search_window * self.sampling_rate)
        
        # つま先の速度（Y方向）を計算
        toe_velocity = np.diff(toe_y, prepend=toe_y[0])
        
        for hs in heel_strikes:
            # 探索範囲：踵接地の前
            start_idx = max(0, hs - search_frames)
            end_idx = hs
            
            if start_idx >= end_idx:
                continue
            
            # 探索範囲内でつま先速度が最大（最も上向き）のポイント
            search_region = toe_velocity[start_idx:end_idx]
            
            if len(search_region) == 0 or np.all(np.isnan(search_region)):
                continue
            
            # 極大値を検出
            local_peaks, _ = signal.find_peaks(
                search_region,
                prominence=np.nanstd(toe_velocity) * 0.2
            )
            
            if len(local_peaks) > 0:
                # 最も踵接地に近い極大値を選択
                toe_off_local = local_peaks[-1]
                toe_off = start_idx + toe_off_local
                toe_offs.append(toe_off)
        
        return sorted(toe_offs)
    
    def detect_events(self,
                     heel_y: np.ndarray,
                     toe_y: np.ndarray,
                     threshold_percentile: float = 10.0) -> Dict[str, List[int]]:
        """
        踵接地と足離地を両方検出
        
        Parameters:
        -----------
        heel_y : np.ndarray
            踵のY座標の時系列データ
        toe_y : np.ndarray
            つま先のY座標の時系列データ
        threshold_percentile : float
            踵接地検出の閾値パーセンタイル
            
        Returns:
        --------
        events : Dict[str, List[int]]
            'heel_strikes': 踵接地のリスト
            'toe_offs': 足離地のリスト
        """
        # 踵接地を検出
        heel_strikes = self.detect_heel_strikes(
            heel_y, toe_y, threshold_percentile
        )
        
        # 足離地を検出
        toe_offs = self.detect_toe_offs(
            heel_y, toe_y, heel_strikes
        )
        
        return {
            'heel_strikes': heel_strikes,
            'toe_offs': toe_offs
        }
    
    def calculate_gait_cycles(self,
                             events: Dict[str, List[int]]) -> List[Dict]:
        """
        歩行周期を計算
        
        Parameters:
        -----------
        events : Dict[str, List[int]]
            検出された歩行イベント
            
        Returns:
        --------
        cycles : List[Dict]
            各歩行周期の情報
            - 'start_frame': 開始フレーム（踵接地）
            - 'end_frame': 終了フレーム（次の踵接地）
            - 'toe_off_frame': 足離地のフレーム
            - 'stance_duration': 立脚期の長さ [フレーム]
            - 'swing_duration': 遊脚期の長さ [フレーム]
            - 'stride_duration': ストライドの長さ [フレーム]
        """
        heel_strikes = events['heel_strikes']
        toe_offs = events['toe_offs']
        
        cycles = []
        
        for i in range(len(heel_strikes) - 1):
            start_hs = heel_strikes[i]
            end_hs = heel_strikes[i + 1]
            
            # この周期内の足離地を探す
            toe_off_in_cycle = [to for to in toe_offs if start_hs < to < end_hs]
            
            if len(toe_off_in_cycle) > 0:
                toe_off = toe_off_in_cycle[0]  # 最初の足離地を使用
                
                cycle_info = {
                    'start_frame': start_hs,
                    'end_frame': end_hs,
                    'toe_off_frame': toe_off,
                    'stance_duration': toe_off - start_hs,
                    'swing_duration': end_hs - toe_off,
                    'stride_duration': end_hs - start_hs,
                    'stance_percentage': ((toe_off - start_hs) / (end_hs - start_hs)) * 100
                }
                
                cycles.append(cycle_info)
        
        return cycles
    
    def get_events_dataframe(self,
                            events: Dict[str, List[int]],
                            time_vector: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        検出されたイベントをDataFrameとして返す
        
        Parameters:
        -----------
        events : Dict[str, List[int]]
            検出された歩行イベント
        time_vector : np.ndarray, optional
            時間ベクトル [秒]
            
        Returns:
        --------
        df : pd.DataFrame
            イベントの情報を含むDataFrame
        """
        all_events = []
        
        # 踵接地
        for frame in events['heel_strikes']:
            event_dict = {
                'frame': frame,
                'event_type': 'heel_strike',
                'time': time_vector[frame] if time_vector is not None else frame / self.sampling_rate
            }
            all_events.append(event_dict)
        
        # 足離地
        for frame in events['toe_offs']:
            event_dict = {
                'frame': frame,
                'event_type': 'toe_off',
                'time': time_vector[frame] if time_vector is not None else frame / self.sampling_rate
            }
            all_events.append(event_dict)
        
        df = pd.DataFrame(all_events)
        if len(df) > 0:
            df = df.sort_values('frame').reset_index(drop=True)
        
        return df


# テスト用のシンプルな例
if __name__ == "__main__":
    # テストデータの生成（模擬歩行データ）
    t = np.linspace(0, 10, 300)  # 10秒、30Hz
    
    # 模擬的な踵のY座標（周期的に上下）
    heel_y = -50 + 30 * np.sin(2 * np.pi * 1.0 * t)  # 1Hz（1歩/秒）
    
    # 模擬的なつま先のY座標（踵より少し位相がずれる）
    toe_y = -40 + 25 * np.sin(2 * np.pi * 1.0 * t + np.pi/6)
    
    # 検出器の作成
    detector = GaitEventDetector(sampling_rate=30.0)
    
    # イベント検出
    events = detector.detect_events(heel_y, toe_y)
    
    print("=== 検出された歩行イベント ===")
    print(f"踵接地: {len(events['heel_strikes'])}回")
    print(f"  フレーム: {events['heel_strikes']}")
    print(f"足離地: {len(events['toe_offs'])}回")
    print(f"  フレーム: {events['toe_offs']}")
    
    # 歩行周期の計算
    cycles = detector.calculate_gait_cycles(events)
    print(f"\n=== 歩行周期 ===")
    print(f"検出された周期数: {len(cycles)}")
    
    if len(cycles) > 0:
        print("\n最初の周期の詳細:")
        first_cycle = cycles[0]
        for key, value in first_cycle.items():
            print(f"  {key}: {value}")
