from typing import List, Literal

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
    QuantileTransformer, PowerTransformer, Normalizer, 
    KBinsDiscretizer, Binarizer, PolynomialFeatures, SplineTransformer
)
from scipy import signal
from scipy.fft import fft, rfft
from scipy.stats import skew, kurtosis
from pydantic import BaseModel

class DataItem(BaseModel):
    data: List[float]
    label: str

class DataSettings(BaseModel):
    input_axes: List[str]
    """List of input data axis names, e.g., ['x', 'y', 'z'] for accelerometer data"""
    output_class: List[str]
    """List of output classification labels, e.g., ['class1', 'class2']"""
    use_data_dot: int
    """Number of data points to use for model training"""
    time_interval: int
    """Global timing parameter - sampling interval in milliseconds"""

class FlattenSettings(BaseModel):
    enabled: bool
    SimpleImputer: bool
    strategy: Literal["mean", "median", "most_frequent", "constant"] = "constant"
    fill_value: float = 0.0
    StandardScaler: bool
    MinMaxScaler: bool
    MaxAbsScaler: bool
    RobustScaler: bool
    QuantileTransformer: bool
    n_quantiles: int = 100
    PowerTransformer: bool
    Normalizer: bool
    norm: Literal['l1', 'l2', 'max'] = 'l2'
    KBinsDiscretizer: bool
    n_bins: int = 5
    encode: Literal['onehot', 'onehot-dense', 'ordinal'] = 'ordinal'
    Binarizer: bool
    threshold: float = 0.0
    PolynomialFeatures: bool
    degree: Literal[2, 3] = 2
    SplineTransformer: bool
    n_knots: int = 5
    average: bool
    min: bool
    max: bool
    std: bool
    rms: bool
    skew: bool
    kurtosis: bool
    slope: bool
    var: bool
    mean: bool
    median: bool
    ptp: bool

class AnalysisSettings(BaseModel):
    enabled: bool
    stft: bool
    fs: float
    nperseg: int
    noverlap: int
    nfft: int
    fft: bool
    n: int
    rfft: bool

class FilterSettings(BaseModel):
    enabled: bool
    btype: Literal['low', 'high']
    Wn: float
    N: int
    fs: float

class PreprocessSettings(BaseModel):
    Flatten: FlattenSettings
    Analysis: AnalysisSettings
    Filter: FilterSettings
# ================ PREPROCESSING IMPLEMENTATION FUNCTIONS ================

# ---------------- 各功能实现 -----------------

def normalize_axis_length(data: np.ndarray, target_len: int, flatten_cfg: FlattenSettings) -> np.ndarray:
    if len(data) > target_len:
        return data[:target_len]
    elif len(data) < target_len:
        fill = flatten_cfg.fill_value if flatten_cfg.SimpleImputer else 0.0
        return np.pad(data, (0, target_len - len(data)), constant_values=fill)
    return data

def apply_filter_to_axis(data: np.ndarray, fcfg: FilterSettings) -> np.ndarray:
    nyquist = fcfg.fs / 2
    if 0 < fcfg.Wn < nyquist:
        cutoff = fcfg.Wn / nyquist
    else:
        cutoff = fcfg.Wn  

    sos = signal.butter(fcfg.N, cutoff, btype=fcfg.btype, fs=fcfg.fs, output='sos')
    return signal.sosfilt(sos, data)

def apply_analysis_to_axis(data: np.ndarray, acfg: AnalysisSettings, target_length: int) -> np.ndarray:

    if acfg.stft:
        if acfg.nperseg > target_length:
            raise ValueError(f"STFT nperseg ({acfg.nperseg}) cannot be greater than target_length ({target_length})")
        if acfg.noverlap >= acfg.nperseg:
            raise ValueError(f"STFT noverlap ({acfg.noverlap}) must be less than nperseg ({acfg.nperseg})")
        if acfg.nfft < acfg.nperseg:
            raise ValueError(f"STFT nfft ({acfg.nfft}) must be >= nperseg ({acfg.nperseg})")

        _, _, Zxx = signal.stft(data, fs=acfg.fs,
                                nperseg=acfg.nperseg,
                                noverlap=acfg.noverlap,
                                nfft=acfg.nfft)
        return np.abs(Zxx).flatten()

    elif acfg.fft:
        if acfg.n > target_length:
            raise ValueError(f"FFT n ({acfg.n}) cannot be greater than target_length ({target_length})")
        return np.abs(fft(data, n=acfg.n))

    elif acfg.rfft:
        # rfft 的 n 不需要显式指定，但可以限制 data 长度
        return np.abs(rfft(data,32))

    return data

def apply_flatten_to_axis(data: np.ndarray, fcfg: FlattenSettings) -> np.ndarray:
    x = data.reshape(1, -1)

    if fcfg.SimpleImputer:
        x = SimpleImputer(strategy=fcfg.strategy, fill_value=fcfg.fill_value).fit_transform(x)
    
    # 只启用第一个缩放器
    scaler_applied = False
    if not scaler_applied and fcfg.StandardScaler:
        x = StandardScaler().fit_transform(x)
        scaler_applied = True
    if not scaler_applied and fcfg.MinMaxScaler:
        x = MinMaxScaler().fit_transform(x)
        scaler_applied = True
    if not scaler_applied and fcfg.MaxAbsScaler:
        x = MaxAbsScaler().fit_transform(x)
        scaler_applied = True
    if not scaler_applied and fcfg.RobustScaler:
        x = RobustScaler().fit_transform(x)
        scaler_applied = True

    if fcfg.QuantileTransformer:
        x = QuantileTransformer(n_quantiles=fcfg.n_quantiles).fit_transform(x)
    if fcfg.PowerTransformer:
        x = PowerTransformer().fit_transform(x)
    if fcfg.Normalizer:
        x = Normalizer(norm=fcfg.norm).fit_transform(x)
    if fcfg.KBinsDiscretizer:
        x = KBinsDiscretizer(n_bins=fcfg.n_bins, encode=fcfg.encode).fit_transform(x)
    if fcfg.Binarizer:
        x = Binarizer(threshold=fcfg.threshold).fit_transform(x)
    if fcfg.PolynomialFeatures:
        x = PolynomialFeatures(degree=fcfg.degree).fit_transform(x)
    if fcfg.SplineTransformer:
        x = SplineTransformer(degree=fcfg.degree, n_knots=fcfg.n_knots).fit_transform(x)

    x = x.flatten()

    # 判断是否启用任意统计特征
    stats_enabled = any([
        fcfg.average, fcfg.min, fcfg.max, fcfg.std, fcfg.rms,
        fcfg.skew, fcfg.kurtosis, fcfg.slope, fcfg.var,
        fcfg.mean, fcfg.median, fcfg.ptp
    ])

    if stats_enabled:
        feats = []
        if fcfg.average or fcfg.mean: feats.append(np.mean(x))
        if fcfg.min: feats.append(np.min(x))
        if fcfg.max: feats.append(np.max(x))
        if fcfg.std: feats.append(np.std(x))
        if fcfg.rms: feats.append(np.sqrt(np.mean(np.square(x))))
        if fcfg.skew: feats.append(skew(x))
        if fcfg.kurtosis: feats.append(kurtosis(x))
        if fcfg.slope:
            xi = np.arange(len(x))
            feats.append(np.polyfit(xi, x, 1)[0])
        if fcfg.var: feats.append(np.var(x))
        if fcfg.median: feats.append(np.median(x))
        if fcfg.ptp: feats.append(np.ptp(x))
        return np.array(feats)

    return x

# ---------------- 预处理主流程 -----------------

def preprocess(data_list: List[DataItem], 
               data_settings: DataSettings, 
               preprocess_settings: PreprocessSettings) -> List[DataItem]:
    # from utils.logger import app_logger
    # app_logger.info(f"[preprocess] data_settings: {data_settings}")
    # app_logger.info(f"[preprocess] preprocess_settings: {preprocess_settings}")
    
    processed_data = []
    num_axes = len(data_settings.input_axes)
    target_length = data_settings.use_data_dot
    
    for idx, item in enumerate(data_list):
        raw = np.array(item.data, dtype=np.float32)
        
        # 先保证总长度一致（各轴点数 * 轴数）
        expected_len = target_length * num_axes
        raw = normalize_axis_length(raw, expected_len, preprocess_settings.Flatten)

        # 按轴分割数据 shape (num_axes, target_length)
        reshaped = raw.reshape(-1, num_axes).T  
        # print("reshaped:")
        # print(reshaped.shape)
        # print(reshaped[:3])
        
        transformed_axes = []
        for axis_data in reshaped:
            # 确保每轴长度一致
            axis_data = normalize_axis_length(axis_data, target_length, preprocess_settings.Flatten)

            if preprocess_settings.Filter.enabled:
                axis_data = apply_filter_to_axis(axis_data, preprocess_settings.Filter)
                # print("filter axis_data:")
                # print(axis_data.shape)
                # print(axis_data[:3])

            if preprocess_settings.Analysis.enabled:
                axis_data = apply_analysis_to_axis(axis_data, preprocess_settings.Analysis, target_length)
                # print("Analysis axis_data:")
                # print(axis_data.shape)
                # print(axis_data[:3])

            if preprocess_settings.Flatten.enabled:
                axis_data = apply_flatten_to_axis(axis_data, preprocess_settings.Flatten)
                # print("Flatten axis_data:")
                # print(axis_data.shape)
                # print(axis_data[:3])

            transformed_axes.append(axis_data)

        # 多轴结果拼接成一维向量
        final = np.stack(transformed_axes, axis=1).reshape(-1)
        processed_data.append(DataItem(data=final.tolist(), label=item.label))

    return processed_data