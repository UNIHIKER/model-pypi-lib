from typing import List
from typing import List, Tuple, Union, Literal
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
    """
    A list of floating-point numbers representing the actual data values.
    Can be time series data, sensor readings, or extracted features.
    """
    label: str
    """
    Classification label or category name for supervised learning tasks.
    """
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
    """
    Configuration for Data Flattening and Normalization Operations.
    
    Comprehensive settings for applying various scikit-learn preprocessing
    transformations to normalize and standardize input data. Includes
    imputation, scaling, discretization, and feature engineering options.
    
    Key Features:
    - Missing value imputation strategies
    - Multiple scaling and normalization methods  
    - Discretization and binarization options
    - Polynomial feature generation
    - Statistical feature extraction
    
    Attributes:
        enabled (bool): Master switch to enable/disable flattening operations
        SimpleImputer (bool): Fill missing values using specified strategy
        strategy (Literal): Imputation method - mean, median, most_frequent, constant
        fill_value (float): Value to use when strategy is 'constant'
        StandardScaler (bool): Z-score normalization (mean=0, std=1)
        MinMaxScaler (bool): Scale features to [0,1] range
        MaxAbsScaler (bool): Scale by maximum absolute value
        RobustScaler (bool): Scale using median and IQR (outlier-resistant)
        QuantileTransformer (bool): Transform to uniform distribution
        n_quantiles (int): Number of quantiles for transformation
        PowerTransformer (bool): Apply power transformation for normality
        Normalizer (bool): Normalize samples to unit norm
        norm (Literal): Normalization type - l1, l2, or max
        KBinsDiscretizer (bool): Convert continuous to categorical bins
        n_bins (int): Number of discrete bins to create
        encode (Literal): Encoding method - onehot, onehot-dense, ordinal
        Binarizer (bool): Convert to binary values using threshold
        threshold (float): Binarization threshold value
        PolynomialFeatures (bool): Generate polynomial feature combinations
        degree (Literal): Polynomial degree (2 or 3)
        SplineTransformer (bool): Apply spline basis transformation
        n_knots (int): Number of knots for spline transformation
        average (bool): Calculate mean of data values
        min (bool): Extract minimum value feature
        max (bool): Extract maximum value feature
        std (bool): Calculate standard deviation feature
        rms (bool): Calculate root mean square feature
        skew (bool): Calculate skewness of data distribution
        kurtosis (bool): Calculate kurtosis of data distribution
        slope (bool): Calculate linear trend slope
        var (bool): Calculate variance of data
        mean (bool): Calculate arithmetic mean (same as average)
        median (bool): Calculate median value
        ptp (bool): Calculate peak-to-peak range
    """
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
    """
    Configuration for Signal Analysis and Frequency Domain Processing.
    
    Controls frequency domain transformations including FFT, RFFT, and STFT
    for extracting spectral features from time series data.
    
    Attributes:
        enabled (bool): Master switch to enable/disable analysis operations
        stft (bool): Enable Short-Time Fourier Transform
        fs (float): Sampling frequency in Hz
        nperseg (int): Length of each STFT segment
        noverlap (int): Number of overlapping points between segments
        nfft (int): Length of FFT used for STFT
        fft (bool): Enable standard Fast Fourier Transform
        n (int): Number of points for FFT computation
        rfft (bool): Enable Real-valued FFT (more efficient for real signals)
    """
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
    """
    Configuration for Digital Signal Filtering Operations.
    
    Settings for applying Butterworth filters to remove noise and extract
    specific frequency components from input signals.
    
    Attributes:
        enabled (bool): Master switch to enable/disable filtering
        btype (Literal): Filter type - 'low' for low-pass, 'high' for high-pass
        Wn (float): Critical frequency or cutoff frequency
        N (int): Filter order (higher order = steeper rolloff)
        fs (float): Sampling frequency in Hz
    """
    enabled: bool
    btype: Literal['low', 'high']
    Wn: float
    N: int
    fs: float

class PreprocessSettings(BaseModel):
    """
    Master Configuration Container for All Preprocessing Operations.
    
    Combines all preprocessing configuration objects into a single
    comprehensive settings structure for pipeline management.
    
    Attributes:
        Flatten (FlattenSettings): Data normalization and feature extraction settings
        Analysis (AnalysisSettings): Frequency domain analysis configuration
        Filter (FilterSettings): Digital filtering parameters
    """
    Flatten: FlattenSettings
    Analysis: AnalysisSettings
    Filter: FilterSettings
# ================ PREPROCESSING IMPLEMENTATION FUNCTIONS ================

def normalize_axis_length(data: np.ndarray, target_len: int, flatten_cfg: FlattenSettings) -> np.ndarray:
    """
    Normalize data length to match target requirements.
    
    Adjusts the length of input data arrays by truncation or padding to ensure
    consistent dimensions for model input. Uses configured fill values for padding.
    
    Args:
        data (np.ndarray): Input data array to normalize
        target_len (int): Desired output length
        flatten_cfg (FlattenSettings): Configuration containing fill value settings
        
    Returns:
        np.ndarray: Data array with exactly target_len elements
        
    Note:
        - If data is longer than target: truncates to first target_len elements
        - If data is shorter than target: pads with fill_value or zeros
    """
    if len(data) > target_len:
        return data[:target_len]
    elif len(data) < target_len:
        fill = flatten_cfg.fill_value if flatten_cfg.SimpleImputer else 0.0
        return np.pad(data, (0, target_len - len(data)), constant_values=fill)
    return data

def apply_filter_to_axis(data: np.ndarray, fcfg: FilterSettings) -> np.ndarray:
    """
    Apply digital Butterworth filter to signal data.
    
    Implements low-pass or high-pass filtering using scipy's Butterworth filter
    with Second-Order Sections (SOS) format for numerical stability.
    
    Args:
        data (np.ndarray): Input signal data to filter
        fcfg (FilterSettings): Filter configuration parameters
        
    Returns:
        np.ndarray: Filtered signal data
        
    Note:
        - Automatically handles cutoff frequency normalization
        - Uses SOS format to prevent numerical instability
        - Supports both low-pass and high-pass filtering
    """
    nyquist = fcfg.fs / 2
    if 0 < fcfg.Wn < nyquist:
        cutoff = fcfg.Wn / nyquist
    else:
        cutoff = fcfg.Wn  

    sos = signal.butter(fcfg.N, cutoff, btype=fcfg.btype, fs=fcfg.fs, output='sos')
    return signal.sosfilt(sos, data)

def apply_analysis_to_axis(data: np.ndarray, acfg: AnalysisSettings, target_length: int) -> np.ndarray:
    """
    Apply frequency domain analysis transformations to signal data.
    
    Extracts spectral features using various frequency domain transforms including
    FFT, Real FFT, and Short-Time Fourier Transform (STFT) with comprehensive
    parameter validation.
    
    Args:
        data (np.ndarray): Input time series data
        acfg (AnalysisSettings): Analysis configuration parameters
        target_length (int): Maximum allowed data length for validation
        
    Returns:
        np.ndarray: Transformed data in frequency domain or original data if no transform
        
    Raises:
        ValueError: If STFT parameters are incompatible or FFT size exceeds data length
        
    Transforms:
        - STFT: Time-frequency representation with configurable window parameters
        - FFT: Full complex frequency spectrum (magnitude only)
        - RFFT: Real-valued FFT for computational efficiency
    """

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
        # Real FFT does not require explicit n specification, but data length can be limited
        return np.abs(rfft(data))

    return data

def apply_flatten_to_axis(data: np.ndarray, fcfg: FlattenSettings) -> np.ndarray:
    """
    Apply comprehensive data flattening and feature extraction pipeline.
    
    Processes input data through a configurable sequence of scikit-learn
    transformations including imputation, scaling, discretization, and
    statistical feature extraction. Only applies the first enabled scaler
    to prevent conflicting transformations.
    
    Args:
        data (np.ndarray): Input data array to process
        fcfg (FlattenSettings): Configuration for all flattening operations
        
    Returns:
        np.ndarray: Transformed and flattened feature vector
        
    Processing Pipeline:
        1. Reshape data to 2D format for sklearn compatibility
        2. Apply imputation if enabled
        3. Apply first enabled scaler (prevents conflicts)
        4. Apply discretization/binarization if enabled
        5. Apply feature engineering (polynomial, spline)
        6. Extract statistical features if enabled
        7. Return flattened feature vector
        
    Note:
        - Only the first enabled scaler is applied to prevent conflicts
        - Statistical features are appended to the transformed data
        - All transformations are fit and applied in a single pass
    """
    x = data.reshape(1, -1)

    if fcfg.SimpleImputer:
        x = SimpleImputer(strategy=fcfg.strategy, fill_value=fcfg.fill_value).fit_transform(x)
    
    # Apply only the first enabled scaler to prevent conflicting transformations
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

    # Check if any statistical features are enabled
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

# ================ MAIN PREPROCESSING PIPELINE ================

def preprocess(data_list: List[DataItem], 
               data_settings: DataSettings, 
               preprocess_settings: PreprocessSettings) -> List[DataItem]:
    """
    Main preprocessing pipeline for multi-axis sensor data.
    
    Processes a list of data items through a comprehensive pipeline including
    normalization, filtering, frequency analysis, and feature extraction.
    Handles multi-axis data by processing each axis independently before
    concatenating results.
    
    Args:
        data_list (List[DataItem]): List of raw data items to process
        data_settings (DataSettings): Configuration for data structure and timing
        preprocess_settings (PreprocessSettings): Configuration for all processing steps
        
    Returns:
        List[DataItem]: Processed data items with extracted features
        
    Processing Pipeline:
        1. Normalize data length to match target requirements
        2. Reshape data to separate individual axes
        3. Apply filtering to each axis independently
        4. Apply frequency domain analysis if enabled
        5. Apply flattening and feature extraction if enabled
        6. Concatenate all axes results into final feature vector
        7. Return processed DataItem with same label
        
    Example:
        processed = preprocess(raw_data, data_config, preprocess_config)
        print(f"Processed {len(processed)} samples")
    """
    
    processed_data = []
    num_axes = len(data_settings.input_axes)
    target_length = data_settings.use_data_dot
    
    for idx, item in enumerate(data_list):
        raw = np.array(item.data, dtype=np.float32)
        
        # Ensure total length consistency (data points per axis * number of axes)
        expected_len = target_length * num_axes
        raw = normalize_axis_length(raw, expected_len, preprocess_settings.Flatten)

        # Split data by axis: shape (num_axes, target_length)
        reshaped = raw.reshape(-1, num_axes).T  
        # print("reshaped:")
        # print(reshaped.shape)
        # print(reshaped[:3])
        
        transformed_axes = []
        for axis_data in reshaped:
            # Ensure consistent length for each axis
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

        # Concatenate multi-axis results into one-dimensional vector
        final = np.stack(transformed_axes, axis=1).reshape(-1)
        processed_data.append(DataItem(data=final.tolist(), label=item.label))

    return processed_data