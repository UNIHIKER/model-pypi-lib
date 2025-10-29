# tinyml_inference.py
import time
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Optional   # ADDED
from .preprocess import preprocess, DataItem, DataSettings, PreprocessSettings, FlattenSettings, AnalysisSettings, FilterSettings
import yaml
from collections import deque

class RingBufferTrigger:
    def __init__(self,
                 use_data_dot: int,
                 model_input_axes_count: int,
                 step_size: int):
        self.USE_DATA_DOT = use_data_dot
        self.MODEL_INPUT_AXES_COUNT = model_input_axes_count
        self.step_size = step_size

        self.buf_max = self.USE_DATA_DOT * self.MODEL_INPUT_AXES_COUNT

        self.buffer: deque[float] = deque(maxlen=self.buf_max)

        self.buffer_full = False
        self.step_counter = 0  

    def add_xyz_str(self, value: str) -> bool:
        if not value.strip():
            return False
        try:
            parts = value.split(',')
            if len(parts) != self.MODEL_INPUT_AXES_COUNT:
                return False
            floats = [float(p) for p in parts]
        except ValueError:
            return False

        self.buffer.extend(floats)

        if not self.buffer_full and len(self.buffer) == self.buf_max:
            self.buffer_full = True

        self.step_counter += 1
        return True

    # ------------- 触发条件 -------------
    def has_enough(self) -> bool:
        return self.buffer_full and self.step_counter >= self.step_size

    # ------------- 获取当前窗口 -------------
    def get_buffer(self) -> np.ndarray:
        return np.array(self.buffer, dtype=np.float32)

    # ------------- 滑动窗口 -------------
    def slide(self):
        discard_len = self.step_size * self.MODEL_INPUT_AXES_COUNT
        if len(self.buffer) >= discard_len:
            for _ in range(discard_len):
                self.buffer.popleft()
        self.step_counter = max(0, self.step_counter - self.step_size)

        if len(self.buffer) < self.buf_max:
            self.buffer_full = False

class TinyMLInference:
    def __init__(self,
                 onnx_model_path: str,
                 onnx_yaml_path: str,
                 step_size: int = 4):         
        self.onnx_model_path = onnx_model_path
        self.onnx_yaml_path = onnx_yaml_path

        self.mean = None
        self.scale = None
        self._load_yaml_config()

        self.session = ort.InferenceSession(self.onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.buffer = RingBufferTrigger(
            use_data_dot=self.data_settings.use_data_dot,
            model_input_axes_count=len(self.data_settings.input_axes),
            step_size=step_size
        )


    def _load_yaml_config(self):
        with open(self.onnx_yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.model_name = config.get('base_model', None)
        data_settings = config.get('data_settings', {})
        flatten_cfg = config.get('preprocess_setting_default', {}).get('Flatten', {})
        analysis_cfg = config.get('preprocess_setting_default', {}).get('Analysis', {})
        filter_cfg = config.get('preprocess_setting_default', {}).get('Filter', {})
        scaler_params = config.get('scaler_params', {})

        self.data_settings = DataSettings(
            input_axes=data_settings.get('input_axes', []),
            output_class=data_settings.get('output_class', []),
            use_data_dot=data_settings.get('use_data_dot'),
            time_interval=data_settings.get('time_interval')
        )

        flatten_set = FlattenSettings(**flatten_cfg)
        analysis_set = AnalysisSettings(**analysis_cfg)
        filter_set = FilterSettings(**filter_cfg)
        self.preprocess_settings = PreprocessSettings(
            Flatten=flatten_set,
            Analysis=analysis_set,
            Filter=filter_set
        )
        self.mean = np.array(scaler_params.get('mean', []))
        self.scale = np.array(scaler_params.get('scale', []))


    def inference(self, xyz_str: str) -> Dict:
        if not self.buffer.add_xyz_str(xyz_str):
            return {"error": "输入数据格式错误，每个数值之间用逗号分隔"}

        raw = self.buffer.get_buffer()  

        if not self.buffer.has_enough():
            return {"status": "waiting data"}

        result = self._inference(raw)

        self.buffer.slide()

        return result

    def one_data_preprocess(self, data):
        data = [DataItem(data=data, label="unknown")]
        processed = preprocess(data, self.data_settings, self.preprocess_settings)
        X = np.array([d.data for d in processed], dtype=np.float32)

        if self.mean is None or self.scale is None:
            print("警告：scaler_params中缺少'mean'或'scale'字段，跳过标准化")
        else:
            self.mean = np.array(self.mean)
            self.scale = np.array(self.scale)
            if self.mean.shape[0] != X.shape[1] or self.scale.shape[0] != X.shape[1]:
                raise ValueError("标准化参数维度与特征维度不匹配")
            if np.any(self.scale == 0):
                self.scale = np.where(self.scale == 0, 1e-8, self.scale)
            X = (X - self.mean) / self.scale
            X = X.astype(np.float32)
        return X

    def _inference(self, input_data):

        X = self.one_data_preprocess(input_data)
        outputs = self.session.run([self.output_name], {self.input_name: X})
        logits = outputs[0][0]

        probs = self._softmax(logits)
        class_labels = self.data_settings.output_class
        self.class_names = {i: class_labels[i] for i in range(len(class_labels))}

        results = []
        for class_id, prob in enumerate(probs):
            class_name = self.class_names.get(class_id, str(class_id))
            results.append({"class_id": class_id,
                            "class_name": class_name,
                            "score": float(prob)})
        results.sort(key=lambda x: x["score"], reverse=True)
        for idx, r in enumerate(results):
            r["result_id"] = idx

        return {"model": self.model_name, "result": results}

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()