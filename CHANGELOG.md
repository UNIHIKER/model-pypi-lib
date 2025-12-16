# model_mp_core
## 0.1.6 2025-12-3
-修改segmentation类输出的segmentation results，新增每个id的area(Pixel areaof the mask(calculated as the sum of mask pixels)), mask coords (list):List of [x,y]coordinates defining the mask contour)
## 0.1.5    2025-12-02
- Tinyml添加重置缓冲区的方法。
## 0.1.4    2025-11-26
- 修改置信度输出位数
## 0.1.3    2025-11-17
- 依赖更新，支持 Tinyml 模型推理。
## 0.1.2    2025-11-11
- 修改了tinyml推理的输出格式。

## 0.1.1    2025-11-10
- 优化tinyml对yaml文件的支持。

## 0.1.0    2025-10-22

- 新增对 Tinyml 模型的支持，增加了对 Tinyml 模型的解析和推理功能。

## 0.0.9     2025-10-20

- 解决图像分类置信度过低问题。

## 0.0.8     2025-09-23

- 优化了模型输入解析流程，防止因分辨率设置导致推理报错。

## 0.0.7     2025-09-19

- 优化了模型输出解析流程，增加了对类别概率为空时的健壮性判断，避免因模型输出异常导致推理报错。
- 代码注释更加详细，便于理解推理流程。

## 0.0.6     2025-09-18

- 新增对 Mind+ 导出 YAML 配置文件的适配，支持自动读取模型结构、类别等关键信息。
- 优化配置文件解析流程，提升兼容性和健壮性。


# model_mp_io
## 0.1.3    2025-12-10

- 修改image_writer中save_masked_image函数，功能为根据检测框中的掩码，抠图并返回img。

## 0.1.2    2025-12-3

- 修改image_writer，新增save_masked_image函数，功能为根据掩码抠图并保存图像。

## 0.1.1    2025-11-26

- 修改image_reader分辨率设置

## 0.1.0    2025-11-11

- 优化txt_reader对输入的格式验证
- 更新txt_reader文本和image_reader视频末尾循环读取

## 0.0.9    2025-11-10

- 增加txt_reader

## 0.0.8    2025-10-09

- 增加yolo11支持

## 0.0.7    2025-09-19

- 修改了视频文件读取结束时的行为，原本返回 None，现改为抛出 RuntimeError("End of video file reached")，便于上层统一异常处理。

- show 方法增加了对 frame 为 None 的判断，防止因空帧导致显示报错。

## 0.0.4    2025-09-18

- 新增对 Mind+ 导出 YAML 配置文件的适配，支持自动读取模型结构、类别等关键信息。
- 优化配置文件解析流程，提升兼容性和健壮性。