# AdaAttN - Export ONNX model

## 介绍

本仓库基于 [AdaAttN](https://github.com/Huage001/AdaAttN/) 进行了一些改动。  
**AdaAttN** 是对任意神经风格迁移中的注意力机制的重新审视，其论文 *"AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer, ICCV 2021"*。

## 主要改动

相比于原仓库，本仓库进行了以下改动：
- **修改 ONNX 不支持的算子**：
  - 将模型中所有 **`torch.randperm`** 的调用改为 **`torch.arange`**，以避免 ONNX 不支持的随机算子。
  - 该改动可能会对推理结果产生轻微的变化。
- **调整 `BaseModel` 的继承**：
  - 原仓库 `BaseModel` 继承 `ABC`，本仓库修改为同时继承 `nn.Module` 和 `ABC` 以优化兼容性。
- **Result HTML 展示内容调整**：
  - 修改了表格格式及标题，使其更易阅读和比较。

## 模型信息

### ONNX 模型下载
- 直接导出的onnx模型[`adaattn.onnx`](https://github.com/whyb/AdaAttN-onnx/blob/main/adaattn.onnx)
- 用onnxslim优化后的onnx模型[`adaattn_slim.onnx`](https://github.com/whyb/AdaAttN-onnx/blob/main/adaattn_slim.onnx)

### ONNX 模型大小
- 导出原始大小**151MB**。
- 经过onnxslim图优化后，**AdaAttN** ONNX 模型文件大小约 **101MB**。

### 模型输入输出
- **输入 1**：`context`，形状 `[b, 3, h, w]`
- **输入 2**：`style`，形状 `[b, 3, h, w]`
- **输出**：`output`，形状 `[b, 3, h, w]`

## 环境要求

- Python **3.10**
- 依赖库安装：
  
  ```bash
  pip install torch-2.5.1+cu124-cp310-cp310-win_amd64.whl
  pip install torchvision==0.20+cu124 -f https://download.pytorch.org/whl/torch_stable.html
  pip install torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  pip install dominate
  pip install numpy==1.26.4
  pip install pillow
  ```

## 运行推理&导出
  ```bash
  python test.py ^
    --content_path datasets/contents ^
    --style_path datasets/styles ^
    --name AdaAttN ^
    --model adaattn ^
    --dataset_mode unaligned ^
    --load_size 1024 ^
    --crop_size 1024 ^
    --image_encoder_path checkpoints/vgg_normalised.pth ^
    --gpu_ids 0 ^
    --skip_connection_3 ^
    --shallow_layer
```

## 效果展示

以下为本仓库风格迁移的部分效果示例：
左1：Context
左2：Style
右1：Result

![Example1](picture/01.png)
![Example2](picture/02.png)
![Example3](picture/03.png)

