# Support
如果觉得这个项目有用或对你有启发的话请给一颗⭐️，不胜感激


# 基于 CNN 的手写数字识别

这是我使用 PyTorch 完成的一个手写数字识别小项目，主要目的是练习卷积神经网络（CNN）的基本结构，以及模型训练、测试和图片预测的完整流程。

项目使用 MNIST 数据集进行训练，可以识别 0～9 共 10 个数字，也可以导入自己写的数字图片进行预测。

当前使用的MNIST数据集
训练集：60,000 张图片
测试集：10,000 张图片
图片大小：28×28 像素

注：程序检查项目中的 data/MNIST/ 若不存在会优先从torchvision 0.29.0 下面的镜像下载 MNIST，若主镜像不可用时，会尝试备用地址http://yann.lecun.com/exdb/mnist/ 该项目使用的是第一个 AWS S3 镜像。

## 一、主要功能

- 自动下载和读取 MNIST 数据集；
- 使用 CNN 完成手写数字分类；
- 在每轮训练后计算测试集准确率；
- 自动保存测试准确率最高的模型；
- 绘制训练损失和准确率曲线；
- 对自己准备的手写数字图片进行预测；
- 自动处理白底黑字或黑底白字的图片。

## 二、项目结构

```text
.
├── .venv/                 # Python 虚拟环境
├── artifacts/
│   ├── models/            # 训练得到的模型文件
│   └── plots/             # 样本图和训练曲线
├── data/                  # MNIST 数据集
├── docs/reference/        # 项目整理时使用的参考图片
├── src/mnist_cnn/
│   ├── cli.py             # 命令行功能
│   ├── config.py          # 参数和路径设置
│   ├── data.py            # 数据读取与预处理
│   ├── model.py           # CNN 模型
│   ├── prediction.py      # 图片预测
│   ├── runtime.py         # 设备和随机种子设置
│   ├── training.py        # 模型训练与测试
│   └── visualization.py   # 图片及曲线绘制
├── tests/                 # 简单的程序测试
├── main.py                # 程序入口
├── pyproject.toml         # 项目配置
└── requirements.txt       # 运行所需的第三方库
```

## 三、环境安装

本项目使用 Python 3.12。项目中已经创建了 `.venv` 虚拟环境。

如果需要重新安装依赖，可以在项目根目录打开 PowerShell，然后运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 四、运行方法

### 1. 直接运行完整流程（推荐）

直接运行 `main.py`，不需要添加参数：

```powershell
.\.venv\Scripts\python.exe main.py
```

程序会依次完成以下步骤：

1. 检查是否已经存在训练好的 CNN 模型；
2. 如果存在，询问是直接使用还是重新训练；
3. 如果不存在，自动开始训练模型；
4. 重新训练后保存并显示 MNIST 样本图和训练曲线；
5. 提示输入自己的手写数字图片；
6. 将图片直接拖到终端窗口，按 Enter 后开始预测；
7. 输出预测数字、置信度和其他候选结果。

检测到已有模型时，直接按 Enter 或输入 `y` 会使用已有模型并跳过训练；输入 `n` 会重新训练模型。

显示图片或曲线时，需要先关闭图形窗口，程序才会继续执行下一步。

### 2. 查看帮助

```powershell
.\.venv\Scripts\python.exe main.py --help
```

### 3. 只训练模型

```powershell
.\.venv\Scripts\python.exe main.py train
```

程序第一次运行时会自动下载 MNIST 数据集。训练结束后，效果最好的模型会保存在：

```text
artifacts/models/cnn_model.pth
```

训练曲线会保存在：

```text
artifacts/plots/training_history.png
```

如果只想先训练少量轮数进行测试，可以运行：

```powershell
.\.venv\Scripts\python.exe main.py train --epochs 2
```

### 4. 只预测自己的数字图片

```powershell
.\.venv\Scripts\python.exe main.py predict "C:\你的图片路径\digit.png"
```

程序会输出预测数字、置信度以及概率最高的几个结果。建议图片中只包含一个数字，并尽量保证数字清晰、位于图片中间。

### 5. 查看 MNIST 样本

```powershell
.\.venv\Scripts\python.exe main.py demo --show
```

样本图也会保存到 `artifacts/plots/samples.png`。

### 6. 运行测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

## 五、模型结构

模型一共使用了 3 个卷积层。每个卷积层后通过 ReLU 激活函数提取特征，再使用最大池化减小特征图大小。卷积部分结束后，将特征展开并输入全连接层，最后输出 10 个数字类别的预测结果。

为了减少过拟合，模型中加入了 Dropout。训练时使用交叉熵损失函数和 AdamW 优化器。

## 六、运行结果与分析

测试了手写1，4，7，9等数字，项目完成了自主反色，并分析预测数字，测试结果基本都与手写数字一致，代表其具有一定的识别能力，不过对数字4的置信度较低，可能对于形状特征不明显的数字识别能力还比较差，可能与样本量较小有关，提高样本量理论上会提高识别准确率和预测稳定性。

## 七、目前的不足

这个项目目前主要针对 MNIST 数据集。自己拍摄或手写的图片和 MNIST 图片可能存在差别，因此实际预测效果不一定和测试集准确率完全一致。程序已经加入了自动反色、裁剪和居中处理，但如果图片背景复杂、数字太小或者同时出现多个数字，仍可能预测错误。

后续可以继续尝试数据增强、调整网络结构，或者增加一个更方便的图形界面。
