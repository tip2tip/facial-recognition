# 人脸识别系统

一个基于PyQt5的实时人脸识别系统，使用MTCNN进行人脸检测和FaceNet进行人脸特征提取，支持人脸注册、识别和管理功能。

## 功能特点

- 实时人脸检测和识别
- 支持多人脸同时识别
- 人脸注册和删除功能
- 数据增强技术提高识别准确率
- 美观的图形用户界面
- 自动保存人脸数据
- 实时显示识别信息和置信度

## 系统要求

- Python 3.8 或更高版本
- CUDA支持（可选，用于GPU加速）
- 摄像头

## 安装步骤

1. 克隆项目到本地：
   ```bash
   git clone https://github.com/your-username/facial_recognition.git
   cd facial_recognition
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. 运行程序：
   ```bash
   python gui.py
   ```

2. 注册新人脸：
   - 点击「注册新人脸」按钮
   - 输入人名
   - 保持面部在摄像头画面中，等待系统完成注册

3. 开始识别：
   - 点击「开始识别」按钮
   - 系统会实时显示识别结果和置信度

4. 删除人脸：
   - 在已注册人脸列表中选择要删除的人脸
   - 点击「删除人脸」按钮

## 技术实现

- 使用MTCNN进行人脸检测
- 采用FaceNet（InceptionResnetV1）提取人脸特征
- PyQt5实现图形界面
- 使用数据增强技术提高识别准确率
- 支持GPU加速（如果可用）

## 项目结构

```
├── gui.py          # 图形界面实现
├── main.py         # 核心人脸识别功能
├── requirements.txt # 项目依赖
├── model/          # 预训练模型
└── faces/          # 人脸数据存储
```

## 注意事项

1. 首次运行时会自动下载预训练模型
2. 建议在光线充足的环境下使用
3. 注册人脸时保持面部正对摄像头

## 许可证

MIT License

## 贡献指南

1. Fork 项目
2. 创建新的分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。