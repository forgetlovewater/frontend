import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image

# 加载训练好的模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

model = CNN()
model.load_state_dict(torch.load('simple_cnn_model.pth'))
model.eval()

# 定义图像处理函数
def process_image(image):
    image = ToTensor()(image).unsqueeze(0)
    return image

# Streamlit界面
st.title("简单的图像分类应用")

# 上传图像
uploaded_file = st.file_uploader("选择一张图片进行分类", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示上传的图像
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # 预测
    with torch.no_grad():
        processed_image = process_image(image)
        outputs = model(processed_image)
        probabilities = F.softmax(outputs, dim=1)[0]
        _, predicted_class = torch.max(outputs, 1)

    # 显示结果
    st.subheader("模型预测结果:")
    st.write(f"预测类别: {predicted_class.item()}")
    st.write(f"类别概率: {probabilities[predicted_class].item():.4f}")
