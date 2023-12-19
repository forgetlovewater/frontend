import streamlit as st
from PIL import Image
import io

# 主应用
def main():
    st.title("拖放图片并处理")

    # 添加文件上传部件
    uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 读取上传的图片
        image = Image.open(uploaded_file)

        # 显示图片
        st.image(image, caption="上传的图片", use_column_width=True)

        # 处理上传的图片（在这里你可以添加你的处理逻辑）
        # 例如，显示图片的大小
        image_size = image.size
        st.write(f"图片大小: {image_size}")

if __name__ == "__main__":
    main()
