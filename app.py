"""
交互式图像滤波应用
功能：空间域滤波、图像梯度、频率域滤波
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from PIL import Image
import os

# 确保temp目录存在
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

class ImageFilteringApp:
    def __init__(self):
        self.image = None
        self.gray_image = None
        self.selected_region = None
        self.filtered_images = {}
    
    def load_image(self, uploaded_file):
        """加载图像"""
        try:
            # 读取图像
            img = Image.open(uploaded_file)
            self.image = np.array(img)
            
            # 转换为灰度图
            if len(self.image.shape) == 3:
                self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            else:
                self.gray_image = self.image.copy()
            
            return True
        except Exception as e:
            st.error(f"加载图像失败: {e}")
            return False
    
    # 空间域滤波
    def box_filter(self, kernel_size=3):
        """方框滤波"""
        if self.image is None:
            return None
        return cv2.boxFilter(self.image, -1, (kernel_size, kernel_size))
    
    def gaussian_filter(self, kernel_size=3, sigma=1.0):
        """高斯滤波"""
        if self.image is None:
            return None
        return cv2.GaussianBlur(self.image, (kernel_size, kernel_size), sigma)
    
    def median_filter(self, kernel_size=3):
        """中值滤波"""
        if self.image is None:
            return None
        return cv2.medianBlur(self.image, kernel_size)
    
    def sobel_edge_detection(self, direction='combined'):
        """Sobel边缘检测"""
        if self.gray_image is None:
            return None
        
        # 计算X和Y方向的梯度
        sobel_x = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
        
        if direction == 'x':
            return np.uint8(np.clip(np.abs(sobel_x), 0, 255))
        elif direction == 'y':
            return np.uint8(np.clip(np.abs(sobel_y), 0, 255))
        else:
            return sobel_combined
    
    # 图像梯度计算
    def calculate_gradient(self, region):
        """计算指定区域的梯度"""
        if self.gray_image is None or region is None:
            return None
        
        x1, y1, x2, y2 = region
        region_roi = self.gray_image[y1:y2, x1:x2]
        
        # 计算梯度
        grad_x = cv2.Sobel(region_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(region_roi, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值和方向
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)  # 弧度
        grad_direction_deg = np.degrees(grad_direction)  # 角度
        
        return {
            'magnitude': grad_magnitude,
            'direction': grad_direction,
            'direction_deg': grad_direction_deg,
            'region': region_roi
        }
    
    # 频率域滤波
    def fft_transform(self, image):
        """傅里叶变换"""
        if image is None:
            return None
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 傅里叶变换
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        phase_spectrum = np.angle(fshift)
        
        return {
            'fshift': fshift,
            'magnitude': magnitude_spectrum,
            'phase': phase_spectrum
        }
    
    def ifft_transform(self, fshift):
        """逆傅里叶变换"""
        if fshift is None:
            return None
        
        # 逆傅里叶变换
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = np.uint8(img_back)
        
        return img_back
    
    def ideal_filter(self, shape, cutoff, type='low'):
        """理想滤波器"""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        mask = np.zeros((rows, cols), np.uint8)
        if type == 'low':
            mask = np.zeros((rows, cols), np.uint8)
            for i in range(rows):
                for j in range(cols):
                    if np.sqrt((i - crow)**2 + (j - ccol)**2) <= cutoff:
                        mask[i, j] = 1
        else:  # high
            mask = np.ones((rows, cols), np.uint8)
            for i in range(rows):
                for j in range(cols):
                    if np.sqrt((i - crow)**2 + (j - ccol)**2) <= cutoff:
                        mask[i, j] = 0
        
        return mask
    
    def gaussian_filter_fft(self, shape, cutoff, type='low'):
        """高斯滤波器"""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
                if type == 'low':
                    mask[i, j] = np.exp(-(distance**2) / (2 * (cutoff**2)))
                else:  # high
                    mask[i, j] = 1 - np.exp(-(distance**2) / (2 * (cutoff**2)))
        
        return mask
    
    def butterworth_filter(self, shape, cutoff, order=2, type='low'):
        """巴特沃斯滤波器"""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
                if distance == 0:
                    mask[i, j] = 0 if type == 'high' else 1
                else:
                    if type == 'low':
                        mask[i, j] = 1 / (1 + (distance / cutoff)**(2 * order))
                    else:  # high
                        mask[i, j] = 1 / (1 + (cutoff / distance)**(2 * order))
        
        return mask
    
    def apply_freq_filter(self, filter_type, cutoff, order=2):
        """应用频域滤波器"""
        if self.image is None:
            return None
        
        # 傅里叶变换
        fft_result = self.fft_transform(self.image)
        if fft_result is None:
            return None
        
        # 生成滤波器
        if filter_type.startswith('ideal'):
            filter_type = filter_type.replace('ideal_', '')
            mask = self.ideal_filter(fft_result['fshift'].shape, cutoff, filter_type)
        elif filter_type.startswith('gaussian'):
            filter_type = filter_type.replace('gaussian_', '')
            mask = self.gaussian_filter_fft(fft_result['fshift'].shape, cutoff, filter_type)
        elif filter_type.startswith('butterworth'):
            filter_type = filter_type.replace('butterworth_', '')
            mask = self.butterworth_filter(fft_result['fshift'].shape, cutoff, order, filter_type)
        else:
            return None
        
        # 应用滤波器
        fshift_filtered = fft_result['fshift'] * mask
        
        # 逆傅里叶变换
        img_back = self.ifft_transform(fshift_filtered)
        
        # 转换为RGB格式
        if len(self.image.shape) == 3:
            img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB)
        
        return {
            'filtered': img_back,
            'mask': mask,
            'fft_result': fft_result
        }
    
    def save_result(self, image, filename):
        """保存结果图像"""
        try:
            filepath = os.path.join(temp_dir, filename)
            if len(image.shape) == 2:
                img = Image.fromarray(image, mode='L')
            else:
                img = Image.fromarray(image)
            img.save(filepath)
            return filepath
        except Exception as e:
            st.error(f"保存图像失败: {e}")
            return None

def main():
    st.set_page_config(
        page_title="交互式图像滤波应用",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("交互式图像滤波应用")
    st.write("实现空间域滤波、图像梯度、频率域滤波三大核心功能")
    
    # 创建应用实例
    app = ImageFilteringApp()
    
    # 侧边栏 - 上传图像
    st.sidebar.title("操作面板")
    uploaded_file = st.sidebar.file_uploader("上传图像", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # 加载图像
        if app.load_image(uploaded_file):
            st.success("图像加载成功！")
            
            # 显示原始图像
            st.subheader("原始图像")
            col1, col2 = st.columns(2)
            with col1:
                st.image(app.image, caption="原始彩色图像", use_column_width=True)
            with col2:
                if app.gray_image is not None:
                    st.image(app.gray_image, caption="灰度图像", use_column_width=True)
            
            # 功能选择
            tab1, tab2, tab3 = st.tabs(["空间域滤波", "图像梯度", "频率域滤波"])
            
            # 空间域滤波
            with tab1:
                st.subheader("空间域滤波")
                
                # 滤波器选择
                filter_type = st.selectbox(
                    "选择滤波器",
                    ["方框滤波", "高斯滤波", "中值滤波", "Sobel边缘检测"]
                )
                
                # 滤波器参数
                if filter_type == "方框滤波":
                    kernel_size = st.slider("核大小", 3, 11, 3, step=2)
                    filtered = app.box_filter(kernel_size)
                
                elif filter_type == "高斯滤波":
                    kernel_size = st.slider("核大小", 3, 11, 3, step=2)
                    sigma = st.slider("σ 值", 0.1, 5.0, 1.0, step=0.1)
                    filtered = app.gaussian_filter(kernel_size, sigma)
                
                elif filter_type == "中值滤波":
                    kernel_size = st.slider("核大小", 3, 11, 3, step=2)
                    filtered = app.median_filter(kernel_size)
                
                elif filter_type == "Sobel边缘检测":
                    direction = st.selectbox("边缘方向", ["X方向", "Y方向", "融合"])
                    dir_map = {"X方向": "x", "Y方向": "y", "融合": "combined"}
                    filtered = app.sobel_edge_detection(dir_map[direction])
                
                # 显示滤波结果
                if filtered is not None:
                    st.subheader("滤波结果")
                    st.image(filtered, caption=f"{filter_type}结果", use_column_width=True)
                    
                    # 保存结果
                    if st.button("保存结果"):
                        filename = f"{filter_type.replace(' ', '_')}_result.png"
                        filepath = app.save_result(filtered, filename)
                        if filepath:
                            st.success(f"结果已保存到: {filepath}")
            
            # 图像梯度
            with tab2:
                st.subheader("图像梯度")
                
                # 区域选择
                st.write("请在灰度图像上选择一个区域")
                
                # 显示灰度图像供选择
                if app.gray_image is not None:
                    # 使用Streamlit的图像标注功能
                    import streamlit.components.v1 as components
                    
                    # 简单的区域选择界面
                    st.write("提示：在下方输入区域坐标")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        x1 = st.number_input("X1", 0, app.gray_image.shape[1]-1, 0)
                    with col2:
                        y1 = st.number_input("Y1", 0, app.gray_image.shape[0]-1, 0)
                    with col3:
                        x2 = st.number_input("X2", x1+1, app.gray_image.shape[1], app.gray_image.shape[1]//2)
                    with col4:
                        y2 = st.number_input("Y2", y1+1, app.gray_image.shape[0], app.gray_image.shape[0]//2)
                    
                    region = (int(x1), int(y1), int(x2), int(y2))
                    
                    if st.button("计算梯度"):
                        gradient_result = app.calculate_gradient(region)
                        if gradient_result is not None:
                            st.success("梯度计算完成！")
                            
                            # 显示区域
                            st.subheader("选择的区域")
                            st.image(gradient_result['region'], caption="选择的区域", use_column_width=True)
                            
                            # 显示梯度幅值
                            st.subheader("梯度幅值")
                            magnitude_normalized = cv2.normalize(gradient_result['magnitude'], None, 0, 255, cv2.NORM_MINMAX)
                            st.image(magnitude_normalized, caption="梯度幅值", use_column_width=True)
                            
                            # 显示梯度方向
                            st.subheader("梯度方向")
                            # 简单的方向可视化
                            direction_normalized = cv2.normalize(gradient_result['direction_deg'], None, 0, 255, cv2.NORM_MINMAX)
                            st.image(direction_normalized, caption="梯度方向（角度）", use_column_width=True)
                            
                            # 显示统计信息
                            st.subheader("梯度统计信息")
                            st.write(f"区域大小: {gradient_result['region'].shape}")
                            st.write(f"平均梯度幅值: {np.mean(gradient_result['magnitude']):.2f}")
                            st.write(f"最大梯度幅值: {np.max(gradient_result['magnitude']):.2f}")
                            st.write(f"平均梯度方向: {np.mean(gradient_result['direction_deg']):.2f}°")
            
            # 频率域滤波
            with tab3:
                st.subheader("频率域滤波")
                
                # 傅里叶变换
                if app.image is not None:
                    fft_result = app.fft_transform(app.image)
                    if fft_result is not None:
                        st.subheader("傅里叶变换结果")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(fft_result['magnitude'], caption="频谱幅值（对数增强）", use_column_width=True)
                        with col2:
                            phase_normalized = cv2.normalize(fft_result['phase'], None, 0, 255, cv2.NORM_MINMAX)
                            st.image(phase_normalized, caption="相位谱", use_column_width=True)
                    
                # 滤波器选择
                filter_type = st.selectbox(
                    "选择频域滤波器",
                    ["理想低通", "理想高通", "高斯低通", "高斯高通", "巴特沃斯低通", "巴特沃斯高通"]
                )
                
                # 滤波器参数
                cutoff = st.slider("截止频率", 10, 100, 30)
                order = 2
                if "巴特沃斯" in filter_type:
                    order = st.slider("阶数", 1, 5, 2)
                
                # 应用滤波器
                if st.button("应用滤波器"):
                    filter_map = {
                        "理想低通": "ideal_low",
                        "理想高通": "ideal_high",
                        "高斯低通": "gaussian_low",
                        "高斯高通": "gaussian_high",
                        "巴特沃斯低通": "butterworth_low",
                        "巴特沃斯高通": "butterworth_high"
                    }
                    
                    result = app.apply_freq_filter(filter_map[filter_type], cutoff, order)
                    if result is not None:
                        st.success("滤波完成！")
                        
                        # 显示结果
                        st.subheader("滤波结果")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(result['filtered'], caption="滤波后图像", use_column_width=True)
                        with col2:
                            mask_normalized = cv2.normalize(result['mask'], None, 0, 255, cv2.NORM_MINMAX)
                            st.image(mask_normalized, caption="滤波器掩码", use_column_width=True)
                        
                        # 保存结果
                        if st.button("保存频域滤波结果"):
                            filename = f"{filter_type.replace(' ', '_')}_result.png"
                            filepath = app.save_result(result['filtered'], filename)
                            if filepath:
                                st.success(f"结果已保存到: {filepath}")
    
    # 运行说明
    st.sidebar.markdown("""
    ## 运行说明
    1. 上传图像文件（支持jpg、jpeg、png、bmp格式）
    2. 在不同标签页中选择功能：
       - 空间域滤波：选择滤波器类型和参数
       - 图像梯度：输入区域坐标计算梯度
       - 频率域滤波：选择频域滤波器和参数
    3. 查看结果并保存
    """)

if __name__ == "__main__":
    main()
