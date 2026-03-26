"""
生成测试结果图像
用于生成各种滤波器的结果图像，以便插入到实验报告中
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 确保temp目录存在
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

class ImageFilterTester:
    def __init__(self):
        self.image = None
        self.gray_image = None
    
    def load_image(self, image_path):
        """加载图像"""
        try:
            # 读取图像
            self.image = cv2.imread(image_path)
            if self.image is None:
                # 创建一个测试图像
                self.image = np.zeros((300, 400, 3), dtype=np.uint8)
                # 添加红色区域
                self.image[50:150, 50:150] = [0, 0, 255]  # BGR格式
                # 添加绿色区域
                self.image[50:150, 250:350] = [0, 255, 0]
                # 添加蓝色区域
                self.image[150:250, 50:150] = [255, 0, 0]
                # 添加黄色区域
                self.image[150:250, 250:350] = [0, 255, 255]
                # 添加渐变区域
                for i in range(100):
                    self.image[200+i, 100:300] = [100, 255-i*2, i*2]
                cv2.imwrite("test_image.jpg", self.image)
                image_path = "test_image.jpg"
                print(f"创建测试图像: {image_path}")
            
            # 转换为RGB格式
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            # 转换为灰度图
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            
            print(f"成功加载图像: {image_path}")
            print(f"图像尺寸: {self.image.shape}")
            return True
        except Exception as e:
            print(f"加载图像失败: {e}")
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
    
    def save_image(self, image, filename):
        """保存图像"""
        try:
            filepath = os.path.join(temp_dir, filename)
            if len(image.shape) == 2:
                # 灰度图
                cv2.imwrite(filepath, image)
            else:
                # RGB图，转换为BGR格式保存
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, image_bgr)
            print(f"保存图像: {filepath}")
            return filepath
        except Exception as e:
            print(f"保存图像失败: {e}")
            return None
    
    def visualize_and_save(self, images, titles, save_path):
        """可视化并保存图像"""
        try:
            n = len(images)
            rows = 1
            cols = n
            if n > 3:
                rows = (n + 2) // 3
                cols = 3
            
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            fig.suptitle('滤波器效果对比', fontsize=16, fontweight='bold')
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 显示图像
            for i, (image, title) in enumerate(zip(images, titles)):
                if rows == 1:
                    ax = axes[i]
                else:
                    ax = axes[i // cols, i % cols]
                
                if len(image.shape) == 2:
                    ax.imshow(image, cmap='gray')
                else:
                    ax.imshow(image)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.axis('off')
            
            # 隐藏多余的子图
            for i in range(len(images), rows * cols):
                if rows == 1:
                    axes[i].set_visible(False)
                else:
                    axes[i // cols, i % cols].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"保存对比图: {save_path}")
            return save_path
        except Exception as e:
            print(f"可视化失败: {e}")
            return None

def main():
    print("=" * 60)
    print("生成测试结果图像")
    print("=" * 60)
    print()
    
    # 创建测试器实例
    tester = ImageFilterTester()
    
    # 加载图像
    tester.load_image("test_image.jpg")
    
    print()
    
    # 1. 空间域滤波测试
    print("=== 空间域滤波测试 ===")
    
    # 方框滤波
    box_filtered = tester.box_filter(5)
    tester.save_image(box_filtered, "box_filter_result.png")
    
    # 高斯滤波
    gaussian_filtered = tester.gaussian_filter(5, 2.0)
    tester.save_image(gaussian_filtered, "gaussian_filter_result.png")
    
    # 中值滤波
    median_filtered = tester.median_filter(5)
    tester.save_image(median_filtered, "median_filter_result.png")
    
    # Sobel边缘检测
    sobel_x = tester.sobel_edge_detection('x')
    tester.save_image(sobel_x, "sobel_x_result.png")
    
    sobel_y = tester.sobel_edge_detection('y')
    tester.save_image(sobel_y, "sobel_y_result.png")
    
    sobel_combined = tester.sobel_edge_detection('combined')
    tester.save_image(sobel_combined, "sobel_combined_result.png")
    
    # 空间域滤波对比
    spatial_images = [
        tester.image,
        box_filtered,
        gaussian_filtered,
        median_filtered,
        sobel_combined
    ]
    spatial_titles = [
        '原图',
        '方框滤波 (5x5)',
        '高斯滤波 (5x5, σ=2.0)',
        '中值滤波 (5x5)',
        'Sobel边缘检测'
    ]
    tester.visualize_and_save(spatial_images, spatial_titles, "spatial_filter_comparison.png")
    
    print()
    
    # 2. 频率域滤波测试
    print("=== 频率域滤波测试 ===")
    
    # 理想低通
    ideal_low_result = tester.apply_freq_filter('ideal_low', 30)
    if ideal_low_result:
        tester.save_image(ideal_low_result['filtered'], "ideal_low_result.png")
    
    # 理想高通
    ideal_high_result = tester.apply_freq_filter('ideal_high', 30)
    if ideal_high_result:
        tester.save_image(ideal_high_result['filtered'], "ideal_high_result.png")
    
    # 高斯低通
    gaussian_low_result = tester.apply_freq_filter('gaussian_low', 30)
    if gaussian_low_result:
        tester.save_image(gaussian_low_result['filtered'], "gaussian_low_result.png")
    
    # 高斯高通
    gaussian_high_result = tester.apply_freq_filter('gaussian_high', 30)
    if gaussian_high_result:
        tester.save_image(gaussian_high_result['filtered'], "gaussian_high_result.png")
    
    # 巴特沃斯低通
    butterworth_low_result = tester.apply_freq_filter('butterworth_low', 30, 2)
    if butterworth_low_result:
        tester.save_image(butterworth_low_result['filtered'], "butterworth_low_result.png")
    
    # 巴特沃斯高通
    butterworth_high_result = tester.apply_freq_filter('butterworth_high', 30, 2)
    if butterworth_high_result:
        tester.save_image(butterworth_high_result['filtered'], "butterworth_high_result.png")
    
    # 频率域滤波对比
    freq_images = [
        tester.image,
        ideal_low_result['filtered'] if ideal_low_result else tester.image,
        ideal_high_result['filtered'] if ideal_high_result else tester.image,
        gaussian_low_result['filtered'] if gaussian_low_result else tester.image,
        gaussian_high_result['filtered'] if gaussian_high_result else tester.image,
        butterworth_low_result['filtered'] if butterworth_low_result else tester.image,
        butterworth_high_result['filtered'] if butterworth_high_result else tester.image
    ]
    freq_titles = [
        '原图',
        '理想低通 (截止频率=30)',
        '理想高通 (截止频率=30)',
        '高斯低通 (截止频率=30)',
        '高斯高通 (截止频率=30)',
        '巴特沃斯低通 (截止频率=30, 阶数=2)',
        '巴特沃斯高通 (截止频率=30, 阶数=2)'
    ]
    tester.visualize_and_save(freq_images, freq_titles, "freq_filter_comparison.png")
    
    # 3. 傅里叶变换结果
    print("=== 傅里叶变换测试 ===")
    fft_result = tester.fft_transform(tester.image)
    if fft_result:
        # 保存频谱图
        magnitude_normalized = cv2.normalize(fft_result['magnitude'], None, 0, 255, cv2.NORM_MINMAX)
        tester.save_image(magnitude_normalized, "fft_magnitude.png")
        
        phase_normalized = cv2.normalize(fft_result['phase'], None, 0, 255, cv2.NORM_MINMAX)
        tester.save_image(phase_normalized, "fft_phase.png")
        
        # 显示傅里叶变换结果
        fft_images = [
            tester.image,
            magnitude_normalized,
            phase_normalized
        ]
        fft_titles = [
            '原图',
            '频谱幅值（对数增强）',
            '相位谱'
        ]
        tester.visualize_and_save(fft_images, fft_titles, "fft_results.png")
    
    print()
    print("=" * 60)
    print("测试结果生成完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
