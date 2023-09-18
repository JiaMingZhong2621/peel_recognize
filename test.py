import cv2
import os

# 定义输入图像的路径和输出路径
input_folder = 'chip_orignal/'  # 输入图像所在文件夹
output_folder = 'chip_crop/'  # 裁剪后图像的输出文件夹

# 定义裁剪参数
crop_size = (416, 416)  # YOLO网络的输入大小
overlap = 0.25  # 重叠的比例

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有图像文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # 读取图像
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path)

    # 获取图像的宽度和高度
    height, width, _ = img.shape

    # 计算重叠的像素数量
    overlap_pixels = 208

    # 循环裁剪图像
    for y in range(0, height, crop_size[1] - overlap_pixels):
        for x in range(0, width, crop_size[0] - overlap_pixels):
            # 计算裁剪区域的坐标
            x1 = x
            y1 = y
            x2 = min(x + crop_size[0], width)
            y2 = min(y + crop_size[1], height)

            # 裁剪图像
            cropped_img = img[y1:y2, x1:x2]
            if cropped_img.shape[0]<416 or cropped_img.shape[1]<416:
                continue
            # 生成输出文件名（可以根据需要进行修改）
            output_file = os.path.splitext(image_file)[0] + f'_crop_{x1}_{y1}.jpg'
            output_path = os.path.join(output_folder, output_file)

            # 保存裁剪后的图像
            cv2.imwrite(output_path, cropped_img)

print("裁剪完成。")
