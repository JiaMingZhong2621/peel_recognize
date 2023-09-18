import os
import json
# 定义数据集根目录和存储数据的字典
data_dir = 'data/Aged tangerine peel'
data = {'train': [], 'test': []}

# 处理训练集和测试集
for dataset_type in data.keys():
    dataset_path = os.path.join(data_dir, dataset_type)
    
    # 获取类别列表
    classes = sorted(os.listdir(dataset_path))
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        
        # 获取图像文件列表
        images = os.listdir(class_path)
        
        for image_name in images:
            image_path = os.path.join(class_name, image_name)
            data[dataset_type].append({
                'image_path': image_path,
                'class_idx': class_idx,
                'class_name': class_name
            })

# 保存数据到JSON文件
with open('dataset_info.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
