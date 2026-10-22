# visual pollution 数据集信息总结表
| 项目 | 内容 |
| :--- | :--- |
| 数据集名称 | visual pollution Dataset |
| 任务类型 | 目标检测 (Object Detection) |
| 图片总数 | 约3570 张 |
| 类别数量 | 3 类 |
| 具体类别 | 1. Concrete barriers（混凝土护栏）<br>2. potholes（坑洼）<br>3. sand on road（路面沙土） |
| 模型 | YOLOv8、YOLOv8s、YOLOv5 |
| 数据格式 | 图像 + 标注文件（目标检测标准格式） |
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/430c6f23d1fb46bd9dc7a22546e66aa3.png)



![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5989355d3f0f4023a21cb469886b250d.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7a3fc7391abe4ecbab183e63e120ec88.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1f54537bce2b4768991e53016168b70c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/82ecaf83497f484f85839b773aaaca1a.png)
1![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/570e3e7cd0b245d991acd894ecf1f327.png)
以下是针对 **Visual Pollution Dataset（视觉污染检测数据集）** 的完整 YOLO 训练方案，包含 **3 类目标**：  
1. `Concrete barriers`（混凝土护栏）  
2. `potholes`（坑洼）  
3. `sand on road`（路面沙土）  

支持 **YOLOv5 / YOLOv8 / YOLOv8s**，提供可直接运行的训练代码。

---

## ✅ 一、数据集配置文件 `data.yaml`

```yaml
# data.yaml
path: ./datasets
train: images/train
val: images/val

nc: 3
names: ['concrete_barriers', 'potholes', 'sand_on_road']
```

> 📌 注意：
> - 类别名使用 **下划线命名**（避免空格），与标注文件中的类别 ID 对应
> - 标注格式为标准 YOLO 格式（每行：`class_id x_center y_center width height`，归一化）

---

## ✅ 二、目录结构要求

```
VisualPollutionDetection/
├── datasets/
│   ├── images/
│   │   ├── train/      # ~2856 张（80%）
│   │   └── val/        # ~714 张（20%）
│   └── labels/
│       ├── train/
│       └── val/
├── data.yaml
├── train_yolov8.py     # YOLOv8 训练脚本
└── requirements.txt
```

---

## ✅ 三、YOLOv8 训练代码（推荐）

```python
# train_yolov8.py
from ultralytics import YOLO
import argparse

def main(
    model_name='yolov8s.pt',
    epochs=100,
    imgsz=640,
    batch=16,
    name='visual_pollution_yolo8'
):
    model = YOLO(model_name)

    results = model.train(
        data='data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        cache=False,
        device=0,
        workers=8,
        patience=20,
        save=True,
        exist_ok=False,
        # 增强策略（适合道路场景）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        mosaic=0.8,
        flipud=0.0,
        fliplr=0.5
    )

    print(f"✅ 训练完成！最佳模型保存在: runs/detect/{name}/weights/best.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8s.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--name', type=str, default='visual_pollution_exp')
    args = parser.parse_args()

    main(**vars(args))
```

---

## ✅ 四、YOLOv5 训练代码（兼容旧项目）

> 需先克隆 YOLOv5 仓库：`git clone https://github.com/ultralytics/yolov5`

```python
# train_yolov5.py
import os
import argparse

def main(
    weights='yolov5s.pt',
    epochs=100,
    imgsz=640,
    batch=16,
    name='visual_pollution_yolov5'
):
    cmd = f"""
    python yolov5/train.py \
        --img {imgsz} \
        --batch {batch} \
        --epochs {epochs} \
        --data data.yaml \
        --weights {weights} \
        --name {name} \
        --cache ram \
        --workers 8
    """
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--name', type=str, default='visual_pollution_yolov5')
    args = parser.parse_args()

    main(**vars(args))
```

---

## ✅ 五、依赖安装

```txt
# requirements.txt (YOLOv8)
ultralytics>=8.3.0
opencv-python
numpy
torch>=2.0.0
```

> 若使用 YOLOv5，需额外安装：
```bash
pip install -r yolov5/requirements.txt
```

---

## ✅ 六、启动训练

### YOLOv8（推荐）
```bash
python train_yolov8.py --model yolov8s.pt --epochs 100 --batch 16 --name vp_yolo8s
```

### YOLOv5
```bash
python train_yolov5.py --weights yolov5s.pt --epochs 100 --batch 16 --name vp_yolov5s
```

---

## ✅ 七、推理示例（YOLOv8）

```python
from ultralytics import YOLO

model = YOLO('runs/detect/vp_yolo8s/weights/best.pt')
results = model('road_image.jpg')

# 可视化
results[0].show()

# 获取检测结果
for box in results[0].boxes:
    cls_id = int(box.cls.item())
    label = model.names[cls_id]
    conf = float(box.conf.item())
    print(f"检测到: {label}, 置信度: {conf:.2f}")
```

---

## ✅ 八、应用场景

- 城市道路视觉污染自动巡检  
- 智慧环卫系统（识别路面异常物）  
- 交通养护决策支持  
- 无人机/车载实时监测  

---


