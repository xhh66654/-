import os
import json
import torch
from torchvision import transforms, models
from PIL import Image

# 路径
save_dir = r"E:/pycharm/中草药智能识别/代码/02"
model_path = os.path.join(save_dir, "model.pth")
label_map_path = os.path.join(save_dir, "label_map.json")

# 加载标签映射
with open(label_map_path, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

with open(model_path, "rb") as f:
    model.load_state_dict(torch.load(f, map_location=device))
model = model.to(device)
model.eval()

# 预测函数
def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]
