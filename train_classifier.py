import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 数据路径（固定）
data_dir = r"E:/pycharm/中草药智能识别/Chinese Medicine"
save_dir = r"E:/pycharm/中草药智能识别/代码/02"

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 保存路径
model_path = os.path.join(save_dir, "model.pth")
label_map_path = os.path.join(save_dir, "label_map.json")

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据增强与预处理
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 数据加载
dataset = datasets.ImageFolder(data_dir, transform=train_tfms)

# 数据划分
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_tfms  # 验证集不用数据增强

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# 保存标签映射
class_names = dataset.classes
print(f"Classes: {class_names}")
with open(label_map_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False)
print(f"标签映射已保存到: {label_map_path}")

# 模型
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# 损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
best_acc = 0.0
best_state = None
epochs = 15

for epoch in range(epochs):
    model.train()
    train_loss, train_correct = 0.0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    # 验证
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
          f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        best_state = model.state_dict()

if best_state is not None:
    with open(model_path, "wb") as f:
        torch.save(best_state, f)
    print(f"最佳模型已保存到: {model_path}")
