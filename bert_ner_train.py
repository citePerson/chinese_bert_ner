import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import json
import os
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 定义标签映射
label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6
}
id2label = {v: k for k, v in label2id.items()}


def load_data(file_path):
    """加载处理好的数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"加载数据量: {len(data['texts'])}")
    
    # 将字符串标签转换为数字标签
    converted_labels = []
    for labels in data['labels']:
        converted = [label2id[label] for label in labels]
        converted_labels.append(converted)
    
    return data['texts'], converted_labels


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 确保标签长度与文本长度匹配
        if len(label) != len(text):
            print(f"警告：文本长度({len(text)})与标签长度({len(label)})不匹配")
            print(f"文本: {text}")
            print(f"标签: {[id2label[l] for l in label]}")
            # 如果标签长度小于文本长度，用O标签填充
            if len(label) < len(text):
                label = label + [0] * (len(text) - len(label))
            # 如果标签长度大于文本长度，截断标签
            else:
                label = label[:len(text)]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 处理标签
        # 1. 添加特殊token的标签
        label = [0] + label + [0]  # [CLS]和[SEP]的标签为0
        # 2. 填充或截断到max_length
        if len(label) < self.max_len:
            label = label + [0] * (self.max_len - len(label))
        else:
            label = label[:self.max_len]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_model(model, train_loader, val_loader, device, num_epochs=5):
    # 计算类别权重
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].numpy().flatten())
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print("\n类别权重:", class_weights)

    # 使用Focal Loss，增加gamma值以更关注难分类样本
    criterion = FocalLoss(gamma=3)  # 增加gamma值
    # 使用更小的学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)  # 降低学习率
    # 添加学习率调度器，增加耐心值
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    patience = 3  # 早停耐心值
    patience_counter = 0
   
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        batch_count = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            # 将数据移动到GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 使用Focal Loss
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            total_train_loss += loss.item()
            batch_count += 1

            if batch_count % 5 == 0:  # 更频繁地打印损失
                print(f"\nBatch {batch_count}, Loss: {loss.item():.4f}")

            loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                # 将数据移动到GPU
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f'\nEpoch {epoch + 1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')

        # 更新学习率
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存模型时同时保存设备信息
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                'device': device
            }, 'best_ner_model.pt')
            print('Saved best model')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=2)

            # Remove padding tokens and special tokens
            for pred, label, mask in zip(predictions, labels, attention_mask):
                # 只保留非填充和非特殊token的预测结果
                pred = pred[mask == 1]
                label = label[mask == 1]

                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

    # 确保所有标签都被包含在评估中
    unique_labels = sorted(set(all_labels) | set(all_predictions))
    target_names = [id2label[label] for label in unique_labels]

    return all_predictions, all_labels, target_names


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")

    # 加载预训练模型和tokenizer
    model_name = 'models/chinese-bert-wwm-ext'  # 使用中文BERT-wwm
    print(f"加载模型: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(label2id))

    # 将模型移动到设备
    model = model.to(device)

    # 加载数据
    data_dir = "processed_data"
    train_texts, train_labels = load_data(os.path.join(data_dir, 'train.json'))
    val_texts, val_labels = load_data(os.path.join(data_dir, 'val.json'))
    test_texts, test_labels = load_data(os.path.join(data_dir, 'test.json'))

    # 详细的数据检查
    print("\n=== 数据检查 ===")
    print(f"训练集样本数: {len(train_texts)}")
    print(f"验证集样本数: {len(val_texts)}")
    print(f"测试集样本数: {len(test_texts)}")
    
    # 检查文本长度分布
    train_lengths = [len(text) for text in train_texts]
    print(f"\n文本长度统计:")
    print(f"平均长度: {np.mean(train_lengths):.2f}")
    print(f"最大长度: {max(train_lengths)}")
    print(f"最小长度: {min(train_lengths)}")
    
    # 检查标签分布
    print("\n训练集标签分布:")
    train_label_counts = {}
    for labels in train_labels:
        for label_id in labels:
            label = id2label[label_id]
            train_label_counts[label] = train_label_counts.get(label, 0) + 1
    for label, count in train_label_counts.items():
        print(f"{label}: {count}")
    
    # 检查标签对齐
    print("\n标签对齐检查:")
    for i, (text, labels) in enumerate(zip(train_texts[:3], train_labels[:3])):
        print(f"\n样本 {i+1}:")
        print(f"文本: {text}")
        print(f"标签: {[id2label[l] for l in labels]}")
        print(f"长度是否匹配: {len(text) == len(labels)}")

    # 创建数据集
    train_dataset = NERDataset(train_texts, train_labels, tokenizer)
    val_dataset = NERDataset(val_texts, val_labels, tokenizer)
    test_dataset = NERDataset(test_texts, test_labels, tokenizer)

    # 创建数据加载器，使用更小的batch size
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # 减小batch size
    val_loader = DataLoader(val_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)

    # 训练模型
    train_model(model, train_loader, val_loader, device)

    # 评估模型
    predictions, labels, target_names = evaluate_model(model, test_loader, device)

    # 打印评估报告
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=target_names))

    # 保存最终模型和tokenizer
    output_dir = "ner_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(output_dir)
    # 保存tokenizer
    tokenizer.save_pretrained(output_dir)
    # 保存标签映射
    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({
            "label2id": label2id,
            "id2label": id2label
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n模型已保存到: {output_dir}")


if __name__ == "__main__":
    main()
