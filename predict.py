import torch
from transformers import BertTokenizer, BertForTokenClassification
import json
from typing import List, Tuple

def load_model(model_path: str) -> Tuple[BertForTokenClassification, BertTokenizer, dict, dict]:
    """加载保存的模型、tokenizer和标签映射"""
    # 加载模型和tokenizer
    model = BertForTokenClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # 加载标签映射
    with open(f"{model_path}/label_mapping.json", "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
        label2id = label_mapping["label2id"]
        # 确保id2label的键是整数类型
        id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
    
    return model, tokenizer, label2id, id2label

def predict_entities(text: str, model: BertForTokenClassification, tokenizer: BertTokenizer, 
                    id2label: dict, device: str = "cpu") -> List[Tuple[str, str]]:
    """预测文本中的实体"""
    # 将模型移到指定设备
    model.to(device)
    model.eval()
    
    # 对输入文本进行编码
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 进行预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # 获取预测的标签
    predicted_labels = predictions[0].cpu().numpy()
    
    # 获取token对应的文本
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    print("\n调试信息:")
    print(f"输入文本: {text}")
    print(f"Token列表: {tokens}")
    print(f"预测标签: {predicted_labels}")
    print(f"标签映射: {id2label}")
    
    # 将预测结果转换为实体列表
    entities = []
    current_entity = None
    current_text = ""
    
    for token, label_id in zip(tokens, predicted_labels):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
            
        try:
            label = id2label[label_id]
            print(f"Token: {token}, Label: {label}")
        except KeyError:
            print(f"警告: 未知的标签ID {label_id}")
            continue
        
        if label.startswith("B-"):
            if current_entity:
                entities.append((current_entity, current_text))
            current_entity = label[2:]
            current_text = token.replace("##", "")
        elif label.startswith("I-"):
            if current_entity and current_entity == label[2:]:
                current_text += token.replace("##", "")
            else:
                if current_entity:
                    entities.append((current_entity, current_text))
                current_entity = None
                current_text = ""
        else:
            if current_entity:
                entities.append((current_entity, current_text))
                current_entity = None
                current_text = ""
    
    # 添加最后一个实体
    if current_entity:
        entities.append((current_entity, current_text))
    
    return entities

def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型和tokenizer
    model_path = "ner_model"
    model, tokenizer, label2id, id2label = load_model(model_path)
    
    # 测试文本
    test_texts = [
        "张三在北京大学学习计算机科学",
        "李四在上海工作",
        "王五在阿里巴巴担任工程师"
    ]
    
    print("\n开始预测...")
    for text in test_texts:
        print(f"\n输入文本: {text}")
        entities = predict_entities(text, model, tokenizer, id2label, device)
        print("预测结果:")
        if entities:
            for entity_type, entity_text in entities:
                print(f"- {entity_type}: {entity_text}")
        else:
            print("未识别出任何实体")

if __name__ == "__main__":
    main() 