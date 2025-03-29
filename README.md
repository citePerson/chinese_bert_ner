# BERT NER Training Project

这个项目使用BERT模型进行中文命名实体识别(NER)训练。

## 环境要求

- Python 3.7+
- PyTorch
- Transformers
- 其他依赖见 requirements.txt

## 安装

1. 克隆项目到本地
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据格式

训练数据需要按照以下格式准备（ner_data.txt）：
```
文本1   标签1 标签2 标签3 ...
文本2   标签1 标签2 标签3 ...
```

标签说明：
- O: 非实体
- B-PER: 人名开始
- I-PER: 人名中间
- B-ORG: 组织开始
- I-ORG: 组织中间
- B-LOC: 地点开始
- I-LOC: 地点中间

## 使用步骤

1. 训练模型：
```bash
python bert_ner_train.py
```

## 模型输出

- 训练过程中会显示每个epoch的训练损失和验证损失
- 最佳模型会保存为 'best_ner_model.pt'
- 训练结束后会显示测试集的分类报告

## 注意事项

1. 确保有足够的GPU内存（如果使用GPU）
2. 可以通过修改 `bert_ner_train.py` 中的参数来调整：
   - batch_size
   - learning_rate
   - num_epochs
   - max_len
3. 默认使用 'bert-base-chinese' 预训练模型，可以根据需要更换其他模型 