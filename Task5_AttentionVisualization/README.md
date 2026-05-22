# 作业四：注意力机制可视化

## 1. 任务场景

**文本分类 - 情感分析**

本项目选择**文本分类**中的情感分析任务。通过可视化预训练模型在判断文本情感（正面/负面）时的注意力权重，直观理解模型"关注"了哪些词语来完成分类决策。

## 2. 模型与工具

| 组件 | 选择 |
|------|------|
| 预训练模型 | `distilbert-base-uncased-finetuned-sst-2-english` |
| 框架 | PyTorch + Transformers |
| 可视化工具 | Matplotlib + Seaborn (热力图) |
| 注意力实现 | `eager` (必须使用以提取注意力权重) |

**模型说明**：DistilBERT是BERT的轻量化版本，参数更少但效果相近。该模型在SST-2（斯坦福情感树库二分类数据集）上微调过，专门用于英文情感分类。

## 3. 可视化策略

### 可视化的层与注意力头

1. **单层单头**：第3层第5个注意力头
   - 文件：`sample_X_layer3_head5.png`
   - 目的：观察特定注意力头的精细模式

2. **多层概览**（每层内平均所有12个注意力头）
   - 文件：`sample_X_all_layers.png`
   - 6个子图展示全部6层Transformer层
   - 目的：观察注意力在网络中的演变过程

3. **Token重要性柱状图**
   - 文件：`sample_X_token_importance.png`
   - 跨所有层、所有头计算每个Token收到的平均注意力
   - 目的：识别哪些词对模型决策最重要

## 4. 分析与解释

### 测试样本

| 样本 | 文本 | 预测结果 | 置信度 |
|------|------|----------|--------|
| 1 | "This movie is absolutely fantastic! I loved every minute of it." | 正面 | 99.99% |
| 2 | "Terrible experience. The worst film I've ever seen." | 负面 | 99.97% |
| 3 | "The product quality is amazing and delivery was fast." | 正面 | 99.98% |
| 4 | "Waste of money. The item broke after one day." | 负面 | 99.82% |
| 5 | "Great customer service and excellent product quality!" | 正面 | 99.99% |

### 热力图发现的注意力模式

#### (1) 局部相邻词依赖
在**底层（Layer 1-2）**，注意力主要分布在相邻词之间。这反映了模型在早期层关注局部的语法结构，例如：
- 形容词与名词的搭配（"fantastic" → "movie"）
- 动词与宾语的关系（"loved" → "minute"）

#### (2) 情感词聚焦
**情感关键词**（如"fantastic"、"loved"、"amazing"、"terrible"、"worst"）在所有层中都收到较高的注意力。这说明模型学会了将目光聚焦在携带强烈情感信号的词汇上。

#### (3) [CLS] Token 聚合
[CLS]（分类Token）在后续层中持续从内容词接收注意力。这是预期的行为，因为DistilBERT使用[CLS]的表示来生成分类logits。

#### (4) 长距离依赖
在**高层（Layer 5-6）**，注意力的分布变得更加分散，出现跨句子级别的依赖。这帮助模型理解上下文和隐含语义，例如感叹号与正面情感的关联。

### 注意力分配的合理性解释

| 模式 | 为什么有帮助 |
|------|-------------|
| 局部依赖 | 捕获基本语法关系，理解词序和搭配 |
| 情感词聚焦 | 直接识别情感信号，加速判断 |
| [CLS]聚合 | 将全句信息汇总用于最终分类 |
| 长距离依赖 | 捕捉上下文 nuance，如双重否定、讽刺 |

## 5. 输出文件

所有可视化结果保存在 `attention_visualization/` 文件夹中：

| 文件名 | 描述 |
|--------|------|
| `sample_1_layer3_head5.png` | 样本1的第3层第5头热力图 |
| `sample_1_all_layers.png` | 样本1的6层概览 |
| `sample_1_token_importance.png` | 样本1的Token重要性图 |
| `sample_2_*.png` ~ `sample_5_*.png` | 其他4个样本的可视化 |
| `detailed_positive_analysis.png` | 正面情感文本的详细6层分析 |

## 6. 运行方式

```bash
# 安装依赖
uv pip install torch transformers matplotlib seaborn

# 运行程序
cd Homework4_Attention_Visualization
python attention_visualization.py
```

## 7. 代码结构

```
Homework4_Attention_Visualization/
├── README.md                          # 本文档（中文）
├── attention_visualization.py         # 主程序代码
├── pyproject.toml                     # 依赖管理
└── attention_visualization/           # 可视化输出文件夹
    ├── sample_1_layer3_head5.png
    ├── sample_1_all_layers.png
    ├── sample_1_token_importance.png
    ├── detailed_positive_analysis.png
    └── ...
```

## 8. 核心代码逻辑

```python
# 1. 加载模型（使用eager注意力以支持权重提取）
model = DistilBertForSequenceClassification.from_pretrained(
    model_name, attn_implementation="eager"
)

# 2. 前向传播，获取注意力权重
outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions  # tuple: (num_layers, batch, heads, seq, seq)

# 3. 可视化指定层和头
attn = attentions[layer_idx][0][head_idx]  # (seq_len, seq_len)
sns.heatmap(attn, ...)  # 使用seaborn绘制热力图
```
