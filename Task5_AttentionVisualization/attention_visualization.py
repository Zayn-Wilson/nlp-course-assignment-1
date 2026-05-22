"""
注意力热力图可视化 - 文本分类任务
使用 DistilBERT 模型分析情感分类时的注意力分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_tokenizer():
    """加载DistilBERT情感分类模型和分词器"""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"正在加载模型: {model_name}")

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, attn_implementation="eager"
    )
    model.eval()

    return model, tokenizer


def analyze_text(text, model, tokenizer):
    """分析单个文本的注意力分布"""
    print(f"\n{'='*60}")
    print(f"输入文本: {text}")
    print(f"{'='*60}")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        predictions = torch.argmax(outputs.logits, dim=-1)

    pred_label = "正面" if predictions.item() == 1 else "负面"
    probs = torch.softmax(outputs.logits, dim=-1)
    print(f"预测结果: {pred_label}")
    print(f"置信度: 正面={probs[0][1].item():.4f}, 负面={probs[0][0].item():.4f}")

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attentions = outputs.attentions

    return tokens, attentions


def plot_attention_heatmap(tokens, attentions, layer_idx, head_idx, save_path):
    """绘制单层单头的注意力热力图"""
    attn = attentions[layer_idx][0][head_idx].numpy()

    fig, ax = plt.subplots(figsize=(12, 10))

    display_len = min(len(tokens), attn.shape[0])
    attn_display = attn[:display_len, :display_len]
    tokens_display = tokens[:display_len]

    sns.heatmap(attn_display,
               xticklabels=tokens_display,
               yticklabels=tokens_display,
               cmap='viridis',
               ax=ax,
               cbar_kws={'label': 'Attention Weight'})

    ax.set_xlabel('Key Tokens', fontsize=12)
    ax.set_ylabel('Query Tokens', fontsize=12)
    ax.set_title(f'Layer {layer_idx + 1}, Head {head_idx + 1} - Attention Heatmap', fontsize=14)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"热力图已保存: {save_path}")


def plot_all_layers_summary(tokens, attentions, save_path):
    """绘制所有层的注意力概览（平均所有头）"""
    num_layers = len(attentions)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    display_len = min(64, len(tokens))

    for idx in range(min(6, num_layers)):
        attn = attentions[idx][0].mean(dim=0).numpy()[:display_len, :display_len]
        tokens_display = tokens[:display_len]

        sns.heatmap(attn,
                   xticklabels=tokens_display,
                   yticklabels=tokens_display,
                   cmap='viridis',
                   ax=axes[idx],
                   cbar_kws={'label': 'Avg Weight'})

        axes[idx].set_title(f'Layer {idx + 1} (Avg All Heads)', fontsize=11)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.suptitle('Multi-Layer Attention Overview', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"多层概览已保存: {save_path}")


def plot_token_attention(tokens, attentions, save_path):
    """绘制每个token的平均注意力分布"""
    all_attn = torch.stack([a[0] for a in attentions])
    all_attn = all_attn.mean(dim=[0, 1]).numpy()

    token_attention = all_attn.mean(axis=0)

    valid_tokens = []
    valid_attentions = []
    for i, t in enumerate(tokens):
        if t not in ['[PAD]', '[CLS]', '[SEP]']:
            valid_tokens.append(t)
            valid_attentions.append(token_attention[i])

    fig, ax = plt.subplots(figsize=(14, 6))

    y_pos = np.arange(len(valid_tokens))
    ax.barh(y_pos, valid_attentions, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid_tokens)
    ax.invert_yaxis()
    ax.set_xlabel('Average Attention Received', fontsize=12)
    ax.set_title('Token Importance Based on Attention', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Token重要性图已保存: {save_path}")


def main():
    output_dir = "attention_visualization"
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()

    test_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible experience. The worst film I've ever seen.",
        "The product quality is amazing and delivery was fast.",
        "Waste of money. The item broke after one day.",
        "Great customer service and excellent product quality!"
    ]

    for i, text in enumerate(test_texts):
        print(f"\n{'#'*60}")
        print(f"样本 {i + 1}")
        print(f"{'#'*60}")

        tokens, attentions = analyze_text(text, model, tokenizer)

        base_name = f"sample_{i+1}"

        plot_attention_heatmap(tokens, attentions,
                              layer_idx=2, head_idx=4,
                              save_path=os.path.join(output_dir, f"{base_name}_layer3_head5.png"))

        plot_all_layers_summary(tokens, attentions,
                               save_path=os.path.join(output_dir, f"{base_name}_all_layers.png"))

        plot_token_attention(tokens, attentions,
                           save_path=os.path.join(output_dir, f"{base_name}_token_importance.png"))

    print("\n" + "="*60)
    print("详细分析: 正面情感文本")
    print("="*60)

    positive_text = "I really love this product. It works perfectly and exceeded my expectations!"
    tokens, attentions = analyze_text(positive_text, model, tokenizer)

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    display_len = min(50, len(tokens))

    for layer_idx in range(min(6, len(attentions))):
        attn = attentions[layer_idx][0].mean(dim=0).numpy()[:display_len, :display_len]
        tokens_display = tokens[:display_len]

        sns.heatmap(attn,
                   xticklabels=tokens_display,
                   yticklabels=tokens_display,
                   cmap='viridis',
                   ax=axes[layer_idx],
                   cbar_kws={'label': 'Attention'})

        axes[layer_idx].set_title(f'Layer {layer_idx + 1} - Mean Attention', fontsize=11)
        axes[layer_idx].tick_params(axis='x', rotation=45)

    plt.suptitle(f'Positive Sentiment: "{positive_text[:50]}..."', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detailed_positive_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n详细分析图已保存: {output_dir}/detailed_positive_analysis.png")

    print("\n" + "="*60)
    print("可视化完成！")
    print("="*60)
    print(f"所有结果保存在: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
