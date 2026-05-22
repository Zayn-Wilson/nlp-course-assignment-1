import os
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Vocabulary:
    def __init__(self, token2idx=None):
        if token2idx is None:
            token2idx = {}
        self._token2idx = token2idx
        self._idx2token = {idx: token for token, idx in self._token2idx.items()}

    def add_token(self, token):
        if token in self._token2idx:
            return self._token2idx[token]
        idx = len(self._token2idx)
        self._token2idx[token] = idx
        self._idx2token[idx] = token
        return idx

    def lookup_token(self, token):
        return self._token2idx[token]

    def lookup_index(self, index):
        return self._idx2token[index]

    def __len__(self):
        return len(self._token2idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token2idx=None):
        super().__init__(token2idx)
        self.mask_token = "<MASK>"
        self.unk_token = "<UNK>"
        self.begin_seq_token = "<BEGIN>"
        self.end_seq_token = "<END>"
        self.mask_index = self.add_token(self.mask_token)
        self.unk_index = self.add_token(self.unk_token)
        self.begin_seq_index = self.add_token(self.begin_seq_token)
        self.end_seq_index = self.add_token(self.end_seq_token)

    def lookup_token(self, token):
        return self._token2idx.get(token, self.unk_index)


class SurnameVectorizer:
    def __init__(self, char_vocab, nationality_vocab):
        self.char_vocab = char_vocab
        self.nationality_vocab = nationality_vocab

    def vectorize(self, surname, vector_length=-1):
        indices = [self.char_vocab.begin_seq_index]
        indices.extend(self.char_vocab.lookup_token(ch) for ch in surname)
        indices.append(self.char_vocab.end_seq_index)
        if vector_length < 0:
            vector_length = len(indices)
        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.char_vocab.mask_index
        return out_vector, len(indices)

    @classmethod
    def from_dataframe(cls, surname_df):
        char_vocab = SequenceVocabulary()
        nationality_vocab = Vocabulary()
        for _, row in surname_df.iterrows():
            for ch in row.surname:
                char_vocab.add_token(ch)
            nationality_vocab.add_token(row.nationality)
        return cls(char_vocab, nationality_vocab)


class SurnameDataset(Dataset):
    def __init__(self, surname_df, vectorizer):
        self.surname_df = surname_df
        self._vectorizer = vectorizer
        self._max_seq_length = max(len(s) for s in surname_df.surname) + 2

        self.train_df = self.surname_df[self.surname_df.split == 'train']
        self.val_df = self.surname_df[self.surname_df.split == 'val']
        self.test_df = self.surname_df[self.surname_df.split == 'test']
        self._lookup_dict = {
            'train': (self.train_df, len(self.train_df)),
            'val': (self.val_df, len(self.val_df)),
            'test': (self.test_df, len(self.test_df)),
        }
        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv):
        surname_df = pd.read_csv(surname_csv)
        train_df = surname_df[surname_df.split == 'train']
        return cls(surname_df, SurnameVectorizer.from_dataframe(train_df))

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split='train'):
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        surname_vector, vec_length = self._vectorizer.vectorize(row.surname, self._max_seq_length)
        nationality_index = self._vectorizer.nationality_vocab.lookup_token(row.nationality)
        return {
            'x_data': torch.tensor(surname_vector, dtype=torch.long),
            'y_target': torch.tensor(nationality_index, dtype=torch.long),
            'x_length': torch.tensor(vec_length, dtype=torch.long),
        }

    def get_num_batches(self, batch_size):
        return len(self) // batch_size


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        out = {}
        for name, tensor in data_dict.items():
            out[name] = tensor.to(device)
        yield out


class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.batch_first = batch_first
        self.hidden_size = hidden_size

    def forward(self, x_in, initial_hidden=None):
        if self.batch_first:
            batch_size, seq_size, feat_size = x_in.size()
            x_in = x_in.permute(1, 0, 2)
        else:
            seq_size, batch_size, feat_size = x_in.size()
        hiddens = []
        if initial_hidden is None:
            initial_hidden = torch.zeros(batch_size, self.hidden_size, device=x_in.device)
        hidden_t = initial_hidden
        for t in range(seq_size):
            hidden_t = self.rnn_cell(x_in[t], hidden_t)
            hiddens.append(hidden_t)
        hiddens = torch.stack(hiddens)
        if self.batch_first:
            hiddens = hiddens.permute(1, 0, 2)
        return hiddens


def column_gather(y_out, x_lengths):
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1
    out = []
    for batch_idx, column_idx in enumerate(x_lengths):
        out.append(y_out[batch_idx, column_idx])
    return torch.stack(out)


class SurnameClassifier(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_classes,
                 rnn_hidden_size, padding_idx=0):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_embeddings,
                                embedding_dim=embedding_size,
                                padding_idx=padding_idx)
        self.rnn = ElmanRNN(input_size=embedding_size,
                            hidden_size=rnn_hidden_size)
        self.fc1 = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.fc2 = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x_in, x_lengths=None, apply_softmax=False, debug=False):
        x_embedded = self.emb(x_in)
        if debug:
            print(f"输入序列形状：{x_in.shape} -> 嵌入后：{x_embedded.shape}")
        y_out = self.rnn(x_embedded)
        if debug:
            if x_lengths is None:
                agg_info = y_out[:, -1, :].shape
            else:
                agg_info = '动态索引'
            print(f"RNN输出形状：{y_out.shape} -> 聚合后：{agg_info}")
        if x_lengths is not None:
            y_out = column_gather(y_out, x_lengths)
        else:
            y_out = y_out[:, -1, :]
        y_out = F.relu(self.fc1(F.dropout(y_out, 0.5, training=self.training)))
        y_out = self.fc2(F.dropout(y_out, 0.5, training=self.training))
        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)
        return y_out


def compute_accuracy(y_pred, y_target):
    y_pred_indices = y_pred.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def predict_nationality(surname, classifier, vectorizer, device="cpu"):
    vec, length = vectorizer.vectorize(surname)
    vec_tensor = torch.tensor(vec, dtype=torch.long).unsqueeze(0).to(device)
    length_tensor = torch.tensor([length], dtype=torch.long).to(device)
    classifier.eval()
    with torch.no_grad():
        y_pred = classifier(vec_tensor, length_tensor, apply_softmax=True)
    prob_values, indices = y_pred.max(dim=1)
    index = indices.item()
    prob = prob_values.item()
    predicted_nationality = vectorizer.nationality_vocab.lookup_index(index)
    return {'nationality': predicted_nationality, 'probability': prob}


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    CSV_PATH = r"F:\coding\2026-SpringClass\NLP\surnames_with_splits.csv"

    dataset = SurnameDataset.load_dataset_and_make_vectorizer(CSV_PATH)
    vectorizer = dataset.get_vectorizer()

    # ===== 任务2: 数据预处理验证 =====
    print("\n" + "=" * 60)
    print("【任务2】数据预处理验证")
    print("=" * 60)
    print("Char Vocab特殊标记索引：",
          f"<BEGIN>:{vectorizer.char_vocab.begin_seq_index}",
          f"<END>:{vectorizer.char_vocab.end_seq_index}")
    sample_str = "Zhang"
    vec, length = vectorizer.vectorize(sample_str)
    print(f"样本'{sample_str}'的向量化结果:\n{vec}\n有效长度:{length}")

    # ===== 任务3: 模型结构验证 =====
    num_classes = len(vectorizer.nationality_vocab)
    num_embeddings = len(vectorizer.char_vocab)
    embedding_size = 100
    rnn_hidden_size = 64

    classifier = SurnameClassifier(
        embedding_size=embedding_size,
        num_embeddings=num_embeddings,
        num_classes=num_classes,
        rnn_hidden_size=rnn_hidden_size,
        padding_idx=vectorizer.char_vocab.mask_index,
    ).to(device)

    print("\n" + "=" * 60)
    print("【任务3】模型结构验证")
    print("=" * 60)
    print("\n模型结构：")
    for name, param in classifier.named_parameters():
        print(f"{name.ljust(25)} | 维度：{tuple(param.size())}")

    # ===== 任务4: RNN序列处理验证 =====
    print("\n" + "=" * 60)
    print("【任务4】RNN序列处理验证")
    print("=" * 60)
    dataset.set_split('train')
    batch_gen = generate_batches(dataset, batch_size=4, device=device)
    sample_batch = next(batch_gen)
    classifier.eval()
    with torch.no_grad():
        _ = classifier(sample_batch['x_data'], sample_batch['x_length'], debug=True)

    # ===== 任务5: 训练50 epoch 并评估 =====
    print("\n" + "=" * 60)
    print("【任务5】模型训练 (50 epochs)")
    print("=" * 60)

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50

    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    for epoch_index in range(NUM_EPOCHS):
        dataset.set_split('train')
        classifier.train()
        running_loss, running_acc, batch_count = 0.0, 0.0, 0
        for batch_dict in generate_batches(dataset, batch_size=BATCH_SIZE, device=device):
            optimizer.zero_grad()
            y_pred = classifier(batch_dict['x_data'], batch_dict['x_length'])
            loss = loss_func(y_pred, batch_dict['y_target'])
            loss.backward()
            optimizer.step()
            running_loss += (loss.item() - running_loss) / (batch_count + 1)
            running_acc += (compute_accuracy(y_pred, batch_dict['y_target']) - running_acc) / (batch_count + 1)
            batch_count += 1
        train_loss_list.append(running_loss)
        train_acc_list.append(running_acc)

        dataset.set_split('val')
        classifier.eval()
        running_loss, running_acc, batch_count = 0.0, 0.0, 0
        with torch.no_grad():
            for batch_dict in generate_batches(dataset, batch_size=BATCH_SIZE, device=device):
                y_pred = classifier(batch_dict['x_data'], batch_dict['x_length'])
                loss = loss_func(y_pred, batch_dict['y_target'])
                running_loss += (loss.item() - running_loss) / (batch_count + 1)
                running_acc += (compute_accuracy(y_pred, batch_dict['y_target']) - running_acc) / (batch_count + 1)
                batch_count += 1
        val_loss_list.append(running_loss)
        val_acc_list.append(running_acc)

        if (epoch_index + 1) % 10 == 0 or epoch_index == 0:
            print(f"Epoch {epoch_index+1:3d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss_list[-1]:.4f} | "
                  f"Train Acc: {train_acc_list[-1]:.2f}% | "
                  f"Val Loss: {val_loss_list[-1]:.4f} | "
                  f"Val Acc: {val_acc_list[-1]:.2f}%")

    dataset.set_split('test')
    classifier.eval()
    running_loss, running_acc, batch_count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch_dict in generate_batches(dataset, batch_size=BATCH_SIZE, device=device):
            y_pred = classifier(batch_dict['x_data'], batch_dict['x_length'])
            loss = loss_func(y_pred, batch_dict['y_target'])
            running_loss += (loss.item() - running_loss) / (batch_count + 1)
            running_acc += (compute_accuracy(y_pred, batch_dict['y_target']) - running_acc) / (batch_count + 1)
            batch_count += 1
    print(f"\n测试集评估结果:")
    print(f"  Test Loss: {running_loss:.4f}")
    print(f"  Test Accuracy: {running_acc:.2f}%")

    # ===== 任务6: 推理能力验证 =====
    print("\n" + "=" * 60)
    print("【任务6】推理能力验证")
    print("=" * 60)
    test_surnames = ["McMahan", "Nakamoto", "Wan", "Cho"]
    for surname in test_surnames:
        result = predict_nationality(surname, classifier, vectorizer, device=device)
        print(f"  {surname:>10s} -> {result['nationality']:<12s} (置信度: {result['probability']:.4f})")

    print("\n所有任务完成！")
