import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

x = -90
signal = torch.sigmoid(torch.tensor(x))
print(signal)
# ===== (1) 單一句子當資料 =====
sentence = "arrive Taipei on November 2nd".lower().split()

# 建簡易詞表：<pad>=0, <unk>=1
vocab = {w: i+2 for i, w in enumerate(sorted(set(sentence)))}
vocab["<pad>"], vocab["<unk>"] = 0, 1

# 把句子映成索引 (sequence of ints)
seq = [vocab.get(w, 1) for w in sentence]

# ==== (2) 偽資料集：同一句話配 3 種 label (0/1/2) ====
class ToyDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return torch.tensor(self.X[i]), torch.tensor(self.y[i])

X_all = [seq, seq, seq]   # 3 筆內容相同
y_all = [0, 1, 2]         # 3 個不同類別
ds = ToyDataset(X_all, y_all)
dl = DataLoader(ds, batch_size=3, shuffle=True)

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64,
                 num_classes=3, num_layers=1, bidirectional=False):
        super().__init__()
        # (2) Embedding：把詞 id 轉成向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # (3) RNN：最基本的 Elman RNN
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity='tanh',      # 可改 'relu'
            batch_first=True,
            bidirectional=bidirectional
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):             # x: (B, T)
        x = self.embedding(x)         # → (B, T, E)
        _, h_n = self.rnn(x)          # h_n: (L*D, B, H)
        last = h_n[-1]                # 取最後一層最後方向
        logits = self.fc(last)        # → (B, C)
        return logits


model = RNNClassifier(len(vocab))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for xb, yb in dl:
    logits = model(xb)               # 前向
    loss = loss_fn(logits, yb)       # 計算 loss
    optimizer.zero_grad()
    loss.backward()                  # 反傳
    optimizer.step()
    print("train loss:", loss.item())

