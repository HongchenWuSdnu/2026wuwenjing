

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建输出目录
output_dir = 'deep_learning_experiment_results'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 第一部分：数据加载与传播树构建
# ============================================================

DATA_PATH = r"C:\Users\九黎\Downloads\sydneysiege"


def parse_time(time_str):
    """解析Twitter时间格式"""
    from datetime import datetime
    if time_str:
        try:
            return datetime.strptime(time_str, '%a %b %d %H:%M:%S +0000 %Y').timestamp()
        except:
            return 0
    return 0


def build_propagation_tree(thread_path):
    """构建单个对话线程的传播树"""
    texts = []
    edges = []
    timestamps = []
    id_to_idx = {}

    # 读取源推文
    source_dir = os.path.join(thread_path, 'source-tweets')
    if os.path.exists(source_dir):
        for file in os.listdir(source_dir):
            if file.endswith('.json'):
                with open(os.path.join(source_dir, file), 'r', encoding='utf-8') as f:
                    tweet = json.load(f)
                    idx = len(texts)
                    tweet_id = str(tweet.get('id_str', tweet.get('id', file)))
                    id_to_idx[tweet_id] = idx
                    texts.append(tweet.get('text', ''))
                    timestamps.append(parse_time(tweet.get('created_at')))

    # 读取回复推文
    reactions_dir = os.path.join(thread_path, 'reactions')
    if os.path.exists(reactions_dir):
        for file in os.listdir(reactions_dir):
            if file.endswith('.json'):
                with open(os.path.join(reactions_dir, file), 'r', encoding='utf-8') as f:
                    reaction = json.load(f)
                    idx = len(texts)
                    reply_id = reaction.get('in_reply_to_status_id_str', reaction.get('in_reply_to_status_id'))
                    if reply_id and str(reply_id) in id_to_idx:
                        edges.append((id_to_idx[str(reply_id)], idx))
                    tweet_id = str(reaction.get('id_str', reaction.get('id', file)))
                    id_to_idx[tweet_id] = idx
                    texts.append(reaction.get('text', ''))
                    timestamps.append(parse_time(reaction.get('created_at')))

    return texts, edges, timestamps


def load_all_threads(data_path, min_nodes=2):
    """加载所有对话线程"""
    all_graphs = []
    all_labels = []

    for category, label in [('rumours', 1), ('non-rumours', 0)]:
        cat_path = os.path.join(data_path, category)
        if not os.path.exists(cat_path):
            print(f"警告: 路径不存在 {cat_path}")
            continue

        print(f"加载 {category}...")
        for thread_id in os.listdir(cat_path):
            thread_path = os.path.join(cat_path, thread_id)
            if os.path.isdir(thread_path):
                texts, edges, timestamps = build_propagation_tree(thread_path)
                if len(texts) >= min_nodes:
                    all_graphs.append({
                        'texts': texts,
                        'edges': edges,
                        'timestamps': timestamps,
                        'thread_id': thread_id
                    })
                    all_labels.append(label)

    return all_graphs, all_labels


print("=" * 60)
print("加载数据并构建传播树...")
print("=" * 60)

graphs, labels = load_all_threads(DATA_PATH, min_nodes=2)
print(f"\n加载完成: {len(graphs)} 个有效传播树")
print(f"谣言: {sum(labels)} 个")
print(f"非谣言: {len(labels) - sum(labels)} 个")

if len(graphs) == 0:
    print("错误: 没有加载到任何数据，请检查数据路径")
    exit()

# ============================================================
# 第二部分：文本编码
# ============================================================

print("\n" + "=" * 60)
print("文本编码...")
print("=" * 60)

all_texts = []
for g in graphs:
    all_texts.extend(g['texts'])

tfidf = TfidfVectorizer(max_features=128, stop_words='english')
tfidf.fit(all_texts)
input_dim = 128
print(f"文本特征维度: {input_dim}")


def encode_graph(graph, tfidf_model):
    """将单个图编码为模型输入"""
    texts = graph['texts']
    edges = graph['edges']
    timestamps = graph['timestamps']

    num_nodes = len(texts)
    node_features = tfidf_model.transform(texts).toarray()
    node_features = torch.FloatTensor(node_features)

    adj = torch.zeros(num_nodes, num_nodes)
    for parent, child in edges:
        if parent < num_nodes and child < num_nodes:
            adj[parent, child] = 1
            adj[child, parent] = 1

    adj = adj + torch.eye(num_nodes)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    timestamps = torch.FloatTensor(timestamps)
    if timestamps.max() > timestamps.min():
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)

    return node_features, adj, timestamps


# ============================================================
# 第三部分：深度学习模型定义
# ============================================================

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output + self.bias


class PropagationGCN(nn.Module):
    """完整版：GCN + LSTM + Attention"""
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super(PropagationGCN, self).__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3)
        )
        self.gc1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, adj, timestamps):
        x = self.text_encoder(x)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.gc2(x, adj))
        sorted_indices = torch.argsort(timestamps)
        x_sorted = x[sorted_indices].unsqueeze(0)
        lstm_out, _ = self.lstm(x_sorted)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        graph_embedding = attn_out.mean(dim=1)
        return self.classifier(graph_embedding)


class GCNOnlyModel(nn.Module):
    """仅GCN模型"""
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super(GCNOnlyModel, self).__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3)
        )
        self.gc1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, adj, timestamps):
        x = self.text_encoder(x)
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        return self.classifier(x.mean(dim=0))


class LSTMOnlyModel(nn.Module):
    """仅LSTM模型"""
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super(LSTMOnlyModel, self).__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, adj, timestamps):
        x = self.text_encoder(x)
        sorted_indices = torch.argsort(timestamps)
        x_sorted = x[sorted_indices].unsqueeze(0)
        lstm_out, _ = self.lstm(x_sorted)
        return self.classifier(lstm_out.mean(dim=1))


class TextOnlyModel(nn.Module):
    """仅文本模型（基线）"""
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super(TextOnlyModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, adj, timestamps):
        return self.encoder(x.mean(dim=0))


# ============================================================
# 第四部分：数据集类与训练函数
# ============================================================

class RumorDataset(Dataset):
    def __init__(self, graphs, labels, tfidf_model):
        self.graphs = graphs
        self.labels = labels
        self.tfidf_model = tfidf_model

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        x, adj, ts = encode_graph(self.graphs[idx], self.tfidf_model)
        return x, adj, ts, self.labels[idx]


def collate_fn(batch):
    xs, adjs, tss, ys = zip(*batch)
    return list(xs), list(adjs), list(tss), torch.LongTensor(ys)


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for xs, adjs, tss, ys in dataloader:
        optimizer.zero_grad()
        batch_loss = 0
        batch_preds = []
        for x, adj, ts, y in zip(xs, adjs, tss, ys):
            x, adj, ts, y = x.to(device), adj.to(device), ts.to(device), y.to(device)
            logits = model(x, adj, ts)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            loss = criterion(logits, y.unsqueeze(0))
            batch_loss += loss
            pred = torch.argmax(logits, dim=1).item()
            batch_preds.append(pred)
            all_labels.append(y.item())
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        all_preds.extend(batch_preds)
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0
    return total_loss / max(len(dataloader), 1), acc


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for xs, adjs, tss, ys in dataloader:
            for x, adj, ts, y in zip(xs, adjs, tss, ys):
                x, adj, ts, y = x.to(device), adj.to(device), ts.to(device), y.to(device)
                logits = model(x, adj, ts)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                loss = criterion(logits, y.unsqueeze(0))
                total_loss += loss.item()
                probs = F.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1).item()
                all_preds.append(pred)
                all_labels.append(y.item())
                all_probs.append(probs[0, 1].item())
    if len(all_labels) == 0:
        return 0, 0, 0, 0, 0, 0
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    return total_loss / max(len(dataloader), 1), acc, prec, rec, f1, auc


# ============================================================
# 第五部分：消融实验与可视化
# ============================================================

def run_ablation_experiment(graphs, labels, tfidf_model):
    """运行消融实验并生成可视化"""

    # 划分数据集
    train_idx, test_idx = train_test_split(
        range(len(graphs)), test_size=0.2, random_state=42, stratify=labels
    )

    train_graphs = [graphs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"\n训练集: {len(train_graphs)} 个传播树")
    print(f"测试集: {len(test_graphs)} 个传播树")

    train_dataset = RumorDataset(train_graphs, train_labels, tfidf_model)
    test_dataset = RumorDataset(test_graphs, test_labels, tfidf_model)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    models_config = {
        '完整版 (GCN+LSTM+Attention)': PropagationGCN(input_dim, 64),
        '仅GCN (无LSTM)': GCNOnlyModel(input_dim, 64),
        '仅LSTM (无GCN)': LSTMOnlyModel(input_dim, 64),
        '基线 (仅文本)': TextOnlyModel(input_dim, 64),
    }

    results = []
    all_predictions = {}  # 存储每个模型的预测结果用于混淆矩阵

    for name, model in models_config.items():
        print(f"\n{'=' * 50}")
        print(f"训练模型: {name}")
        print('=' * 50)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        best_f1 = 0
        best_model_state = None

        for epoch in range(30):
            loss, acc = train_epoch(model, train_loader, optimizer, criterion)
            if epoch % 10 == 0:
                _, _, _, _, val_f1, _ = evaluate(model, test_loader, criterion)
                print(f"  Epoch {epoch:2d}: Loss={loss:.4f}, Train Acc={acc:.4f}, Val F1={val_f1:.4f}")
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if best_model_state:
            model.load_state_dict(best_model_state)
        model = model.to(device)

        loss, acc, prec, rec, f1, auc = evaluate(model, test_loader, criterion)

        # 获取预测结果用于混淆矩阵
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xs, adjs, tss, ys in test_loader:
                for x, adj, ts, y in zip(xs, adjs, tss, ys):
                    x, adj, ts = x.to(device), adj.to(device), ts.to(device)
                    logits = model(x, adj, ts)
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    pred = torch.argmax(logits, dim=1).item()
                    y_true.append(y)
                    y_pred.append(pred)
        all_predictions[name] = (y_true, y_pred)

        results.append({
            '模型': name,
            '准确率': acc,
            '精确率': prec,
            '召回率': rec,
            'F1分数': f1,
            'AUC': auc
        })

        print(f"\n  测试结果:")
        print(f"    准确率: {acc:.4f}")
        print(f"    精确率: {prec:.4f}")
        print(f"    召回率: {rec:.4f}")
        print(f"    F1分数: {f1:.4f}")
        print(f"    AUC: {auc:.4f}")

    return pd.DataFrame(results), all_predictions


print("\n" + "=" * 60)
print("开始消融实验")
print("=" * 60)

ablation_results, all_predictions = run_ablation_experiment(graphs, labels, tfidf)

# 保存结果到CSV
ablation_results.to_csv(os.path.join(output_dir, 'deep_learning_results.csv'), index=False)

print("\n" + "=" * 60)
print("消融实验结果汇总")
print("=" * 60)
print(ablation_results.to_string(index=False))

# ============================================================
# 第六部分：可视化（生成与机器学习实验类似的图片）
# ============================================================

print("\n" + "=" * 60)
print("生成可视化图表...")
print("=" * 60)

# 图1：深度学习模型性能对比柱状图（与机器学习实验的图4类似）
fig, ax = plt.subplots(figsize=(12, 6))

models_names = ablation_results['模型'].tolist()
accs = ablation_results['准确率'].tolist()
precs = ablation_results['精确率'].tolist()
recs = ablation_results['召回率'].tolist()
f1s = ablation_results['F1分数'].tolist()

x = np.arange(len(models_names))
width = 0.2

bars1 = ax.bar(x - 1.5*width, accs, width, label='准确率', color='steelblue')
bars2 = ax.bar(x - 0.5*width, precs, width, label='精确率', color='coral')
bars3 = ax.bar(x + 0.5*width, recs, width, label='召回率', color='seagreen')
bars4 = ax.bar(x + 1.5*width, f1s, width, label='F1分数', color='goldenrod')

ax.set_xlabel('模型', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('深度学习模型检测效果对比（含消融实验）', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models_names, rotation=15, ha='right')
ax.legend()
ax.set_ylim([0, 1])

# 在柱状图上添加数值
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_deep_learning_model_comparison.png'), dpi=150, bbox_inches='tight')
print(f"  已保存: {output_dir}/1_deep_learning_model_comparison.png")

# 图2：最佳模型的混淆矩阵（类似机器学习实验的图5左侧）
best_model_name = ablation_results.loc[ablation_results['F1分数'].idxmax(), '模型']
best_y_true, best_y_pred = all_predictions[best_model_name]

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(best_y_true, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['非谣言', '谣言'], yticklabels=['非谣言', '谣言'])
ax.set_xlabel('预测标签', fontsize=12)
ax.set_ylabel('真实标签', fontsize=12)
ax.set_title(f'{best_model_name} 混淆矩阵', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_best_model_confusion_matrix.png'), dpi=150, bbox_inches='tight')
print(f"  已保存: {output_dir}/2_best_model_confusion_matrix.png")

# 图3：消融实验F1分数对比柱状图（突出完整版 vs 消融版）
fig, ax = plt.subplots(figsize=(10, 6))

ablation_names = ablation_results['模型'].tolist()
ablation_f1 = ablation_results['F1分数'].tolist()

colors = ['#2ecc71' if '完整版' in n else '#e74c3c' if '基线' in n else '#3498db' for n in ablation_names]
bars = ax.bar(ablation_names, ablation_f1, color=colors)

ax.set_xlabel('模型变体', fontsize=12)
ax.set_ylabel('F1分数', fontsize=12)
ax.set_title('深度学习消融实验：各模块贡献对比', fontsize=14)
ax.set_ylim([0, 1])
ax.tick_params(axis='x', rotation=15)

for bar, f1 in zip(bars, ablation_f1):
    ax.annotate(f'{f1:.4f}',
               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
               xytext=(0, 3), textcoords="offset points",
               ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_ablation_study_f1.png'), dpi=150, bbox_inches='tight')
print(f"  已保存: {output_dir}/3_ablation_study_f1.png")

# 图4：性能排序（水平柱状图，类似机器学习实验的风格）
fig, ax = plt.subplots(figsize=(10, 6))
sorted_results = ablation_results.sort_values('F1分数', ascending=True)

colors = ['#2ecc71' if '完整版' in n else '#e74c3c' if '基线' in n else '#3498db' for n in sorted_results['模型']]
ax.barh(sorted_results['模型'], sorted_results['F1分数'], color=colors)

ax.set_xlabel('F1分数', fontsize=12)
ax.set_ylabel('模型', fontsize=12)
ax.set_title('深度学习模型性能排序（F1分数）', fontsize=14)
ax.set_xlim([0, 1])

for i, (_, row) in enumerate(sorted_results.iterrows()):
    ax.text(row['F1分数'] + 0.01, i, f"{row['F1分数']:.4f}", va='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_performance_ranking.png'), dpi=150, bbox_inches='tight')
print(f"  已保存: {output_dir}/4_performance_ranking.png")

# 图5：准确率、精确率、召回率、F1的多维度对比（雷达图）
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

metrics = ['准确率', '精确率', '召回率', 'F1分数']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

for _, row in ablation_results.iterrows():
    values = [row['准确率'], row['精确率'], row['召回率'], row['F1分数']]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=row['模型'])
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.set_title('深度学习模型多维度性能对比', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '5_radar_chart.png'), dpi=150, bbox_inches='tight')
print(f"  已保存: {output_dir}/5_radar_chart.png")

# ============================================================
# 第七部分：生成实验报告
# ============================================================

print("\n" + "=" * 60)
print("实验完成！生成报告...")
print("=" * 60)

print("\n" + "=" * 60)
print("深度学习实验摘要")
print("=" * 60)
print(f"数据集规模: {len(graphs)} 个传播树")
print(f"谣言数量: {sum(labels)} 个")
print(f"非谣言数量: {len(labels) - sum(labels)} 个")

print(f"\n最佳模型: {ablation_results.loc[ablation_results['F1分数'].idxmax(), '模型']}")
print(f"最高F1分数: {ablation_results['F1分数'].max():.4f}")

# 计算提升幅度
full_row = ablation_results[ablation_results['模型'] == '完整版 (GCN+LSTM+Attention)']
text_row = ablation_results[ablation_results['模型'] == '基线 (仅文本)']

if len(full_row) > 0 and len(text_row) > 0:
    full_f1 = full_row['F1分数'].values[0]
    text_f1 = text_row['F1分数'].values[0]
    if text_f1 > 0:
        improvement = (full_f1 - text_f1) / text_f1 * 100
        print(f"\n完整版 vs 基线:")
        print(f"  完整版 F1: {full_f1:.4f}")
        print(f"  基线 F1: {text_f1:.4f}")
        print(f"  提升幅度: {improvement:.1f}%")

print("\n" + "=" * 60)
print(f"所有结果已保存到: {output_dir}")
print("  生成的文件:")
print("    1_deep_learning_model_comparison.png - 深度学习模型对比柱状图")
print("    2_best_model_confusion_matrix.png - 最佳模型混淆矩阵")
print("    3_ablation_study_f1.png - 消融实验F1对比图")
print("    4_performance_ranking.png - 模型性能排序图")
print("    5_radar_chart.png - 多维度雷达图")
print("    deep_learning_results.csv - 详细实验结果")
print("=" * 60)

# 打印结果表格
print("\n深度学习消融实验结果汇总表:")
print(ablation_results[['模型', '准确率', '精确率', '召回率', 'F1分数', 'AUC']].to_string(index=False))