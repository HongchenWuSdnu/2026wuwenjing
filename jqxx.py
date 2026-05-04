

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
from sklearn.naive_bayes import MultinomialNB
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("悉尼人质事件 - 网络谣言实时检测实验")
print("=" * 60)

# ============================================================
# 第一部分：数据加载与预处理
# ============================================================

# 请确认这个路径是你的数据文件夹路径
DATA_PATH = "C:\\Users\\九黎\\Downloads\\sydneysiege"


def load_pheme_event(event_path):

    data = []

    # 遍历谣言和非谣言文件夹
    for category in ['rumours', 'non-rumours']:
        category_path = os.path.join(event_path, category)
        if not os.path.exists(category_path):
            print(f"警告: 路径不存在 {category_path}")
            continue

        print(f"正在加载 {category}...")

        for thread_id in os.listdir(category_path):
            thread_path = os.path.join(category_path, thread_id)
            if not os.path.isdir(thread_path):
                continue

            # 读取源推文
            source_dir = os.path.join(thread_path, 'source-tweets')
            if os.path.exists(source_dir):
                for source_file in os.listdir(source_dir):
                    if source_file.endswith('.json'):
                        source_path = os.path.join(source_dir, source_file)
                        with open(source_path, 'r', encoding='utf-8') as f:
                            try:
                                tweet = json.load(f)
                                tweet['is_rumor'] = 1 if category == 'rumours' else 0
                                tweet['is_source'] = True
                                tweet['thread_id'] = thread_id
                                data.append(tweet)
                            except:
                                print(f"  跳过损坏文件: {source_path}")

            # 读取回复推文
            reactions_dir = os.path.join(thread_path, 'reactions')
            if os.path.exists(reactions_dir):
                for reaction_file in os.listdir(reactions_dir):
                    if reaction_file.endswith('.json'):
                        reaction_path = os.path.join(reactions_dir, reaction_file)
                        with open(reaction_path, 'r', encoding='utf-8') as f:
                            try:
                                reaction = json.load(f)
                                reaction['is_rumor'] = 1 if category == 'rumours' else 0
                                reaction['is_source'] = False
                                reaction['thread_id'] = thread_id
                                data.append(reaction)
                            except:
                                continue

    df = pd.DataFrame(data)
    print(f"  加载完成: {len(df)} 条推文")
    return df


def preprocess_data(df):
    """
    数据预处理：提取时间戳、清洗文本等
    """
    # 提取时间戳
    if 'created_at' in df.columns:
        df['datetime'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y', errors='coerce')
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    else:
        df['datetime'] = pd.NaT

    # 计算相对于事件开始的时间（秒）
    if len(df) > 0 and df['datetime'].notna().any():
        event_start = df['datetime'].min()
        df['time_seconds'] = (df['datetime'] - event_start).dt.total_seconds()
        df['time_minutes'] = df['time_seconds'] / 60

    # 清洗文本
    if 'text' in df.columns:
        df['clean_text'] = df['text'].fillna('').astype(str)
        # 移除URL
        df['clean_text'] = df['clean_text'].str.replace(r'http\S+', '', regex=True)
        # 移除@提及
        df['clean_text'] = df['clean_text'].str.replace(r'@\w+', '', regex=True)
        # 移除特殊字符（保留字母、数字、空格）
        df['clean_text'] = df['clean_text'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        df['clean_text'] = df['clean_text'].str.lower()

    return df


def extract_text_features(df):
    """
    提取文本特征（长度、标点、情感等）
    """
    features = pd.DataFrame(index=df.index)

    # 文本长度特征
    features['text_length'] = df['clean_text'].str.len()
    features['word_count'] = df['clean_text'].str.split().str.len()

    # 标点符号特征
    features['exclamation_count'] = df['text'].fillna('').astype(str).str.count('!')
    features['question_count'] = df['text'].fillna('').astype(str).str.count(r'\?')
    features['uppercase_ratio'] = df['text'].fillna('').astype(str).apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )

    # 谣言关键词特征
    rumor_keywords = ['confirmed', 'breaking', 'urgent', 'alert', 'warning',
                      'source', 'report', 'update', 'developing', 'breaking news']
    df_lower = df['clean_text'].fillna('').astype(str).str.lower()
    for keyword in rumor_keywords:
        features[f'kw_{keyword.replace(" ", "_")}'] = df_lower.str.contains(keyword).astype(int)

    # 不确定性词特征
    uncertainty_words = ['maybe', 'perhaps', 'rumor', 'heard', 'allegedly',
                         'unconfirmed', 'reportedly', 'supposedly', 'claim']
    features['uncertainty_count'] = df_lower.apply(
        lambda x: sum(1 for w in uncertainty_words if w in x)
    )

    # 简单情感特征（基于关键词）
    positive_words = ['good', 'great', 'safe', 'thank', 'hope', 'support']
    negative_words = ['dead', 'death', 'kill', 'attack', 'terror', 'fear',
                      'hostage', 'shoot', 'bomb', 'warning', 'danger']

    features['positive_score'] = df_lower.apply(
        lambda x: sum(1 for w in positive_words if w in x)
    )
    features['negative_score'] = df_lower.apply(
        lambda x: sum(1 for w in negative_words if w in x)
    )
    features['sentiment'] = features['positive_score'] - features['negative_score']

    return features


# ============================================================
# 第二部分：加载数据
# ============================================================

print("\n[1/6] 加载数据...")

# 检查路径是否存在
if not os.path.exists(DATA_PATH):
    print(f"错误: 数据路径不存在: {DATA_PATH}")
    print("请确认路径是否正确，例如: C:\\Users\\九黎\\Downloads\\sydneysiege")
    exit()

df = load_pheme_event(DATA_PATH)

if len(df) == 0:
    print("错误: 未加载到任何数据，请检查数据路径和文件夹结构")
    print("期望的文件夹结构:")
    print("  sydneysiege/")
    print("    ├── rumours/")
    print("    │   └── [thread_id]/")
    print("    │       ├── source-tweets/")
    print("    │       └── reactions/")
    print("    └── non-rumours/")
    print("        └── [thread_id]/")
    print("            ├── source-tweets/")
    print("            └── reactions/")
    exit()

print(f"数据加载完成: {len(df)} 条推文")
print(f"谣言: {df['is_rumor'].sum()} 条")
print(f"非谣言: {len(df) - df['is_rumor'].sum()} 条")

# 预处理
df = preprocess_data(df)
print(f"有效时间戳数据: {df['datetime'].notna().sum()} 条")

# 提取文本特征
text_features = extract_text_features(df)
df = pd.concat([df, text_features], axis=1)

# ============================================================
# 第三部分：数据分析与可视化
# ============================================================

print("\n[2/6] 数据分析...")

# 创建输出目录
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3.1 数据分布可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 谣言 vs 非谣言分布
ax1 = axes[0, 0]
counts = df['is_rumor'].value_counts()
ax1.bar(['非谣言', '谣言'], counts.values, color=['steelblue', 'coral'])
ax1.set_title('谣言与非谣言分布')
ax1.set_ylabel('推文数量')
for i, v in enumerate(counts.values):
    ax1.text(i, v + 10, str(v), ha='center')

# 时间分布
ax2 = axes[0, 1]
if df['time_minutes'].notna().any():
    rumor_time = df[df['is_rumor'] == 1]['time_minutes'].dropna()
    non_rumor_time = df[df['is_rumor'] == 0]['time_minutes'].dropna()
    ax2.hist([rumor_time, non_rumor_time], bins=30, alpha=0.7,
             label=['谣言', '非谣言'], color=['coral', 'steelblue'])
    ax2.set_xlabel('时间（分钟）')
    ax2.set_ylabel('推文数量')
    ax2.set_title('推文时间分布')
    ax2.legend()

# 文本长度对比
ax3 = axes[1, 0]
df.boxplot(column='text_length', by='is_rumor', ax=ax3)
ax3.set_title('文本长度对比')
ax3.set_xlabel('谣言（1）vs 非谣言（0）')
ax3.set_ylabel('文本长度')

# 情感得分对比
ax4 = axes[1, 1]
df.boxplot(column='sentiment', by='is_rumor', ax=ax4)
ax4.set_title('情感得分对比')
ax4.set_xlabel('谣言（1）vs 非谣言（0）')
ax4.set_ylabel('情感得分')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_data_analysis.png'), dpi=150, bbox_inches='tight')
print(f"  已保存: {output_dir}/1_data_analysis.png")

# ============================================================
# 第四部分：早期检测实验（方向一 - 核心）
# ============================================================

print("\n[3/6] 早期检测实验...")


def early_detection_experiment(df, time_thresholds, model_type='lr'):
    """
    早期检测实验：测试不同时间窗口的检测效果
    """
    # 按时间排序
    df_sorted = df[df['time_seconds'].notna()].sort_values('time_seconds')

    if len(df_sorted) == 0:
        print("  错误：没有有效的时间数据")
        return pd.DataFrame()

    results = []

    for threshold in time_thresholds:
        # 前threshold秒的数据作为测试集
        test_df = df_sorted[df_sorted['time_seconds'] <= threshold]
        train_df = df_sorted[df_sorted['time_seconds'] > threshold]

        if len(train_df) < 10 or len(test_df) < 5:
            print(f"  时间窗口 {threshold // 60}min: 数据不足，跳过")
            continue

        # 特征：使用TF-IDF
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')

        try:
            X_train = tfidf.fit_transform(train_df['clean_text'].fillna(''))
            X_test = tfidf.transform(test_df['clean_text'].fillna(''))
            y_train = train_df['is_rumor']
            y_test = test_df['is_rumor']

            # 选择模型
            if model_type == 'lr':
                model = LogisticRegression(max_iter=1000)
            elif model_type == 'svm':
                model = SVC(kernel='linear', probability=True)
            elif model_type == 'rf':
                model = RandomForestClassifier(n_estimators=100)
            else:
                model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                'time_window_min': threshold // 60,
                'time_window_sec': threshold,
                'test_size': len(test_df),
                'train_size': len(train_df),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            })

            print(f"  {threshold // 60}分钟窗口: Acc={accuracy_score(y_test, y_pred):.3f}, "
                  f"F1={f1_score(y_test, y_pred, zero_division=0):.3f}, "
                  f"样本数={len(test_df)}")

        except Exception as e:
            print(f"  时间窗口 {threshold // 60}min 出错: {e}")
            continue

    return pd.DataFrame(results)


# 定义时间窗口（秒）
time_windows = [300, 600, 900, 1800, 3600, 7200]  # 5,10,15,30,60,120分钟

early_results = early_detection_experiment(df, time_windows, model_type='lr')

# 可视化早期检测结果
if len(early_results) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(early_results['time_window_min'], early_results['accuracy'],
            'o-', label='准确率', linewidth=2, markersize=8)
    ax.plot(early_results['time_window_min'], early_results['f1'],
            's-', label='F1分数', linewidth=2, markersize=8)
    ax.plot(early_results['time_window_min'], early_results['precision'],
            '^-', label='精确率', linewidth=2, markersize=8)
    ax.plot(early_results['time_window_min'], early_results['recall'],
            'd-', label='召回率', linewidth=2, markersize=8)

    ax.set_xlabel('时间窗口（分钟）', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('早期检测性能随时间变化', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_early_detection.png'), dpi=150, bbox_inches='tight')
    print(f"  已保存: {output_dir}/2_early_detection.png")

    # 保存结果到CSV
    early_results.to_csv(os.path.join(output_dir, 'early_detection_results.csv'), index=False)
    print(f"  已保存: {output_dir}/early_detection_results.csv")
else:
    print("  早期检测实验无有效结果")

# ============================================================
# 第五部分：特征工程实验（方向三）
# ============================================================

print("\n[4/6] 特征工程实验...")


def feature_importance_experiment(df):
    """
    特征重要性分析：找出哪些特征对谣言检测最重要
    """
    # 准备特征
    feature_cols = ['text_length', 'word_count', 'exclamation_count', 'question_count',
                    'uppercase_ratio', 'uncertainty_count', 'positive_score',
                    'negative_score', 'sentiment']

    # 添加关键词特征
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    feature_cols.extend(kw_cols)

    # 过滤存在的特征
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df['is_rumor']

    # 随机森林特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = importance_df.head(10)
    ax.barh(top_features['feature'], top_features['importance'], color='steelblue')
    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_title('谣言检测特征重要性排名', fontsize=14)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_feature_importance.png'), dpi=150, bbox_inches='tight')
    print(f"  已保存: {output_dir}/3_feature_importance.png")

    print("\n  特征重要性排名:")
    print(importance_df.head(10).to_string(index=False))

    # 保存结果
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    return importance_df


importance_df = feature_importance_experiment(df)

# ============================================================
# 第六部分：模型对比实验
# ============================================================

print("\n[5/6] 模型对比实验...")


def model_comparison(df):
    """
    比较不同机器学习模型的检测效果
    """
    # 准备数据
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
    X = tfidf.fit_transform(df['clean_text'].fillna(''))
    y = df['is_rumor']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        '逻辑回归': LogisticRegression(max_iter=1000),
        '支持向量机': SVC(kernel='linear'),
        '随机森林': RandomForestClassifier(n_estimators=100),
        '朴素贝叶斯': MultinomialNB()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            '模型': name,
            '准确率': accuracy_score(y_test, y_pred),
            '精确率': precision_score(y_test, y_pred, zero_division=0),
            '召回率': recall_score(y_test, y_pred, zero_division=0),
            'F1分数': f1_score(y_test, y_pred, zero_division=0)
        })

        print(f"  {name}: Acc={accuracy_score(y_test, y_pred):.3f}, "
              f"F1={f1_score(y_test, y_pred, zero_division=0):.3f}")

    results_df = pd.DataFrame(results)

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results_df))
    width = 0.2

    ax.bar(x - 1.5 * width, results_df['准确率'], width, label='准确率')
    ax.bar(x - 0.5 * width, results_df['精确率'], width, label='精确率')
    ax.bar(x + 0.5 * width, results_df['召回率'], width, label='召回率')
    ax.bar(x + 1.5 * width, results_df['F1分数'], width, label='F1分数')

    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('不同模型检测效果对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['模型'])
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_model_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  已保存: {output_dir}/4_model_comparison.png")

    # 保存结果
    results_df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)

    return results_df


model_results = model_comparison(df)

# ============================================================
# 第七部分：混淆矩阵与ROC曲线
# ============================================================

print("\n[6/6] 混淆矩阵与ROC曲线...")


def confusion_matrix_and_roc(df):
    """
    绘制混淆矩阵和ROC曲线
    """
    # 使用最佳模型（逻辑回归）
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
    X = tfidf.fit_transform(df['clean_text'].fillna(''))
    y = df['is_rumor']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['非谣言', '谣言'], yticklabels=['非谣言', '谣言'])
    axes[0].set_xlabel('预测标签', fontsize=12)
    axes[0].set_ylabel('真实标签', fontsize=12)
    axes[0].set_title('混淆矩阵', fontsize=14)

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('假正率', fontsize=12)
    axes[1].set_ylabel('真正率', fontsize=12)
    axes[1].set_title('ROC曲线', fontsize=14)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_confusion_matrix_roc.png'), dpi=150, bbox_inches='tight')
    print(f"  已保存: {output_dir}/5_confusion_matrix_roc.png")


confusion_matrix_and_roc(df)

# ============================================================
# 第八部分：生成实验报告
# ============================================================

print("\n" + "=" * 60)
print("实验完成！生成报告...")
print("=" * 60)

# 生成实验摘要
print("\n" + "=" * 60)
print("实验摘要")
print("=" * 60)
print(f"数据集规模: {len(df)} 条推文")
print(f"谣言数量: {df['is_rumor'].sum()} 条")
print(f"非谣言数量: {len(df) - df['is_rumor'].sum()} 条")
print(f"对话线程数: {df['thread_id'].nunique()} 个")

if len(early_results) > 0:
    best_f1 = early_results.loc[early_results['f1'].idxmax()]
    print(f"\n最佳检测性能: {best_f1['time_window_min']}分钟窗口, F1={best_f1['f1']:.3f}")

print(f"\n最佳模型: {model_results.loc[model_results['F1分数'].idxmax(), '模型']}")
print(f"最高F1分数: {model_results['F1分数'].max():.3f}")

print("\n" + "=" * 60)
print(f"所有结果已保存到: {output_dir}")
print("  1_data_analysis.png - 数据分析图表")
print("  2_early_detection.png - 早期检测曲线")
print("  3_feature_importance.png - 特征重要性")
print("  4_model_comparison.png - 模型对比")
print("  5_confusion_matrix_roc.png - 混淆矩阵和ROC曲线")
print("  early_detection_results.csv - 早期检测详细结果")
print("  model_comparison_results.csv - 模型对比结果")
print("  feature_importance.csv - 特征重要性表")
print("=" * 60)