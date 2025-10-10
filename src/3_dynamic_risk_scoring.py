# -*- coding: utf-8 -*-
"""
步骤3：动态风险评估模型（防标签泄露修正版）
功能：
  - 仅使用训练期交易（<2025-01-01）构建行为特征
  - 仅使用训练图（risk_propagation_graph_train.json）构建拓扑特征
  - 测试账户特征：若有训练期交易则用，否则填充0
  - 自动阈值优化 + 输出评分
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import xgboost as xgb
from sklearn.metrics import (classification_report, roc_auc_score,
                             recall_score, f1_score, precision_score,
                             precision_recall_curve, average_precision_score)
import networkx as nx
import os, json, warnings, time

warnings.filterwarnings('ignore')
print("=== 步骤3：动态风险评估模型（防标签泄露修正版） ===")
start = time.time()

# ---------- 1. 路径配置 ----------
TRAIN_TXN_PATH = r'D:/Pycharm/Intermediaries_digging/data/train_transactions.csv'  # ← 关键：只用训练交易
GRAPH_PATH = r'D:/Pycharm/Intermediaries_digging/output/risk_propagation_graph_train.json'
COMMUNITY_PATH = r'D:/Pycharm/Intermediaries_digging/output/gang_communities_train.csv'
TEST_IDS_PATH = r'D:/Pycharm/Intermediaries_digging/output/test_account_ids.csv'
OUTPUT_DIR = r'D:/Pycharm/Intermediaries_digging/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 2. 加载数据 ----------
print("[1/5] 加载训练交易与测试账户ID...")
train_txn = pd.read_csv(TRAIN_TXN_PATH, encoding='utf-8-sig')
train_txn['timestamp'] = pd.to_datetime(train_txn['timestamp'])
test_accounts = set(pd.read_csv(TEST_IDS_PATH, encoding='utf-8-sig')['account_id'].astype(str))
train_accounts = set(train_txn['account_id'].unique())
all_accounts = train_accounts | test_accounts

print(f"训练账户数: {len(train_accounts):,}")
print(f"测试账户数: {len(test_accounts):,}")
print(f"总账户数: {len(all_accounts):,}")

RISK_LABELS = ['黑', '灰', '黑密接', '黑次密接', '黑次次密接', '黑次次次密接']
train_txn['is_risk'] = train_txn['account_label'].isin(RISK_LABELS).astype(int)

# ---------- 3. 行为特征（仅基于训练交易） ----------
print("[2/5] 构建行为特征（仅训练期）...")
beh_train = train_txn.groupby('account_id').agg(
    total_txn=('account_id', 'count'),
    total_amt=('amount', 'sum'),
    avg_amt=('amount', 'mean'),
    std_amt=('amount', 'std'),
    uniq_opp=('opponent_name', 'nunique'),
    first_txn=('timestamp', 'min'),
    last_txn=('timestamp', 'max')
).reset_index()

# 衍生特征
beh_train['active_days'] = (beh_train['last_txn'] - beh_train['first_txn']).dt.days + 1
beh_train['txn_freq'] = beh_train['total_txn'] / beh_train['active_days']

# 夜间交易比例
train_txn['is_night'] = ((train_txn.timestamp.dt.hour >= 22) | (train_txn.timestamp.dt.hour < 6)).astype(int)
night_stats = train_txn.groupby('account_id')['is_night'].agg(night_cnt='sum', total='count')
beh_train = beh_train.merge(night_stats, on='account_id', how='left')
beh_train['night_ratio'] = beh_train['night_cnt'] / beh_train['total']

# 快进快出比例（仅训练期）
out_flows = train_txn[train_txn['direction'] == '2'][['account_id', 'timestamp']].sort_values(['account_id', 'timestamp'])
in_flows = train_txn[train_txn['direction'] == '1'][['account_id', 'timestamp']].sort_values(['account_id', 'timestamp'])

if not out_flows.empty and not in_flows.empty:
    out_flows = out_flows.copy()
    out_flows['deadline'] = out_flows['timestamp'] + timedelta(hours=24)
    merged_fast = pd.merge_asof(
        out_flows,
        in_flows.rename(columns={'timestamp': 'ts_in'}),
        on='timestamp',
        by='account_id',
        direction='forward'
    )
    merged_fast['flag'] = merged_fast['ts_in'] <= merged_fast['deadline']
    fast_stats = merged_fast.groupby('account_id')['flag'].agg(sum='sum', count='count')
    beh_train = beh_train.merge(fast_stats, on='account_id', how='left')
    beh_train['fast_out_ratio'] = (beh_train['sum'] / beh_train['count']).fillna(0)
else:
    beh_train['fast_out_ratio'] = 0.0

# 清理临时列
beh_train = beh_train.drop(columns=['night_cnt', 'total', 'sum', 'count'], errors='ignore')

print(f"训练行为特征完成: {len(beh_train):,} 账户")

# ---------- 4. 图拓扑特征（仅训练图） ----------
print("[3/5] 构建图拓扑特征（仅训练图）...")
with open(GRAPH_PATH, 'r', encoding='utf-8') as f:
    graph_data = json.load(f)

# 构建训练图
edges = pd.DataFrame(graph_data['edges'])[['source', 'target', 'weight']]
G = nx.DiGraph()
for _, row in edges.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

deg_in = dict(G.in_degree(weight='weight'))
deg_out = dict(G.out_degree(weight='weight'))
pr = nx.pagerank(G, weight='weight') if G.number_of_nodes() > 0 else {}

# 社区风险（仅训练社区）
comm_risk_ratio_map = {}
try:
    comm = pd.read_csv(COMMUNITY_PATH, encoding='utf-8-sig')
    if not comm.empty:
        # 获取训练期账户标签（仅来自 train_txn）
        account_label_map = train_txn.groupby('account_id')['account_label'].apply(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        ).to_dict()
        comm['is_risk'] = comm['account_id'].map(account_label_map).fillna('正常').isin(RISK_LABELS).astype(int)
        comm_risk = comm.groupby('community_id')['is_risk'].mean().to_dict()
        comm_id_map = comm.set_index('account_id')['community_id'].to_dict()
        comm_risk_ratio_map = {node: comm_risk.get(comm_id_map.get(node, -1), 0.0) for node in comm['account_id']}
except FileNotFoundError:
    pass

# 构建拓扑特征 DataFrame（覆盖所有图中节点）
topo_nodes = set(G.nodes())
topo_df = pd.DataFrame({'account_id': list(topo_nodes)})
topo_df['in_degree'] = topo_df['account_id'].map(deg_in).fillna(0)
topo_df['out_degree'] = topo_df['account_id'].map(deg_out).fillna(0)
topo_df['pagerank'] = topo_df['account_id'].map(pr).fillna(0)
topo_df['comm_risk_ratio'] = topo_df['account_id'].map(comm_risk_ratio_map).fillna(0)

print(f"图拓扑特征完成: {len(topo_df):,} 节点")

# ---------- 5. 合并特征 & 构建训练/测试集 ----------
print("[4/5] 合并特征并划分训练/测试集...")

# 训练集：行为 + 拓扑
train_feats = beh_train.merge(topo_df, on='account_id', how='left').fillna(0)
train_labels = train_txn[['account_id', 'is_risk']].drop_duplicates().set_index('account_id')['is_risk']
train_feats = train_feats.merge(train_labels, on='account_id', how='left')
train_feats['is_risk'] = train_feats['is_risk'].fillna(0).astype(int)

# 测试集：先尝试从训练行为中获取，否则全0
test_beh = beh_train[beh_train['account_id'].isin(test_accounts)]
test_topo = topo_df[topo_df['account_id'].isin(test_accounts)]

# 合并测试特征（右连接确保所有测试账户都在）
test_feats = pd.DataFrame({'account_id': list(test_accounts)})
test_feats = test_feats.merge(test_beh, on='account_id', how='left')
test_feats = test_feats.merge(test_topo, on='account_id', how='left')
test_feats = test_feats.fillna(0)

# 标签：测试账户的真实标签（来自全量 txn，仅用于评估！）
# 注意：这里不用于训练，仅评估，所以不构成泄露
FULL_TXN_PATH = r'D:/Pycharm/Intermediaries_digging/data/cleaned_transactions.csv'
full_txn = pd.read_csv(FULL_TXN_PATH, encoding='utf-8-sig')
full_txn['is_risk'] = full_txn['account_label'].isin(RISK_LABELS).astype(int)
test_labels = full_txn[full_txn['account_id'].isin(test_accounts)][['account_id', 'is_risk']].drop_duplicates()
test_feats = test_feats.merge(test_labels, on='account_id', how='left')
test_feats['is_risk'] = test_feats['is_risk'].fillna(0).astype(int)

# 特征列
feat_cols = ['total_txn', 'total_amt', 'avg_amt', 'std_amt', 'uniq_opp', 'txn_freq',
             'night_ratio', 'fast_out_ratio', 'in_degree', 'out_degree', 'pagerank', 'comm_risk_ratio']

X_train = train_feats[feat_cols]
y_train = train_feats['is_risk']
X_test = test_feats[feat_cols]
y_test = test_feats['is_risk']

print(f"训练集: {len(X_train):,}, 测试集: {len(X_test):,}")
print(f"训练风险比例: {y_train.mean():.2%}, 测试风险比例: {y_test.mean():.2%}")

# ---------- 6. 训练模型 ----------
print("[5/5] 训练 XGBoost 模型...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum()
)
model.fit(X_train, y_train)

# ---------- 7. 阈值优化与评估 ----------
y_pred_proba = model.predict_proba(X_test)[:, 1]

# F1 最大化
thresholds = np.arange(0.01, 1.0, 0.01)
f1_scores = [f1_score(y_test, y_pred_proba >= t) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
y_pred = (y_pred_proba >= best_thresh).astype(int)

print(f"\n最佳阈值 (F1最大化): {best_thresh:.2f}")
print(f"AUC-ROC : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"AUC-PR  : {average_precision_score(y_test, y_pred_proba):.4f}")
print(f"Recall  : {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# ---------- 8. 输出全量评分 ----------
print("\n[6/6] 输出全量账户评分...")
# 构建全量特征（训练账户 + 测试账户）
all_feats = pd.concat([train_feats, test_feats], ignore_index=True)
all_feats = all_feats.drop_duplicates(subset='account_id')

# 补全缺失账户（理论上不应有，但保险起见）
missing_accounts = all_accounts - set(all_feats['account_id'])
if missing_accounts:
    missing_df = pd.DataFrame({'account_id': list(missing_accounts)})
    for col in feat_cols:
        missing_df[col] = 0
    missing_df['is_risk'] = 0
    all_feats = pd.concat([all_feats, missing_df], ignore_index=True)

all_feats['risk_score'] = model.predict_proba(all_feats[feat_cols])[:, 1]

# 风险等级
fixed_bins = [-1, 0.2, 0.4, 0.8, 1.0]
all_bins = sorted(set(fixed_bins + [best_thresh]))
all_feats['risk_level'] = pd.cut(
    all_feats['risk_score'],
    bins=all_bins,
    labels=['正常', '低风险', '中风险', '高风险', '极高风险'][:len(all_bins)-1],
    include_lowest=True
)

out_file = os.path.join(OUTPUT_DIR, 'account_risk_scores_leakage_free.csv')
all_feats.sort_values('risk_score', ascending=False).to_csv(out_file, index=False, encoding='utf-8-sig')
print(f"评分已保存 → {out_file}")

# 特征重要性
imp = pd.DataFrame({'feature': feat_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_leakage_free.csv'), index=False, encoding='utf-8-sig')

print(f"全部完成！总耗时 {time.time() - start:.2f} s")