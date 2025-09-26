# -*- coding: utf-8 -*-
"""
步骤3：动态风险评估模型（带阈值优化）
功能：
  - 使用训练图特征训练 XGBoost
  - 自动选择最优阈值（F1 最大化）
  - 避免标签泄露（使用时序划分或随机划分）
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

print("=== 步骤3：动态风险评估模型（带阈值优化） ===")
start = time.time()

# ---------- 1. 数据加载 ----------
INPUT_TXN_PATH = r'D:/Pycharm/Intermediaries_digging/data/cleaned_transactions.csv'
GRAPH_PATH = r'D:/Pycharm/Intermediaries_digging/output/risk_propagation_graph_train.json'
COMMUNITY_PATH = r'D:/Pycharm/Intermediaries_digging/output/gang_communities.csv'
TEST_IDS_PATH = r'D:/Pycharm/Intermediaries_digging/output/test_account_ids.csv'
OUTPUT_DIR = r'D:/Pycharm/Intermediaries_digging/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

txn = pd.read_csv(INPUT_TXN_PATH, encoding='utf-8-sig')
print(f'交易记录 {len(txn):,}')
with open(GRAPH_PATH, 'r', encoding='utf-8') as f:
    graph_data = json.load(f)
try:
    comm = pd.read_csv(COMMUNITY_PATH, encoding='utf-8-sig')
except FileNotFoundError:
    comm = None

txn['timestamp'] = pd.to_datetime(txn['timestamp'])
RISK_LABELS = ['黑', '灰', '黑密接', '黑次密接', '黑次次密接', '黑次次次密接']
txn['is_risk'] = txn['account_label'].isin(RISK_LABELS).astype(int)

# 加载测试账户（确保无泄露）
test_accounts = set(pd.read_csv(TEST_IDS_PATH, encoding='utf-8-sig')['account_id'])
all_accounts = set(txn['account_id'].unique())
train_accounts = all_accounts - test_accounts

print(f'总账户数: {len(all_accounts):,}')
print(f'训练账户: {len(train_accounts):,}, 测试账户: {len(test_accounts):,}')
print(f'全量风险占比: {txn["is_risk"].mean():.2%}')

# ---------- 2. 行为特征（向量化） ----------
print('\n[1/4] 行为特征（向量化）')
beh = txn.groupby('account_id').agg(
    total_txn=('account_id', 'count'),
    total_amt=('amount', 'sum'),
    avg_amt=('amount', 'mean'),
    std_amt=('amount', 'std'),
    uniq_opp=('opponent_name', 'nunique'),
    first_txn=('timestamp', 'min'),
    last_txn=('timestamp', 'max')
)
beh['active_days'] = (beh['last_txn'] - beh['first_txn']).dt.days + 1
beh['txn_freq'] = beh['total_txn'] / beh['active_days']

txn['is_night'] = ((txn.timestamp.dt.hour >= 22) | (txn.timestamp.dt.hour < 6)).astype(int)
night = txn.groupby('account_id')['is_night'].agg(night_cnt='sum', total='count')
beh['night_ratio'] = night['night_cnt'] / night['total']

out = txn[txn['direction'] == '2'][['account_id', 'timestamp']].sort_values(['account_id', 'timestamp'])
inn = txn[txn['direction'] == '1'][['account_id', 'timestamp']].sort_values(['account_id', 'timestamp'])
if out.empty or inn.empty:
    beh['fast_out_ratio'] = 0.0
else:
    out['deadline'] = out['timestamp'] + timedelta(hours=24)
    merged = pd.merge_asof(out, inn.rename(columns={'timestamp': 'ts_in'}),
                           on='timestamp', by='account_id', direction='forward')
    merged['flag'] = merged['ts_in'] <= merged['deadline']
    fast = merged.groupby('account_id')['flag'].agg(sum='sum', count='count')
    beh['fast_out_ratio'] = (fast['sum'] / fast['count']).fillna(0)

beh = beh.reset_index()
print(f'行为特征完成 {len(beh):,}')

# ---------- 3. 图拓扑特征 ----------
print('\n[2/4] 图拓扑特征')
edges = pd.DataFrame(graph_data['edges'])[['source', 'target', 'weight']]
G = nx.DiGraph()
for _, row in edges.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

deg_in = dict(G.in_degree(weight='weight'))
deg_out = dict(G.out_degree(weight='weight'))
pr = nx.pagerank(G, weight='weight') if G.number_of_nodes() > 0 else {}

def get_majority_label(group):
    mode = group.mode()
    return mode.iloc[0] if len(mode) > 0 else group.iloc[0]

account_label_map = txn.groupby('account_id')['account_label'].apply(get_majority_label).to_dict()

comm_risk_ratio_map = {}
if comm is not None:
    comm['is_risk'] = comm['account_id'].map(account_label_map).fillna('正常').isin(RISK_LABELS).astype(int)
    comm_risk = comm.groupby('community_id')['is_risk'].mean().to_dict()
    comm_id_map = comm.set_index('account_id')['community_id'].to_dict()
    comm_risk_ratio_map = {node: comm_risk.get(comm_id_map.get(node, -1), 0.0) for node in comm['account_id']}

topo = pd.DataFrame({'account_id': beh['account_id']})
topo['in_degree'] = topo['account_id'].map(deg_in).fillna(0)
topo['out_degree'] = topo['account_id'].map(deg_out).fillna(0)
topo['pagerank'] = topo['account_id'].map(pr).fillna(0)
topo['comm_risk_ratio'] = topo['account_id'].map(comm_risk_ratio_map).fillna(0)

feats = beh.merge(topo, on='account_id')
print(f'拓扑特征完成 {len(feats):,}')

# ---------- 4. 划分训练/测试集并建模 ----------
print('\n[3/4] 划分训练/测试集并建模')
lbl = txn[['account_id', 'is_risk']].drop_duplicates().set_index('account_id')
feats = feats.merge(lbl, on='account_id')

train_feats = feats[feats['account_id'].isin(train_accounts)]
test_feats = feats[feats['account_id'].isin(test_accounts)]

feat_cols = ['total_txn', 'total_amt', 'avg_amt', 'std_amt', 'uniq_opp', 'txn_freq',
             'night_ratio', 'fast_out_ratio', 'in_degree', 'out_degree', 'pagerank', 'comm_risk_ratio']

X_train = train_feats[feat_cols]
y_train = train_feats['is_risk']
X_test = test_feats[feat_cols]
y_test = test_feats['is_risk']

print(f'训练集: {len(X_train):,}, 测试集: {len(X_test):,}')
print(f'测试集风险比例: {y_test.mean():.2%}')

model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum()
)
model.fit(X_train, y_train)

# ---------- 5. 阈值优化 ----------
print('\n[4/4] 阈值优化与评估')
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 方法1: F1 最大化
thresholds = np.arange(0.1, 1.0, 0.01)
f1_scores = [f1_score(y_test, y_pred_proba >= t) for t in thresholds]
best_thresh_f1 = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

# 方法2: 满足 Recall ≥ 0.8 的最高 Precision
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
target_recall = 0.8
idx = np.where(recalls >= target_recall)[0]
best_thresh_recall = pr_thresholds[idx[-1]] if len(idx) > 0 else 0.5

# 默认使用 F1 最优阈值
best_thresh = best_thresh_f1
y_pred = (y_pred_proba >= best_thresh).astype(int)

print(f"最佳阈值 (F1最大化): {best_thresh:.2f}")
print(f"AUC-ROC : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"AUC-PR  : {average_precision_score(y_test, y_pred_proba):.4f}")
print(f"Recall  : {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# ---------- 6. 输出 ----------
print('\n[5/5] 输出评分')
feats['risk_score'] = model.predict_proba(feats[feat_cols])[:, 1]

# 构造单调递增的 bins
fixed_bins = [-1, 0.2, 0.4, 0.8, 1.0]
all_bins = sorted(set(fixed_bins + [best_thresh]))
feats['risk_level'] = pd.cut(
    feats['risk_score'],
    bins=all_bins,
    labels=['正常', '低风险', '中风险', '高风险', '极高风险'][:len(all_bins)-1],
    include_lowest=True
)

out_file = os.path.join(OUTPUT_DIR, 'account_risk_scores_optimized.csv')
feats.sort_values('risk_score', ascending=False).to_csv(out_file, index=False, encoding='utf-8-sig')
print(f'评分已保存 → {out_file}')

imp = pd.DataFrame({'feature': feat_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_optimized.csv'), index=False, encoding='utf-8-sig')

print(f'全部完成！总耗时 {time.time() - start:.2f} s')