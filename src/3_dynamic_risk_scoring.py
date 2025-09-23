# -*- coding: utf-8 -*-
"""
步骤3：动态风险评估模型
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, recall_score
import networkx as nx
import os, json, warnings, time

warnings.filterwarnings('ignore')
print("=== 步骤3：动态风险评估模型 ===")
start = time.time()

# ---------- 1. 数据加载 ----------
INPUT_TXN_PATH = r'D:/Pycharm/Intermediaries_digging/data/cleaned_transactions.csv'
GRAPH_PATH = r'D:/Pycharm/Intermediaries_digging/output/risk_propagation_graph.json'
COMMUNITY_PATH = r'D:/Pycharm/Intermediaries_digging/output/gang_communities.csv'
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
print(f'风险占比 {txn["is_risk"].mean():.2%}')

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

# 夜间比例
txn['is_night'] = ((txn.timestamp.dt.hour >= 22) | (txn.timestamp.dt.hour < 6)).astype(int)
night = txn.groupby('account_id')['is_night'].agg(night_cnt='sum', total='count')
beh['night_ratio'] = night['night_cnt'] / night['total']

# 快进快出
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
edges['data'] = edges[['weight']].apply(lambda x: {'weight': x[0]}, axis=1)
G = nx.DiGraph()
G.add_edges_from(edges[['source', 'target', 'data']].itertuples(index=False))

# 批量计算中心性
deg_in = dict(G.in_degree(weight='weight'))
deg_out = dict(G.out_degree(weight='weight'))
pr = nx.pagerank(G, weight='weight')

topo = pd.DataFrame({'account_id': beh['account_id']})
topo['in_degree'] = topo['account_id'].map(deg_in).fillna(0)
topo['out_degree'] = topo['account_id'].map(deg_out).fillna(0)
topo['pagerank'] = topo['account_id'].map(pr).fillna(0)
topo['betweenness'] = 0.0   # 可后续加采样

# 社区风险比例
if comm is not None:
    node_lbl = txn[['account_id', 'account_label']].drop_duplicates().set_index('account_id')['account_label']
    comm['is_risk'] = comm['account_id'].map(node_lbl).isin(RISK_LABELS).astype(int)
    comm_risk = comm.groupby('community_id')['is_risk'].mean().reset_index()
    comm_risk.columns = ['community_id', 'comm_risk_ratio']
    comm = comm.merge(comm_risk, on='community_id', how='left')
    topo = topo.merge(comm[['account_id', 'comm_risk_ratio']].drop_duplicates(),
                      on='account_id', how='left')
    topo['comm_risk_ratio'] = topo['comm_risk_ratio'].fillna(0)
else:
    topo['comm_risk_ratio'] = 0.0

feats = beh.merge(topo, on='account_id')
print(f'拓扑特征完成 {len(feats):,}')

# ---------- 4. 建模 ----------
print('\n[3/4] 建模')
feat_cols = ['total_txn', 'total_amt', 'avg_amt', 'std_amt', 'uniq_opp', 'txn_freq',
             'night_ratio', 'fast_out_ratio', 'in_degree', 'out_degree', 'pagerank', 'comm_risk_ratio']
lbl = txn[['account_id', 'is_risk']].drop_duplicates().set_index('account_id')
feats = feats.merge(lbl, left_on='account_id', right_index=True)

X_train, X_test, y_train, y_test = train_test_split(
    feats[feat_cols], feats['is_risk'],
    test_size=0.2, stratify=feats['is_risk'], random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum()
)
model.fit(X_train, y_train)

y_pred_p = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
print(f'AUC  {roc_auc_score(y_test, y_pred_p):.4f}')
print(f'Recall {recall_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))

# ---------- 5. 输出 ----------
print('\n[4/4] 输出评分')
feats['risk_score'] = model.predict_proba(feats[feat_cols])[:, 1]

def level(s):
    if s > 0.8:
        return '极高风险'
    elif s > 0.6:
        return '高风险'
    elif s > 0.4:
        return '中风险'
    elif s > 0.2:
        return '低风险'
    return '正常'

feats['risk_level'] = feats['risk_score'].apply(level)
out_file = os.path.join(OUTPUT_DIR, 'account_risk_scores.csv')
feats.sort_values('risk_score', ascending=False).to_csv(out_file, index=False, encoding='utf-8-sig')
print(f'评分已保存 → {out_file}')

imp = pd.DataFrame({'feature': feat_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False, encoding='utf-8-sig')
print(f'全部完成！总耗时 {time.time() - start:.2f} s')