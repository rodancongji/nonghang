# -*- coding: utf-8 -*-
"""
步骤2：全量图团伙发掘 + 训练图特征构建
功能：
  - **用全量数据构建图 H（用于团伙发掘）**
  - **用训练交易构建图 G_train（用于 XGBoost 图特征）**
  - 保留中介识别、高危对手方输出等所有功能
"""
import os, time, json, numpy as np, pandas as pd, networkx as nx, igraph as ig, leidenalg as la
from collections import defaultdict

print("=== 步骤2：全量图团伙发掘 + 训练图特征构建 ===")
start = time.time()

FULL_TXN_PATH = 'D:/Pycharm/Intermediaries_digging/data/cleaned_transactions.csv'
TRAIN_TXN_PATH = 'D:/Pycharm/Intermediaries_digging/data/train_transactions.csv'
TEST_IDS_PATH = 'D:/Pycharm/Intermediaries_digging/output/test_account_ids.csv'
OUTPUT_DIR = 'D:/Pycharm/Intermediaries_digging/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOISE = ['微信转账', '扫二维码付款', '微信红包', '还款', '手机充值', '零钱通', '生活缴费', '我的钱包', '待报解预算收入']
RISK_LABELS = ['黑', '灰', '黑密接', '黑次密接', '黑次次密接', '黑次次次密接']

# ==================== 1. 加载数据 ====================
print("\n[1/7] 加载全量和训练交易数据...")
full_txn = pd.read_csv(FULL_TXN_PATH, encoding='utf-8-sig')
train_txn = pd.read_csv(TRAIN_TXN_PATH, encoding='utf-8-sig')
test_accounts = set(pd.read_csv(TEST_IDS_PATH, encoding='utf-8-sig')['account_id'])
train_accounts = set(train_txn['account_id'].unique())

risk_account_ids_full = set(full_txn[full_txn['account_label'].isin(RISK_LABELS)]['account_id'].unique())
risk_account_ids_train = set(train_txn[train_txn['account_label'].isin(RISK_LABELS)]['account_id'].unique())

print(f"全量风险账户: {len(risk_account_ids_full):,}, 训练风险账户: {len(risk_account_ids_train):,}")

# ==================== 2. 构建全量图 H（用于团伙发掘） ====================
print("\n[2/7] 构建全量图 H（用于团伙发掘）...")
full_txn_clean = full_txn[~full_txn['opponent_name'].isin(NOISE)].copy()
edges_full = full_txn_clean.copy()
edges_full['weight'] = np.log1p(edges_full['amount'])
edges_full['src'] = np.where(edges_full['direction'] == '2', edges_full['account_id'], edges_full['opponent_name'])
edges_full['tgt'] = np.where(edges_full['direction'] == '2', edges_full['opponent_name'], edges_full['account_id'])

H = nx.DiGraph()
for _, row in edges_full.iterrows():
    H.add_edge(row['src'], row['tgt'], weight=row['weight'], amount=row['amount'])

print(f"全量图 H：{H.number_of_nodes():,} 节点, {H.number_of_edges():,} 边")

# ==================== 3. 全量图 2-hop 子图 + 团伙发掘 ====================
print("\n[3/7] 全量图 2-hop 子图 + 团伙发掘...")
all_nodes = set()
for node in risk_account_ids_full:
    if node in H:
        all_nodes.add(node)
        neighbors = set(H.successors(node)) | set(H.predecessors(node))
        all_nodes.update(neighbors)
        for nb in neighbors:
            if nb in H:
                nb_neighbors = set(H.successors(nb)) | set(H.predecessors(nb))
                all_nodes.update(nb_neighbors)

H_sub = H.subgraph(all_nodes).copy()
print(f"2-hop 子图：{H_sub.number_of_nodes():,} 节点, {H_sub.number_of_edges():,} 边")

community_df = pd.DataFrame()
if H_sub.number_of_nodes() > 0:
    g_ig = ig.Graph.from_networkx(H_sub)
    if 'weight' not in g_ig.edge_attributes():
        g_ig.es['weight'] = [1.0] * g_ig.ecount()

    best_mod = -1
    best_part = None
    for res in [0.1, 0.5, 1.0, 1.5, 2.0]:
        try:
            part = la.find_partition(g_ig, la.RBConfigurationVertexPartition, resolution_parameter=res,
                                     weights=g_ig.es['weight'])
            if part.modularity > best_mod:
                best_mod, best_part = part.modularity, part
        except:
            continue

    if best_part:
        node_list = list(H_sub.nodes())
        comm_map = dict(zip(node_list, best_part.membership))
        community_list = [{'account_id': node, 'community_id': cid} for node, cid in comm_map.items()]
        community_df = pd.DataFrame(community_list)

    node_label_map = full_txn.set_index('account_id')['account_label'].to_dict()
    opponent_label_map = full_txn.set_index('opponent_name')['counterparty_label'].to_dict()


    def get_label(node):
        return node_label_map.get(node, opponent_label_map.get(node, '未知'))


    community_df['account_label'] = community_df['account_id'].apply(get_label)
    community_stats = community_df.groupby('community_id').agg(
        total_nodes=('account_id', 'count'),
        risk_nodes=('account_label', lambda x: x.isin(RISK_LABELS).sum()),
        risk_ratio=('account_label', lambda x: x.isin(RISK_LABELS).mean())
    ).sort_values('risk_ratio', ascending=False)

    community_df.to_csv(os.path.join(OUTPUT_DIR, 'gang_communities.csv'), index=False, encoding='utf-8-sig')
    community_stats.to_csv(os.path.join(OUTPUT_DIR, 'gang_community_stats.csv'), index=False, encoding='utf-8-sig')
    print(f"团伙发掘完成，共 {len(community_stats)} 个社区")

# ==================== 4. 构建训练图 G_train（用于 XGBoost 特征） ====================
print("\n[4/7] 构建训练图 G_train（用于 XGBoost 特征）...")
train_txn_clean = train_txn[~train_txn['opponent_name'].isin(NOISE)].copy()
train_txn_clean = train_txn_clean[
    (train_txn_clean['account_label'].isin(RISK_LABELS)) |
    (train_txn_clean['counterparty_label'].isin(RISK_LABELS))
    ].copy()

edges_train = train_txn_clean.copy()
edges_train['weight'] = np.log1p(edges_train['amount'])
edges_train['src'] = np.where(edges_train['direction'] == '2', edges_train['account_id'], edges_train['opponent_name'])
edges_train['tgt'] = np.where(edges_train['direction'] == '2', edges_train['opponent_name'], edges_train['account_id'])

G_train = nx.DiGraph()
for _, row in edges_train.iterrows():
    G_train.add_edge(row['src'], row['tgt'], weight=row['weight'], amount=row['amount'])

print(f"训练图 G_train：{G_train.number_of_nodes():,} 节点, {G_train.number_of_edges():,} 边")

# ==================== 5. 识别无标签中介账户（基于训练图） ====================
print("\n[5/7] 识别无标签中介账户（基于训练图）...")
intermediaries_df = pd.DataFrame()
if G_train.number_of_nodes() > 0:
    weighted_in_degree = dict(G_train.in_degree(weight='weight'))
    weighted_out_degree = dict(G_train.out_degree(weight='weight'))
    try:
        pagerank = nx.pagerank(G_train, weight='weight')
    except:
        pagerank = {node: 0.0 for node in G_train.nodes()}

    path_critical_nodes = defaultdict(int)
    risk_set = risk_account_ids_train
    for u in risk_set:
        if u in G_train:
            for v in G_train.successors(u):
                path_critical_nodes[v] += 1
            for v in G_train.predecessors(u):
                path_critical_nodes[v] += 1

    nodes_list = list(G_train.nodes())
    node_df = pd.DataFrame({
        'account_id': nodes_list,
        'weighted_in_degree': [weighted_in_degree.get(node, 0) for node in nodes_list],
        'pagerank': [pagerank.get(node, 0) for node in nodes_list],
        'path_count': [path_critical_nodes.get(node, 0) for node in nodes_list],
        'node_type': ['account' if node in risk_set else 'external' for node in nodes_list],
    })

    for col in ['weighted_in_degree', 'pagerank', 'path_count']:
        col_min, col_max = node_df[col].min(), node_df[col].max()
        if col_max > col_min:
            node_df[f'{col}_norm'] = (node_df[col] - col_min) / (col_max - col_min)
        else:
            node_df[f'{col}_norm'] = 0.0

    node_df['intermediary_score'] = (
            node_df['weighted_in_degree_norm'] * 0.4 +
            node_df['path_count_norm'] * 0.4 +
            node_df['pagerank_norm'] * 0.2
    )

    candidates = node_df[(node_df['node_type'] == 'external') & (node_df['path_count'] > 0)].sort_values(
        'intermediary_score', ascending=False)
    intermediaries_df = candidates.head(100).copy()
    intermediaries_df.to_csv(os.path.join(OUTPUT_DIR, 'unlabeled_intermediaries.csv'), index=False,
                             encoding='utf-8-sig')
    print(f"无标签中介账户已保存，共 {len(intermediaries_df)} 个")

# ==================== 6. 保存训练图结构 ====================
print("\n[6/7] 保存训练图结构（用于 XGBoost）...")
if G_train.number_of_nodes() > 0:
    community_id_map = community_df.set_index('account_id')['community_id'].to_dict() if not community_df.empty else {}
    intermediary_score_map = intermediaries_df.set_index('account_id')[
        'intermediary_score'].to_dict() if not intermediaries_df.empty else {}
    node_label_map = train_txn.set_index('account_id')['account_label'].to_dict()
    opponent_label_map = train_txn.set_index('opponent_name')['counterparty_label'].to_dict()

    nodes_data = []
    for node in G_train.nodes():
        label = node_label_map.get(node, opponent_label_map.get(node, '未知'))
        comm_id = community_id_map.get(node, -1)
        inter_score = intermediary_score_map.get(node, 0.0)
        node_type = 'account' if node in risk_account_ids_train else 'external'
        nodes_data.append({
            'id': node,
            'account_label': label,
            'node_type': node_type,
            'community_id': int(comm_id),
            'intermediary_score': float(inter_score),
            'weighted_in_degree': float(weighted_in_degree.get(node, 0.0)),
            'pagerank': float(pagerank.get(node, 0.0)),
            'path_count': int(path_critical_nodes.get(node, 0)),
        })

    edges_data = []
    for u, v, data in G_train.edges(data=True):
        edges_data.append({
            'source': u, 'target': v,
            'weight': float(data.get('weight', 0.0)),
            'amount': float(data.get('amount', 0.0)),
            'timestamp': data.get('timestamp', None)
        })

    with open(os.path.join(OUTPUT_DIR, 'risk_propagation_graph_train.json'), 'w', encoding='utf-8') as f:
        json.dump({"nodes": nodes_data, "edges": edges_data}, f, ensure_ascii=False, indent=2)
    print("训练图结构已保存。")

# ==================== 7. 输出高危对手方 Top 100 ====================
print("\n[7/7] 输出高危对手方 Top 100...")
risk_flows = train_txn_clean[train_txn_clean['account_label'].isin(RISK_LABELS)]
top_opponents = risk_flows.groupby('opponent_name').agg(
    transaction_count=('opponent_name', 'count'),
    total_amount=('amount', 'sum'),
    unique_risk_accounts=('account_id', 'nunique')
).sort_values('transaction_count', ascending=False).head(100)
top_opponents.to_csv(os.path.join(OUTPUT_DIR, 'high_risk_opponents.csv'), index=True, encoding='utf-8-sig')
print("高危对手方Top100已保存。")

print(f"步骤2完成！总耗时 {time.time() - start:.2f} s")