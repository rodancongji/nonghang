import os
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
from cdlib import algorithms
import igraph as ig
import leidenalg as la
from collections import Counter, defaultdict
import warnings

print("=== 步骤2（整合版）：构建风险传播图 + 团伙发掘 + 无标签中介识别 ===")

start_time_total = time.time()

# ==================== 1. 定义路径和参数 ====================
INPUT_TXN_PATH = 'D:/Pycharm/Intermediaries_digging/data/cleaned_transactions.csv'
OUTPUT_DIR = 'D:/Pycharm/Intermediaries_digging/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOISE_NAMES = [
    '微信转账', '扫二维码付款', '微信红包', '还款', '手机充值', '零钱通',
    '生活缴费', '我的钱包', '待报解预算收入'
]

RISK_LABELS = ['黑', '灰', '黑密接', '黑次密接', '黑次次密接', '黑次次次密接']

# ==================== 2. 加载并预处理数据 ====================
print("\n[1/6] 正在加载并预处理交易数据...")

txn_df = pd.read_csv(INPUT_TXN_PATH, encoding='utf-8-sig')
print(f"加载完成，共 {len(txn_df):,} 条交易记录。")

risk_account_ids = set(txn_df[txn_df['account_label'].isin(RISK_LABELS)]['account_id'].unique())
print(f"风险账户数（发起方）：{len(risk_account_ids):,}")

print(f"  → 过滤前交易数: {len(txn_df):,}")
txn_df = txn_df[~txn_df['opponent_name'].isin(NOISE_NAMES)].copy()
print(f"  → 过滤噪声对手方后交易数: {len(txn_df):,}")

txn_df = txn_df[
    (txn_df['account_label'].isin(RISK_LABELS)) |
    (txn_df['counterparty_label'].isin(RISK_LABELS))
].copy()
print(f"  → 聚焦风险账户直接关联交易后交易数: {len(txn_df):,}")

# ==================== 3. 构建有向风险传播图 ====================
print("\n[2/6] 构建有向风险传播图...")

edges = txn_df.copy()
edges['weight'] = np.log1p(edges['amount'])

condition = edges['direction'] == '2'
edges['src'] = np.where(condition, edges['account_id'], edges['opponent_name'])
edges['tgt'] = np.where(condition, edges['opponent_name'], edges['account_id'])

G = nx.DiGraph()
edge_tuples = edges[['src', 'tgt', 'weight', 'amount', 'timestamp', 'direction']].to_dict('records')
G.add_edges_from([(e['src'], e['tgt'], {
    'weight': e['weight'],
    'amount': e['amount'],
    'timestamp': e['timestamp'],
    'direction': e['direction']
}) for e in edge_tuples])

print(f"  → 图构建完成。")
print(f"    节点总数: {G.number_of_nodes():,}")
print(f"    边总数: {G.number_of_edges():,}")

# ==================== 4. 团伙发掘 ====================
print("\n[3/6] 在风险子图上进行团伙发掘 (Leiden Algorithm)...")

community_df = pd.DataFrame()
community_stats = pd.DataFrame()

if G.number_of_nodes() > 0:
    print("  → 构建风险子图...")
    risk_nodes = set(risk_account_ids)
    two_hop_nodes = set(risk_nodes)
    for node in risk_nodes:
        if node in G:
            one_hop = set(G.successors(node)) | set(G.predecessors(node))
            two_hop_nodes.update(one_hop)
            for neighbor in one_hop:
                if neighbor in G:
                    two_hop = set(G.successors(neighbor)) | set(G.predecessors(neighbor))
                    two_hop_nodes.update(two_hop)

    H = G.subgraph(two_hop_nodes).copy()
    print(f"  → 风险子图构建完成，节点数: {H.number_of_nodes():,}, 边数: {H.number_of_edges():,}")

    if H.number_of_nodes() == 0:
        print("  → 风险子图为空，跳过团伙发掘。")
    else:
        print(" → 转换为 igraph 格式...")
        g_ig = ig.Graph.from_networkx(H)

        if 'weight' not in g_ig.edge_attributes():
            g_ig.es['weight'] = [1.0] * g_ig.ecount()
            print("  → 图中无'weight'属性，使用默认权重1.0")

        print("  → 网格搜索最优分辨率...")
        resolutions = [0.1, 0.5, 1.0, 1.5, 2.0]
        best_resolution = 1.0
        best_modularity = -1
        best_partition = None

        for res in resolutions:
            try:
                partition = la.find_partition(
                    g_ig,
                    la.RBConfigurationVertexPartition,
                    resolution_parameter=res,
                    weights=g_ig.es['weight']
                )
                modularity = partition.modularity
                print(f"    分辨率 {res}: 模块度 {modularity:.4f}")
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_resolution = res
                    best_partition = partition
            except Exception as e:
                print(f"    分辨率 {res} 失败: {e}")
                continue

        if best_partition is None:
            print("  → Leiden 算法失败，使用标签传播算法作为备选...")
            communities = nx.algorithms.community.asyn_lpa_communities(H, weight='weight')
        else:
            print(f"  → 最佳分辨率: {best_resolution}, 模块度: {best_modularity:.4f}")
            node_list = list(H.nodes())
            community_map = dict(zip(node_list, best_partition.membership))
            communities = []
            for comm_id in set(best_partition.membership):
                comm_nodes = [node for node, cid in community_map.items() if cid == comm_id]
                communities.append(set(comm_nodes))

        print("  → 处理社区结果...")
        community_list = []
        for i, comm in enumerate(communities):
            for node in comm:
                community_list.append({
                    'account_id': node,
                    'community_id': i
                })

        community_df = pd.DataFrame(community_list)

        node_label_map = txn_df.set_index('account_id')['account_label'].to_dict()
        opponent_label_map = txn_df.set_index('opponent_name')['counterparty_label'].to_dict()

        def get_node_label(node):
            if node in node_label_map:
                return node_label_map[node]
            elif node in opponent_label_map:
                return opponent_label_map[node]
            else:
                return '未知'

        community_df['account_label'] = community_df['account_id'].apply(get_node_label)

        community_stats = community_df.groupby('community_id').agg(
            total_nodes=('account_id', 'count'),
            risk_nodes=('account_label', lambda x: x.isin(RISK_LABELS).sum()),
            risk_ratio=('account_label', lambda x: x.isin(RISK_LABELS).mean())
        ).sort_values('risk_ratio', ascending=False)

        print(f"  → 共发现 {len(community_stats)} 个团伙社区")
        print(f"  → Top 10 高风险团伙社区：")
        top_communities = community_stats.head(10)
        print(top_communities.to_string())

        community_df.to_csv(os.path.join(OUTPUT_DIR, 'gang_communities.csv'), index=False, encoding='utf-8-sig')
        community_stats.to_csv(os.path.join(OUTPUT_DIR, 'gang_community_stats.csv'), index=False, encoding='utf-8-sig')
        print("  → 团伙发掘结果已保存。")
else:
    print(" 图为空，跳过团伙发掘。")

# ==================== 5. 识别无标签中介账户 ====================
print("\n[4/6] 识别无标签中介账户（基于结构中心性和路径分析）...")

intermediaries_df = pd.DataFrame()

if G.number_of_nodes() > 0:
    print(" → 计算中心性指标...")
    weighted_in_degree = dict(G.in_degree(weight='weight'))
    weighted_out_degree = dict(G.out_degree(weight='weight'))
    try:
        pagerank = nx.pagerank(G, weight='weight', max_iter=100, tol=1e-06)
    except:
        pagerank = {node: 0.0 for node in G.nodes()}
        print(" PageRank计算失败，使用默认值0.0")

    # 高效计算 path_count：避免嵌套 for
    print(" → 高效识别路径关键节点（中间跳）...")
    path_critical_nodes = defaultdict(int)

    # 获取所有风险账户
    risk_set = risk_account_ids

    # 步骤1: 风险账户出边 → 第一跳中介
    first_hop_out = set()
    for u in risk_set:
        if u in G:
            first_hop_out.update(G.successors(u))

    # 步骤2: 风险账户入边 ← 第一跳中介（反向）
    first_hop_in = set()
    for u in risk_set:
        if u in G:
            first_hop_in.update(G.predecessors(u))

    # 合并所有潜在中介节点
    candidate_mediators = first_hop_out | first_hop_in

    # 步骤3: 对每个候选中介 v，统计它连接多少风险路径
    for v in candidate_mediators:
        # 出向路径: risk -> v -> x
        in_from_risk = len([u for u in G.predecessors(v) if u in risk_set])
        out_to_any = G.out_degree(v)
        path_critical_nodes[v] += in_from_risk * max(1, out_to_any)  # 粗略估计路径数

        # 入向路径: x -> v -> risk
        out_to_risk = len([w for w in G.successors(v) if w in risk_set])
        in_from_any = G.in_degree(v)
        path_critical_nodes[v] += out_to_risk * max(1, in_from_any)

    # 构建节点特征DataFrame
    nodes_list = list(G.nodes())
    node_df = pd.DataFrame({
        'account_id': nodes_list,
        'weighted_in_degree': [weighted_in_degree.get(node, 0) for node in nodes_list],
        'weighted_out_degree': [weighted_out_degree.get(node, 0) for node in nodes_list],
        'pagerank': [pagerank.get(node, 0) for node in nodes_list],
        'path_count': [path_critical_nodes.get(node, 0) for node in nodes_list],
        'node_type': ['account' if node in risk_account_ids else 'external' for node in nodes_list],
    })

    # 特征归一化
    for col in ['weighted_in_degree', 'weighted_out_degree', 'pagerank', 'path_count']:
        col_min, col_max = node_df[col].min(), node_df[col].max()
        if col_max > col_min:
            node_df[f'{col}_norm'] = (node_df[col] - col_min) / (col_max - col_min)
        else:
            node_df[f'{col}_norm'] = 0.0

    # 综合评分
    node_df['intermediary_score'] = (
        node_df['weighted_in_degree_norm'] * 0.4 +
        node_df['path_count_norm'] * 0.4 +
        node_df['pagerank_norm'] * 0.2
    )

    # 筛选无标签中介候选：external + path_count > 0
    candidates = node_df[
        (node_df['node_type'] == 'external') &
        (node_df['path_count'] > 0)
    ].sort_values('intermediary_score', ascending=False)

    intermediaries_df = candidates.head(100).copy()

    print(f"  → 识别出 {len(intermediaries_df):,} 个无标签中介账户候选")
    if len(intermediaries_df) > 0:
        print(f"\n=== Top-10 无标签中介账户 ===")
        top10 = intermediaries_df.head(10)[['account_id', 'intermediary_score', 'weighted_in_degree', 'path_count']]
        print(top10.to_string(index=False))

        print(f"\n=== 为 Top 3 中介账户输出其关联风险路径 ===")
        for idx, row in intermediaries_df.head(3).iterrows():
            core_node = row['account_id']
            print(f"\n中介账户: {core_node} (评分: {row['intermediary_score']:.4f})")
            risk_paths = []
            for risk_node in risk_account_ids:
                if risk_node in G:
                    if core_node in G.successors(risk_node):
                        risk_paths.append(f"{risk_node} -> {core_node}")
                    for neighbor in G.successors(risk_node):
                        if core_node in G.successors(neighbor):
                            risk_paths.append(f"{risk_node} -> {neighbor} -> {core_node}")
            print(f"  → 关联风险路径数: {len(risk_paths)}")
            if len(risk_paths) > 0:
                print(f"  → 前3条风险路径: {risk_paths[:3]}")

    if not intermediaries_df.empty:
        intermediaries_df.to_csv(os.path.join(OUTPUT_DIR, 'unlabeled_intermediaries.csv'), index=False, encoding='utf-8-sig')
        print(f"  → 无标签中介账户已保存至: unlabeled_intermediaries.csv")
else:
    print(" 图为空，跳过中介账户识别。")

# ==================== 6. 保存图结构（含 timestamp） ====================
print("\n[5/6] 保存图结构（含团伙ID、中介分数和 timestamp）...")

if G.number_of_nodes() > 0:
    community_id_map = community_df.set_index('account_id')['community_id'].to_dict() if not community_df.empty else {}
    intermediary_score_map = intermediaries_df.set_index('account_id')['intermediary_score'].to_dict() if not intermediaries_df.empty else {}

    node_label_map = txn_df.set_index('account_id')['account_label'].to_dict()
    opponent_label_map = txn_df.set_index('opponent_name')['counterparty_label'].to_dict()

    # 确保 path_critical_nodes 存在
    if 'path_critical_nodes' not in locals():
        path_critical_nodes = defaultdict(int)

    nodes_data = []
    for node in G.nodes():
        label = node_label_map.get(node, opponent_label_map.get(node, '未知'))
        comm_id = community_id_map.get(node, -1)
        inter_score = intermediary_score_map.get(node, 0.0)
        node_type = 'account' if node in risk_account_ids else 'external'
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
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'source': u,
            'target': v,
            'weight': float(data.get('weight', 0.0)),
            'amount': float(data.get('amount', 0.0)),
            'timestamp': data.get('timestamp', None),  # ✅ 保留 timestamp
        })

    graph_data = {
        "nodes": nodes_data,
        "edges": edges_data,
        "metadata": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "gang_communities_count": len(community_stats) if not community_stats.empty else 0,
            "intermediaries_count": len(intermediaries_df),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'risk_propagation_graph.json'), 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f" 图结构已保存至：{os.path.join(OUTPUT_DIR, 'risk_propagation_graph.json')}")

# ==================== 7. 输出高危对手方 Top 100 ====================
print("\n[6/6] 输出高危对手方 Top 100...")

risk_flows = txn_df[txn_df['account_label'].isin(RISK_LABELS)]
top_opponents = risk_flows.groupby('opponent_name').agg(
    transaction_count=('opponent_name', 'count'),
    total_amount=('amount', 'sum'),
    unique_risk_accounts=('account_id', 'nunique')
).sort_values('transaction_count', ascending=False).head(100)

top_opponents.to_csv(os.path.join(OUTPUT_DIR, 'high_risk_opponents.csv'), index=True, encoding='utf-8-sig')
print(f"  → 高危对手方Top100已保存。")

print(f"\n 步骤2（整合版）完成！总运行时间: {time.time() - start_time_total:.2f} 秒")