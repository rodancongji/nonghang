# -*- coding: utf-8 -*-
"""
步骤2（无标签中介识别版）：构建风险传播图 + 识别无标签中介账户 + 输出资金路径
功能：
  - 构建风险传播图
  - 识别无标签中介账户（基于结构中心性和路径分析）
  - 输出高危资金路径和中介账户报告
  - 保存图结构和分析结果
"""

import os
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings('ignore')

print("=== 步骤2：构建风险传播图 + 识别无标签中介账户 ===")

start_time_total = time.time()

# ==================== 1. 定义路径和参数 ====================
INPUT_TXN_PATH = 'D:/Pycharm/Intermediaries_digging/data/cleaned_transactions.csv'
OUTPUT_DIR = 'D:/Pycharm/Intermediaries_digging/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义需要过滤的对手方名称（聚合通道或无意义名称）
NOISE_NAMES = [
    '微信转账', '扫二维码付款', '微信红包', '还款', '手机充值', '零钱通',
    '生活缴费', '我的钱包', '待报解预算收入'
]

RISK_LABELS = ['黑', '灰', '黑密接', '黑次密接', '黑次次密接', '黑次次次密接']

# ==================== 2. 加载并预处理数据 ====================
print("\n[1/5] 正在加载并预处理交易数据...")

txn_df = pd.read_csv(INPUT_TXN_PATH, encoding='utf-8-sig')
print(f"加载完成，共 {len(txn_df):,} 条交易记录。")

# 构建风险账户集合
risk_account_ids = set(txn_df[txn_df['account_label'].isin(RISK_LABELS)]['account_id'].unique())
print(f"风险账户数（发起方）：{len(risk_account_ids):,}")

# 过滤噪声对手方
print(f"  → 过滤前交易数: {len(txn_df):,}")
txn_df = txn_df[~txn_df['opponent_name'].isin(NOISE_NAMES)].copy()
print(f"  → 过滤噪声对手方后交易数: {len(txn_df):,}")

# 聚焦风险相关交易（但保留所有对手方，因为中介账户可能在其中）
txn_df = txn_df[
    (txn_df['account_label'].isin(RISK_LABELS)) |
    (txn_df['counterparty_label'].isin(RISK_LABELS))
].copy()
print(f"  → 聚焦风险账户直接关联交易后交易数: {len(txn_df):,}")

# ==================== 3. 构建有向风险传播图 ====================
print("\n[2/5] 构建有向风险传播图...")

# 创建边DataFrame
edges = txn_df.copy()
# 使用对数作为边权重，避免原始金额过大
edges['weight'] = np.log1p(edges['amount'])

# 定义源和目标节点
condition = edges['direction'] == '2'
edges['src'] = np.where(condition, edges['account_id'], edges['opponent_name'])
edges['tgt'] = np.where(condition, edges['opponent_name'], edges['account_id'])

# 创建图
G = nx.DiGraph()
edge_tuples = edges[['src', 'tgt', 'weight', 'amount', 'timestamp', 'direction']].to_dict('records')
G.add_edges_from([(e['src'], e['tgt'], {
    'weight': e['weight'],
    'amount': e['amount'],
    'timestamp': e['timestamp'],
    'direction': e['direction']
}) for e in edge_tuples])

print(f"  → 图构建完成。")
print(f" 节点总数: {G.number_of_nodes():,}")
print(f" 边总数: {G.number_of_edges():,}")

# ==================== 4. 识别无标签中介账户 ====================
print("\n[3/5] 识别无标签中介账户（基于结构中心性和路径分析）...")

if G.number_of_nodes() == 0:
    print(" 图为空，跳过中介账户识别。")
    intermediaries_df = pd.DataFrame()
else:
    # 4.1 计算关键中心性指标
    print(" → 计算中心性指标...")
    # 加权入度（识别资金归集点）
    weighted_in_degree = {node: sum(data['weight'] for _, _, data in G.in_edges(node, data=True)) for node in G.nodes()}
    # 加权出度（识别资金分发点）
    weighted_out_degree = {node: sum(data['weight'] for _, _, data in G.out_edges(node, data=True)) for node in G.nodes()}
    # PageRank（识别全局影响力）
    try:
        pagerank = nx.pagerank(G, weight='weight', max_iter=100, tol=1e-06)
    except:
        pagerank = {node: 0.0 for node in G.nodes()}
        print("  → PageRank计算失败，使用默认值0.0")

    # 4.2 识别“路径关键节点”
    print("  → 识别路径关键节点...")
    # 从所有风险账户出发，进行BFS，寻找1-hop和2-hop路径
    path_critical_nodes = defaultdict(int)  # 节点 -> 经过的路径数
    for risk_node in risk_account_ids:
        if risk_node in G:
            # 1-hop路径
            for neighbor in G.successors(risk_node):
                path_critical_nodes[neighbor] += 1
            # 2-hop路径
            for neighbor in G.successors(risk_node):
                for next_neighbor in G.successors(neighbor):
                    path_critical_nodes[next_neighbor] += 1

    # 4.3 构建候选中介账户列表
    nodes_list = list(G.nodes())
    node_df = pd.DataFrame({
        'account_id': nodes_list,
        'weighted_in_degree': [weighted_in_degree.get(node, 0) for node in nodes_list],
        'weighted_out_degree': [weighted_out_degree.get(node, 0) for node in nodes_list],
        'pagerank': [pagerank.get(node, 0) for node in nodes_list],
        'path_count': [path_critical_nodes.get(node, 0) for node in nodes_list],
        'node_type': ['account' if node in risk_account_ids else 'external' for node in nodes_list],
    })

    # 4.4 计算综合中介分数
    # 特征归一化
    for col in ['weighted_in_degree', 'weighted_out_degree', 'pagerank', 'path_count']:
        if node_df[col].max() > node_df[col].min():
            node_df[f'{col}_norm'] = (node_df[col] - node_df[col].min()) / (node_df[col].max() - node_df[col].min())
        else:
            node_df[f'{col}_norm'] = 0.0

    # 综合评分 = 归一化入度 * 0.4 + 归一化路径数 * 0.4 + 归一化PageRank * 0.2
    # 入度和路径数是核心指标，PageRank作为辅助
    node_df['intermediary_score'] = (
        node_df['weighted_in_degree_norm'] * 0.4 +
        node_df['path_count_norm'] * 00.4 +
        node_df['pagerank_norm'] * 0.2
    )

    # 4.5 筛选候选中介账户
    # 只选择 'external' 类型的节点（因为它们没有初始风险标签，符合中介账户定义）
    # 并且路径数 > 0（至少被一条风险路径经过）
    candidates = node_df[
        (node_df['node_type'] == 'external') &
        (node_df['path_count'] > 0)
    ].sort_values('intermediary_score', ascending=False)

    # 取Top 100作为最终中介账户
    intermediaries_df = candidates.head(100).copy()

    print(f"  → 识别出 {len(intermediaries_df):,} 个无标签中介账户候选")
    if len(intermediaries_df) > 0:
        print(f"\n=== Top-10 无标签中介账户 ===")
        top10 = intermediaries_df.head(10)[['account_id', 'intermediary_score', 'weighted_in_degree', 'path_count']]
        print(top10.to_string(index=False))

        # 4.6 为每个中介账户输出其关联的风险路径
        print(f"\n=== 为 Top 3 中介账户输出其关联风险路径 ===")
        for idx, row in intermediaries_df.head(3).iterrows():
            core_node = row['account_id']
            print(f"\n中介账户: {core_node} (评分: {row['intermediary_score']:.4f})")
            # 找到所有指向该中介账户的风险路径
            risk_paths = []
            for risk_node in risk_account_ids:
                if risk_node in G:
                    # 检查1-hop路径
                    if core_node in G.successors(risk_node):
                        risk_paths.append(f"{risk_node} -> {core_node}")
                    # 检查2-hop路径
                    for neighbor in G.successors(risk_node):
                        if core_node in G.successors(neighbor):
                            risk_paths.append(f"{risk_node} -> {neighbor} -> {core_node}")
            print(f"  → 关联风险路径数: {len(risk_paths)}")
            if len(risk_paths) > 0:
                print(f"  → 前3条风险路径: {risk_paths[:3]}")

# ==================== 5. 保存结果 ====================
print("\n[4/5] 保存最终结果...")

# 保存中介账户
if not intermediaries_df.empty:
    intermediaries_df.to_csv(os.path.join(OUTPUT_DIR, 'unlabeled_intermediaries.csv'), index=False, encoding='utf-8-sig')
    print(f"  → 无标签中介账户已保存至: unlabeled_intermediaries.csv")

# 保存图结构为 JSON
if G.number_of_nodes() > 0:
    nodes_data = []
    for node in G.nodes():
        nodes_data.append({
            'id': node,
            'weighted_in_degree': weighted_in_degree.get(node, 0.0),
            'pagerank': pagerank.get(node, 0.0),
            'path_count': path_critical_nodes.get(node, 0),
            'node_type': 'account' if node in risk_account_ids else 'external'
        })

    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'source': u,
            'target': v,
            'weight': float(data.get('weight', 0.0)),
            'amount': float(data.get('amount', 0.0)),
        })

    graph_data = {
        "nodes": nodes_data,
        "edges": edges_data,
        "metadata": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'risk_propagation_graph.json'), 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f" 图结构已保存至：{os.path.join(OUTPUT_DIR, 'risk_propagation_graph.json')}")

# 输出高危对手方 Top 100
print("\n[5/5] 输出高危对手方 Top 100...")
risk_flows = txn_df[txn_df['account_label'].isin(RISK_LABELS)]
top_opponents = risk_flows.groupby('opponent_name').agg(
    transaction_count=('opponent_name', 'count'),
    total_amount=('amount', 'sum'),
    unique_risk_accounts=('account_id', 'nunique')
).sort_values('transaction_count', ascending=False).head(100)

top_opponents.to_csv(os.path.join(OUTPUT_DIR, 'high_risk_opponents.csv'), index=True, encoding='utf-8-sig')
print(f"  → 高危对手方Top100已保存。")

print(f"\n 步骤2完成！总运行时间: {time.time() - start_time_total:.2f} 秒")