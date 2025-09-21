#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极生产版：Leiden 社区 + 中介账户识别 + 可视化增强 + JSON图导出
优化点：
  - 修正路径：从 data/ 读取清洗后数据
  - 增加图结构JSON导出（兼容Gephi/D3/前端）
  - 增强稳定性：处理节点不在account_df中的情况
  - 保持完整功能：图构建、社区发现、中介识别、可视化
"""
import pandas as pd
import igraph as ig
import leidenalg as la
import networkx as nx  # 仅用于最终可视化
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time, os, warnings
import json
from collections import Counter

warnings.filterwarnings('ignore')

"""D:\Anaconda\envs\RAG_Learning\python.exe D:\Pycharm\Intermediaries_digging\src\2_build_graph_and_leiden.py 
=== 1. 分块构建有向加权图（含预过滤） ===
  chunk 5 processed, edges: 644,040
  chunk 10 processed, edges: 1,595,262
  → 过滤掉权重最低 10% 的边，剩余 2,082,983 / 原 2,314,409 条边
 有向图建成 |V|=1,081,673 |E|=2,082,983  t=395.98s

=== 2. 网格搜索最优分辨率（串行版） ===
  → 评估分辨率: 0.02
    res=0.020  F1=0.008  Q=7492968173.757
 最佳分辨率 = 0.02  F1 = 0.008  总耗时: 118.39s

=== 3. 高风险社区统计 ===
 Top-10 高风险社区：
              total_nodes  total_w  bad_ratio
community_id                                 
5202                   10      5.0   0.500000
5364                   10      5.0   0.500000
4518                   12      5.0   0.416667
4642                   12      5.0   0.416667
4393                   13      5.0   0.384615
4332                   13      5.0   0.384615
4375                   13      5.0   0.384615
4100                   14      5.0   0.357143
3967                   15      5.0   0.333333
4063                   15      5.0   0.333333

=== 4. 识别中介账户（高效版） ===
  → 计算节点中心性...
  → 计算邻居社区分布...
    进度: 0.0% (0/1,081,673 节点)
    进度: 10.0% (108,167/1,081,673 节点)
    进度: 20.0% (216,334/1,081,673 节点)
    进度: 30.0% (324,501/1,081,673 节点)
    进度: 40.0% (432,668/1,081,673 节点)
    进度: 50.0% (540,835/1,081,673 节点)
    进度: 60.0% (649,002/1,081,673 节点)
    进度: 70.0% (757,169/1,081,673 节点)
    进度: 80.0% (865,336/1,081,673 节点)
    进度: 90.0% (973,503/1,081,673 节点)
    进度: 100.0% (1,081,670/1,081,673 节点)
  → 邻居社区计算完成
  → 识别中介账户候选...
 识别出 80 个中介账户候选  t=47.36s

 Top-10 中介账户：
                                              account_id  ... risk_score
6684   6eddaed88b09cdf0cc1f41befd74d4e655f44f59d74254...  ...        0.5
8107   dae6a3ba750619e6dcdd0287ed32fc764e02df595034ef...  ...        0.5
13026  2bb9d0974cb658da06289c9bc3b0376f8de3bdfd02dcd3...  ...        0.5
19669  d1c1c8dea15c36b6c80b9c88b9675c325699466b1c7d7f...  ...        0.5
20844  d774afb6ffae3b07e6ad9ad078b21afd3eee18dd027fb6...  ...        0.5
48808  5bbe189db299e0591a3e88e4cc29007df730c182f48703...  ...        0.5
51916  496c53ac0d8824c44904a562acf1e273a4f5fd66b05aae...  ...        0.5
54950  21f728848f8077cbd1f14b1af5ab4fe2308dc0ccd68ed7...  ...        0.5
55140  c102c00eca303fcb6e2be1b29b4a0aeabae8ac91692b68...  ...        0.5
59200  d53a8b5e74a1f6fc80ad28b000320efd46f3b59a78b555...  ...        0.5

[10 rows x 6 columns]

=== 5. 保存结果 ===
 已保存：intermediaries.csv
 已保存：leiden_communities.csv, leiden_community_risk.csv, node_features.csv

=== 5.5 保存图结构为 JSON 格式 ===
 图结构已保存至：D:/Pycharm/Intermediaries_digging/output/graph_structure.json
    节点数：1,081,673 | 边数：2,082,983

=== 6. 可视化中介账户子图 ===
  → 构建可视化子图...
 已保存中介账户子图：intermediary_subgraph.png, intermediary_subgraph.gexf

=== 7. 可视化社区风险分布 ===

 图构建 + 中介账户识别 + 可视化全流程完成！
关键产出：
   - leiden_communities.csv: 节点-社区映射
   - leiden_community_risk.csv: 社区风险统计
   - node_features.csv: 节点拓扑特征
   - intermediaries.csv: 中介账户列表（如有）
   - graph_structure.json: 完整图结构（JSON格式）
   - intermediary_subgraph.png: 中介账户子图
   - community_risk_distribution.png: 社区风险分布图

 总运行时间: 633.86 秒

Process finished with exit code 0"""
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ===================== 1. 参数区 =====================
#
DATA_PATH = 'D:/Pycharm/Intermediaries_digging/data/cleaned_transactions.csv'
ACCOUNT_PATH = 'D:/Pycharm/Intermediaries_digging/data/cleaned_accounts.csv'

CHUNK_SIZE = 1_000_000
MIN_COMM_SIZE = 10
VIS_TOP_K = 200
os.makedirs('output', exist_ok=True)

# 黑样本权重映射（根据业务调整）
WEIGHT_MAP = {
    '灰': 0.5, '黑': 5, '黑密接': 4, '黑次密接': 3,
    '黑次次密接': 2, '黑次次次密接': 1, '无关': 0, '未知': 0
}

# 通道权重
CHANNEL_W = {'1': 1.0, '2': 1.0, '4': 1.2, '5': 1.5, '6': 1.0, 'iwl': 1.8, '9': 1.3}

# 时间衰减半衰期（天）
HALF_LIFE = 30

# 中介账户筛选阈值
BETWEENNESS_PCT = 95
MIN_COMM_CONNECTED = 2
MIN_RISK_COMM_RATIO = 0.3

# Leiden 优化参数
EDGE_FILTER_PERCENTILE = 10
MAX_LEIDEN_ITERATIONS = 10

# ===================== 2. 分块建「有向权重图」+ 预过滤 =====================
print('=== 1. 分块构建有向加权图（含预过滤） ===')
start_t = time.time()

edge_dict = {}
node_set = set()
reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE,
                     usecols=['account_id', 'counterparty_id', 'direction', 'amount', 'timestamp', 'channel'])

for k, chunk in enumerate(reader, 1):
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
    chunk = chunk.dropna(subset=['timestamp'])
    # 使用数据集中最大时间作为基准
    chunk['days'] = (pd.Timestamp('2025-07-08') - chunk['timestamp']).dt.days
    chunk['decay'] = 2 ** (-chunk['days'] / HALF_LIFE)
    chunk['ch_w'] = chunk['channel'].map(CHANNEL_W).fillna(1.0)
    chunk['weight'] = chunk['amount'] * chunk['decay']  # * chunk['ch_w']

    for _, r in chunk.iterrows():
        if r.direction == '2':  # 付款
            src, tgt = r.account_id, r.counterparty_id
        else:  # 收款
            src, tgt = r.counterparty_id, r.account_id
        key = (src, tgt)
        edge_dict[key] = edge_dict.get(key, 0) + r.weight
        node_set.update([src, tgt])

    if k % 5 == 0:
        print(f'  chunk {k} processed, edges: {len(edge_dict):,}')

# 预过滤：移除权重最低 10% 的边
if len(edge_dict) > 0:
    weights = np.array(list(edge_dict.values()))
    threshold = np.percentile(weights, EDGE_FILTER_PERCENTILE)
    original_edge_count = len(edge_dict)
    edge_dict = {k: v for k, v in edge_dict.items() if v >= threshold}
    print(
        f'  → 过滤掉权重最低 {EDGE_FILTER_PERCENTILE}% 的边，剩余 {len(edge_dict):,} / 原 {original_edge_count:,} 条边')

# 构建 igraph 有向图
edges = list(edge_dict.keys())
weights = list(edge_dict.values())
g = ig.Graph(directed=True)
g.add_vertices(list(node_set))
g.add_edges(edges)
g.es['weight'] = weights

print(f' 有向图建成 |V|={g.vcount():,} |E|={g.ecount():,}  t={time.time() - start_t:.2f}s')

# ===================== 3. 网格搜分辨率 + 加权投票（串行版） =====================
print('\n=== 2. 网格搜索最优分辨率（串行版） ===')

# 加载账户标签（只加载一次）  修正路径
account_df = pd.read_csv(ACCOUNT_PATH)[['account_id', 'label']]
label_weight = account_df.set_index('account_id')['label'].map(WEIGHT_MAP).to_dict()

# 创建 label_map 用于可视化（确保所有节点都有标签）
label_map = dict(zip(account_df['account_id'], account_df['label']))

# 真实风险账户（只计算一次）
truth = account_df[account_df['label'].isin(['灰', '黑', '黑密接', '黑次密接', '黑次次密接'])]['account_id'].values

res_grid = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1]
best_f1 = 0
best_res = None
best_part = None

start_search = time.time()

for res in res_grid:
    try:
        print(f'  → 评估分辨率: {res}')
        partition = la.find_partition(g, la.CPMVertexPartition,
                                      weights='weight',
                                      resolution_parameter=res,
                                      n_iterations=MAX_LEIDEN_ITERATIONS)

        # 构建社区-账户映射
        comm_df = pd.DataFrame({
            'account_id': g.vs['name'],
            'community_id': partition.membership
        })
        comm_df['w'] = comm_df['account_id'].map(label_weight).fillna(0)

        # 社区统计
        comm_stat = (comm_df.groupby('community_id')
                     .agg(total_nodes=('account_id', 'count'),
                          total_w=('w', 'sum'))
                     .query('total_nodes >= @MIN_COMM_SIZE')
                     .assign(bad_ratio=lambda d: d['total_w'] / d['total_nodes'])
                     .sort_values('bad_ratio', ascending=False))

        # 预测风险账户
        high_risk_comms = comm_stat[comm_stat['bad_ratio'] > 0.2].index
        pred_nodes = comm_df[comm_df['community_id'].isin(high_risk_comms)]['account_id'].unique()

        # 计算 F1
        tp = len(set(truth) & set(pred_nodes))
        prec = tp / len(pred_nodes) if len(pred_nodes) > 0 else 0
        rec = tp / len(truth) if len(truth) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        quality = partition.quality()

        print(f'    res={res:.3f}  F1={f1:.3f}  Q={quality:.3f}')

        if f1 > best_f1:
            best_f1, best_res, best_part = f1, res, partition

    except Exception as e:
        print(f"     分辨率 {res} 运行出错: {e}")
        continue

print(f' 最佳分辨率 = {best_res}  F1 = {best_f1:.3f}  总耗时: {time.time() - start_search:.2f}s')

if best_part is None:
    raise RuntimeError("所有分辨率均失败，请检查数据或参数")

# ===================== 4. 社区统计 & 高风险社区 =====================
print('\n=== 3. 高风险社区统计 ===')
community_df = pd.DataFrame({
    'account_id': g.vs['name'],
    'community_id': best_part.membership
})
community_df['w'] = community_df['account_id'].map(label_weight).fillna(0)

comm_stat_final = (community_df.groupby('community_id')
                   .agg(total_nodes=('account_id', 'count'),
                        total_w=('w', 'sum'))
                   .query('total_nodes >= @MIN_COMM_SIZE')
                   .assign(bad_ratio=lambda d: d['total_w'] / d['total_nodes'])
                   .sort_values('bad_ratio', ascending=False))

print(' Top-10 高风险社区：')
print(comm_stat_final.head(10))

high_risk_comms = set(comm_stat_final[comm_stat_final['bad_ratio'] > 0.2].index)

# ===================== 5. 中介账户识别（高效版） =====================
print('\n=== 4. 识别中介账户（高效版） ===')
start_t_inner = time.time()

# 计算节点中心性指标
print("  → 计算节点中心性...")
pr = g.pagerank(weights='weight')
btw = g.betweenness(directed=True, weights='weight')
in_s = g.strength(mode='in', weights='weight')
out_s = g.strength(mode='out', weights='weight')

# 获取节点名称和社区ID
node_names = g.vs['name']
community_ids = best_part.membership
node_to_comm = dict(zip(node_names, community_ids))

# 将社区ID转为numpy数组
comm_array = np.array(community_ids)
high_risk_comm_array = np.array(list(high_risk_comms))

# 创建节点索引映射
node_index_map = {name: idx for idx, name in enumerate(node_names)}

# 计算每个节点的邻居社区分布
print("  → 计算邻居社区分布...")
comm_connections = np.zeros(len(node_names), dtype=int)
risk_ratio_arr = np.zeros(len(node_names), dtype=float)

total_nodes = len(node_names)
report_step = max(1, total_nodes // 10)  # 每10%报告一次

for i in range(total_nodes):
    if i % report_step == 0:
        progress = (i / total_nodes) * 100
        print(f"    进度: {progress:.1f}% ({i:,}/{total_nodes:,} 节点)")

    # 获取当前节点的所有邻居（入+出）
    neighbors_in = g.neighbors(i, mode=ig.IN)
    neighbors_out = g.neighbors(i, mode=ig.OUT)
    neighbor_indices = list(set(neighbors_in + neighbors_out))  # 去重

    if len(neighbor_indices) == 0:
        comm_connections[i] = 0
        risk_ratio_arr[i] = 0.0
        continue

    # 获取邻居的社区ID
    neighbor_comms = comm_array[neighbor_indices]
    unique_comms = np.unique(neighbor_comms)
    comm_connections[i] = len(unique_comms)

    # 计算高风险社区比例
    if len(unique_comms) > 0:
        risk_mask = np.isin(unique_comms, high_risk_comm_array)
        risk_ratio_arr[i] = np.sum(risk_mask) / len(unique_comms)
    else:
        risk_ratio_arr[i] = 0.0

print("  → 邻居社区计算完成")

# 创建节点特征表
node_feat = pd.DataFrame({
    'account_id': node_names,
    'in_strength': in_s,
    'out_strength': out_s,
    'net_flow': np.asarray(out_s) - np.asarray(in_s),
    'pagerank': pr,
    'betweenness': btw,
    'community_id': community_ids,
    'comm_connections': comm_connections,
    'risk_ratio': risk_ratio_arr,
    # 修正：使用 label_map.get 并提供默认值
    'label': [label_map.get(name, '未知') for name in node_names]
})

# 标准化 betweenness
if node_feat['betweenness'].max() > node_feat['betweenness'].min():
    node_feat['betweenness_norm'] = (node_feat['betweenness'] - node_feat['betweenness'].min()) / \
                                    (node_feat['betweenness'].max() - node_feat['betweenness'].min())
else:
    node_feat['betweenness_norm'] = 0

# 识别中介账户候选
print("  → 识别中介账户候选...")
threshold_btw = np.percentile(node_feat['betweenness'], BETWEENNESS_PCT)
candidates = node_feat[
    (node_feat['betweenness'] >= threshold_btw) &
    (node_feat['comm_connections'] >= MIN_COMM_CONNECTED) &
    (node_feat['risk_ratio'] >= MIN_RISK_COMM_RATIO)
    ].copy()

if len(candidates) > 0:
    # 计算综合风险评分
    max_risk_ratio = candidates['risk_ratio'].max()
    max_comm_conn = candidates['comm_connections'].max()

    candidates['risk_score'] = (
            candidates['betweenness_norm'] * 0.4 +
            (candidates['risk_ratio'] / max_risk_ratio if max_risk_ratio > 0 else 0) * 0.3 +
            (candidates['comm_connections'] / max_comm_conn if max_comm_conn > 0 else 0) * 0.3
    )

    candidates = candidates.sort_values('risk_score', ascending=False)

print(f' 识别出 {len(candidates)} 个中介账户候选  t={time.time() - start_t_inner:.2f}s')

if len(candidates) > 0:
    print('\n Top-10 中介账户：')
    print(candidates[['account_id', 'label', 'betweenness', 'comm_connections', 'risk_ratio', 'risk_score']].head(10))
else:
    print(" 未识别到符合条件的中介账户，请调整阈值或检查数据")

# ===================== 6. 保存结果 =====================
print('\n=== 5. 保存结果 ===')
community_df.to_csv('D:/Pycharm/Intermediaries_digging/output/leiden_communities.csv', index=False, encoding='utf-8-sig')
comm_stat_final.to_csv('D:/Pycharm/Intermediaries_digging/output/leiden_community_risk.csv', encoding='utf-8-sig')
node_feat.to_csv('D:/Pycharm/Intermediaries_digging/output/node_features.csv', index=False, encoding='utf-8-sig')

if len(candidates) > 0:
    candidates.to_csv('D:/Pycharm/Intermediaries_digging/output/intermediaries.csv', index=False, encoding='utf-8-sig')
    print(' 已保存：intermediaries.csv')
else:
    print(' 未保存 intermediaries.csv（无候选账户）')

print(' 已保存：leiden_communities.csv, leiden_community_risk.csv, node_features.csv')


# ===================== 6.5 保存图结构为 JSON =====================
print('\n=== 5.5 保存图结构为 JSON 格式 ===')

# 准备节点数据
node_list = []
for i, node_name in enumerate(g.vs['name']):
    node_data = {
        "id": node_name,
        "community_id": int(community_ids[i]) if i < len(community_ids) else -1,
        "pagerank": float(pr[i]) if i < len(pr) else 0.0,
        "betweenness": float(btw[i]) if i < len(btw) else 0.0,
        "in_strength": float(in_s[i]) if i < len(in_s) else 0.0,
        "out_strength": float(out_s[i]) if i < len(out_s) else 0.0,
        "label": str(label_map.get(node_name, '未知')),
        "is_intermediary": node_name in (set(candidates['account_id']) if len(candidates) > 0 else set())
    }
    node_list.append(node_data)

# 准备边数据
edge_list = []
for e in g.es:
    src_idx, tgt_idx = e.tuple
    src_name = g.vs[src_idx]['name']
    tgt_name = g.vs[tgt_idx]['name']
    edge_data = {
        "source": src_name,
        "target": tgt_name,
        "weight": float(e['weight']) if 'weight' in e.attributes() else 1.0
    }
    edge_list.append(edge_data)

# 准备元数据
metadata = {
    "total_nodes": g.vcount(),
    "total_edges": g.ecount(),
    "high_risk_communities": list(high_risk_comms) if 'high_risk_comms' in locals() else [],
    "intermediaries": candidates['account_id'].tolist() if len(candidates) > 0 else [],
    "best_resolution": float(best_res) if best_res is not None else None,
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
}

# 组装完整图结构
graph_json = {
    "nodes": node_list,
    "edges": edge_list,
    "metadata": metadata
}

# 保存为 JSON 文件
json_output_path = 'D:/Pycharm/Intermediaries_digging/output/graph_structure.json'
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(graph_json, f, ensure_ascii=False, indent=2)

print(f' 图结构已保存至：{json_output_path}')
print(f'    节点数：{len(node_list):,} | 边数：{len(edge_list):,}')


# ===================== 7. 可视化：中介账户子图 =====================
print('\n=== 6. 可视化中介账户子图 ===')
if len(candidates) > 0:
    top_intermediary = candidates.iloc[0]['account_id']

    # 为可视化构建 NetworkX 子图（只在最后一步使用，不影响性能）
    print("  → 构建可视化子图...")
    # 找到 top_intermediary 的索引
    if top_intermediary in node_index_map:
        node_idx = node_index_map[top_intermediary]
        # 获取1跳邻居
        neighbors_in = g.neighbors(node_idx, mode=ig.IN)
        neighbors_out = g.neighbors(node_idx, mode=ig.OUT)
        neighbor_indices = list(set(neighbors_in + neighbors_out))

        # 获取邻居名称
        neighbor_names = [g.vs[idx]['name'] for idx in neighbor_indices]
        vis_nodes = [top_intermediary] + neighbor_names

        # 限制节点数
        if len(vis_nodes) > 50:
            vis_nodes = [top_intermediary] + neighbor_names[:49]

        # 构建子图（只包含这些节点和它们之间的边）
        sub_nx = nx.DiGraph()
        # 添加节点
        for node in vis_nodes:
            sub_nx.add_node(node)

        # 添加边（只添加存在于原图中的边）
        for i, src in enumerate(vis_nodes):
            if src in node_index_map:
                src_idx = node_index_map[src]
                for j, tgt in enumerate(vis_nodes):
                    if i != j and tgt in node_index_map:
                        tgt_idx = node_index_map[tgt]
                        # 检查是否存在边
                        try:
                            edge_idx = g.get_eid(src_idx, tgt_idx, directed=True, error=False)
                            if edge_idx != -1:
                                weight = g.es[edge_idx]['weight']
                                sub_nx.add_edge(src, tgt, weight=weight)
                        except:
                            continue

        # 颜色映射
        comm_map = dict(zip(community_df['account_id'], community_df['community_id']))

        node_colors = []
        node_sizes = []
        for node in sub_nx.nodes():
            label = label_map.get(node, '未知')
            if node == top_intermediary:
                color, size = 'red', 300  # 中介账户
            elif label in ['灰', '黑', '黑密接']:
                color, size = 'orange', 150  # 高风险账户
            else:
                color, size = 'lightblue', 100  # 正常账户
            node_colors.append(color)
            node_sizes.append(size)

        # 绘图
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(sub_nx, seed=42, k=2.0, iterations=50)

        # 绘制边
        nx.draw_networkx_edges(sub_nx, pos, edge_color='grey', alpha=0.5, width=0.8)

        # 绘制节点
        nx.draw_networkx_nodes(sub_nx, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)

        # 添加标签
        labels = {}
        for node in sub_nx.nodes():
            label = label_map.get(node, '未知')
            if node == top_intermediary:
                labels[node] = f"中介\n{node[:6]}..."
            elif label in ['灰', '黑', '黑密接']:
                labels[node] = f"{label}\n{node[:6]}..."

        nx.draw_networkx_labels(sub_nx, pos, labels, font_size=8, font_weight='bold')

        plt.title(
            f'中介账户 {top_intermediary[:8]}... 及其直接连接网络\n(连接{len(neighbor_names)}个账户，跨越{len(set(comm_map.get(n, -1) for n in neighbor_names))}个社区)',
            fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('output/intermediary_subgraph.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 保存 GEXF
        nx.write_gexf(sub_nx, 'output/intermediary_subgraph.gexf')
        print(' 已保存中介账户子图：intermediary_subgraph.png, intermediary_subgraph.gexf')
    else:
        print(" 未找到中介账户在图中的索引")
else:
    print(" 跳过可视化（无中介账户）")

# ===================== 8. 可视化：社区风险分布 =====================
print('\n=== 7. 可视化社区风险分布 ===')
plt.figure(figsize=(12, 6))
top20 = comm_stat_final.head(20).reset_index()
bars = plt.bar(range(len(top20)), top20['bad_ratio'],
               color=['red' if r > 0.5 else 'orange' if r > 0.2 else 'green' for r in top20['bad_ratio']])

plt.xlabel('社区ID')
plt.ylabel('风险账户比例')
plt.title('Top 20 高风险社区风险比例分布')
plt.xticks(range(len(top20)), [f"C{i}" for i in top20['community_id']], rotation=45)
plt.grid(axis='y', alpha=0.7)

# 在柱子上添加数值标签
for i, (bar, row) in enumerate(zip(bars, top20.iterrows())):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.2f}\n({int(row[1]["total_nodes"])}节点)',
             ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig('output/community_risk_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n 图构建 + 中介账户识别 + 可视化全流程完成！')
print('关键产出：')
print('   - leiden_communities.csv: 节点-社区映射')
print('   - leiden_community_risk.csv: 社区风险统计')
print('   - node_features.csv: 节点拓扑特征')
print('   - intermediaries.csv: 中介账户列表（如有）')
print('   - graph_structure.json: 完整图结构（JSON格式）')
print('   - intermediary_subgraph.png: 中介账户子图')
print('   - community_risk_distribution.png: 社区风险分布图')

print(f'\n 总运行时间: {time.time() - start_t:.2f} 秒')
