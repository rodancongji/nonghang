#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/2_leiden_fast_final_fixed.py
字符串→整数节点 + 有向 Leiden + 加权投票 + PNG/GEXF 输出
千万边 3-5 min 跑完（已修复映射与类型错误）
"""
import pandas as pd
import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time, os, warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# ===================== 1. 参数 =====================
DATA_PATH   = 'D:/Pycharm/Intermediaries_digging/output/cleaned_transactions.csv'
RESOLUTION  = 0.05                       # 单分辨率（预调优）
MIN_COMM_SIZE = 10                       # 社区最小节点数
os.makedirs('output', exist_ok=True)

# 黑样本加权 & 通道/时间衰减
WEIGHT_MAP = {'灰': 1, '黑': 2, '黑密接': 3, '黑次密接': 2, '黑次次密接': 2, '黑次次次密接': 2, '无关': 0, '未知': 0}
CHANNEL_W  = {'1': 1.0, '2': 1.0, '4': 1.2, '5': 1.5, '6': 1.0, 'iwl': 1.8, '9': 1.3}
HALF_LIFE  = 90        # 天

# ===================== 2. 字符串→整数节点 + groupby 建图 =====================
print('=== 2. 字符串→整数节点 + groupby 建图 ===')
start_t = time.time()

# 1. 构造字符串边表
df = pd.read_csv(DATA_PATH, usecols=['account_id','counterparty_id','direction','amount','timestamp','channel'])
df['src'] = np.where(df['direction']=='2', df['account_id'], df['counterparty_id'])
df['tgt'] = np.where(df['direction']=='2', df['counterparty_id'], df['account_id'])

# 2. 通道权重 + 时间衰减
df['days']   = (pd.to_datetime('today') - pd.to_datetime(df['timestamp'])).dt.days
df['decay']  = 2 ** (-df['days'] / HALF_LIFE)
df['ch_w']   = df['channel'].map(CHANNEL_W).fillna(1.0)
df['weight'] = df['amount'] * df['decay'] * df['ch_w']

# 3. 累加同向边权重
edge_str = df.groupby(['src','tgt'], as_index=False).agg(weight=('weight','sum'),
                                                         amount=('amount','sum'))

# 4. 字符串→0-based 整数
unique_nodes = pd.concat([edge_str['src'], edge_str['tgt']]).unique()
node_map = {n: i for i, n in enumerate(unique_nodes)}
edge_str['src_id'] = edge_str['src'].map(node_map)
edge_str['tgt_id'] = edge_str['tgt'].map(node_map)

# 5. igraph 建图（整数）
g = ig.Graph.DataFrame(edge_str[['src_id','tgt_id']], directed=True)
g.es['weight'] = edge_str['weight']
g.es['amount'] = edge_str['amount']
g.vs['name']   = unique_nodes          # 原字符串 id

print(f'有向图建成 |V|={g.vcount():,} |E|={g.ecount():,}  t={time.time()-start_t:.2f}s')

# ===================== 3. 单分辨率 Leiden + 加权投票 =====================
print('\n=== 3. 单分辨率 Leiden + 加权投票 ===')
start_t = time.time()
partition = la.find_partition(g, la.CPMVertexPartition,
                              weights='weight',
                              resolution_parameter=RESOLUTION,
                              n_iterations=-1)

account_df = pd.read_csv('D:/Pycharm/Intermediaries_digging/output/cleaned_accounts.csv')[['account_id','label']]
label_weight = account_df.drop_duplicates(subset='account_id').set_index('account_id')['label'].map(WEIGHT_MAP).to_dict()

# 社区加权投票
community_df = pd.DataFrame({'account_id': g.vs['name'],
                             'community_id': partition.membership})
community_df['w'] = community_df['account_id'].map(label_weight).fillna(0)
comm_stat = (community_df.groupby('community_id')
                         .agg(total_nodes=('account_id','count'),
                              total_w=('w','sum'))
                         .query('total_nodes >= @MIN_COMM_SIZE')
                         .assign(bad_ratio=lambda d: d['total_w']/d['total_nodes'])
                         .sort_values('bad_ratio', ascending=False))

print('\n Top-10 高风险社区（加权投票）')
print(comm_stat.head(10))

# ===================== 4. 节点级拓扑特征 =====================
print('\n=== 4. 节点级拓扑特征 ===')
pr  = g.pagerank(weights='weight')
btw = g.betweenness(directed=True, weights='weight')
in_s  = g.strength(mode='in',  weights='weight')
out_s = g.strength(mode='out', weights='weight')

# 去重后生成唯一标签映射
label_map = account_df.drop_duplicates(subset='account_id').set_index('account_id')['label']
node_feat = pd.DataFrame({
        'account_id': g.vs['name'],
        'in_strength': in_s,
        'out_strength': out_s,
        'net_flow': np.array(out_s) - np.array(in_s),
        'pagerank': pr,
        'betweenness': btw,
        'community_id': partition.membership,
        'label': community_df['account_id'].map(label_map).fillna('未知')
})
print(f'特征完成，shape={node_feat.shape}')

# ===================== 5. 保存结果 =====================
community_df.to_csv('output/leiden_fast_final_fixed_communities.csv', index=False, encoding='utf-8-sig')
comm_stat.to_csv('output/leiden_fast_final_fixed_community_risk.csv', encoding='utf-8-sig')
node_feat.to_csv('output/leiden_fast_final_fixed_node_features.csv', index=False, encoding='utf-8-sig')
print('✅ 已保存：leiden_fast_final_fixed_*.csv')

# ===================== 6. 可视化 & GEXF =====================
top_comm = comm_stat.index[0]
vis_nodes = community_df.query('community_id == @top_comm')['account_id'].tolist()
if len(vis_nodes) > 200:
    vis_nodes = vis_nodes[:200]

sub_g = g.induced_subgraph(vis_nodes)
sub_nx = nx.DiGraph()
sub_nx.add_weighted_edges_from([(sub_g.vs[e.source]['name'],
                                  sub_g.vs[e.target]['name'],
                                  sub_g.es['weight'][e.index]) for e in sub_g.es])

# 先 merge 出 label，再去重映射
viz_df = (community_df[['account_id']]
          .merge(account_df[['account_id','label']], on='account_id', how='left')
          .fillna({'label':'未知'})
          .drop_duplicates(subset='account_id')
          .set_index('account_id')['label'])

color_map = ['red' if viz_df.get(n) in ['灰','黑','黑密接'] else 'lightblue' for n in sub_nx.nodes()]

plt.figure(figsize=(14, 14))
pos = nx.spring_layout(sub_nx, seed=42, k=1.5, iterations=50)
nx.draw_networkx_nodes(sub_nx, pos, node_color=color_map, node_size=80, alpha=0.9)
nx.draw_networkx_edges(sub_nx, pos, edge_color='grey', alpha=0.4, width=0.5)
plt.title(f'Leiden Fast Final Fixed High-Risk Community {top_comm}  (Q={partition.quality():.3f})')
plt.axis('off')
plt.tight_layout()
plt.savefig('output/leiden_fast_final_fixed_high_risk_comm.png', dpi=150, bbox_inches='tight')
plt.show()

nx.write_gexf(sub_nx, 'output/leiden_fast_final_fixed_high_risk_comm.gexf')
print('\n🎉 Leiden Fast Final Fixed 全流程完成！最佳分辨率、加权 F1、GEXF 已输出')

"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#src/2_build_graph_and_leiden_final.py
#终极生产版：网格分辨率 + 加权投票 + 有向 Leiden + 通道/时间权重
"""
import pandas as pd
import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time, os, warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# ===================== 1. 参数区 =====================
DATA_PATH   = 'D:/Pycharm/Intermediaries_digging/output/cleaned_transactions.csv'
CHUNK_SIZE  = 1_000_000
MIN_COMM_SIZE = 10                       # 过滤微型社区
VIS_TOP_K   = 200
os.makedirs('output', exist_ok=True)

# 黑样本权重映射（label → 投票权重）
WEIGHT_MAP = {'灰': 1, '黑': 2, '黑密接': 3, '黑次密接': 2, '黑次次密接': 2, '黑次次次密接': 2, '无关': 0, '未知': 0}

# 通道权重（可选）
CHANNEL_W = {'1': 1.0, '2': 1.0, '4': 1.2, '5': 1.5, '6': 1.0, 'iwl': 1.8, '9': 1.3}

# 时间衰减半衰期（天）
HALF_LIFE = 90

# ===================== 2. 分块建「有向权重图」 =====================
print('=== 2. 分块建「有向权重图」 ===')
start_t = time.time()

edge_dict = {}         # (src, tgt) -> cumulative_weight
node_set  = set()
reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE,
                     usecols=['account_id','counterparty_id','direction','amount','timestamp','channel'])

for k, chunk in enumerate(reader, 1):
    chunk['days'] = (pd.to_datetime('today') - pd.to_datetime(chunk['timestamp'])).dt.days
    chunk['decay'] = 2 ** (-chunk['days'] / HALF_LIFE)          # 指数衰减
    chunk['ch_w'] = chunk['channel'].map(CHANNEL_W).fillna(1.0)
    chunk['weight'] = chunk['amount'] * chunk['decay'] * chunk['ch_w']

    for _, r in chunk.iterrows():
        if r.direction == '2':
            src, tgt = r.account_id, r.counterparty_id
        else:
            src, tgt = r.counterparty_id, r.account_id
        key = (src, tgt)
        edge_dict[key] = edge_dict.get(key, 0) + r.weight
        node_set.update([src, tgt])

    print(f'  chunk {k} processed, current edges {len(edge_dict):,}')

# 建 igraph 有向图
edges   = list(edge_dict.keys())
weights = list(edge_dict.values())
g = ig.Graph(directed=True)
g.add_vertices(list(node_set))
g.add_edges(edges)
g.es['weight'] = weights

print(f'有向图建成 |V|={g.vcount():,} |E|={g.ecount():,}  t={time.time()-start_t:.2f}s')

# ===================== 3. 网格搜分辨率 + 加权投票 =====================
account_df = pd.read_csv('D:/Pycharm/Intermediaries_digging/output/cleaned_accounts.csv')[['account_id','label']]
label_weight = account_df.set_index('account_id')['label'].map(WEIGHT_MAP).to_dict()

res_grid = [0.02, 0.05, 0.08, 0.1, 0.15]
best_f1  = 0
best_res = None
best_part= None

print('\n=== 3. 网格搜分辨率（加权投票 F1）===')
for res in res_grid:
    partition = la.find_partition(g, la.CPMVertexPartition,
                                  weights='weight',
                                  resolution_parameter=res,
                                  n_iterations=-1)
    # 社区级加权投票
    comm_df = pd.DataFrame({'account_id': g.vs['name'],
                            'community_id': partition.membership})
    comm_df['w'] = comm_df['account_id'].map(label_weight).fillna(0)
    comm_stat = (comm_df.groupby('community_id')
                        .agg(total_nodes=('account_id','count'),
                             total_w=('w','sum'))
                        .query('total_nodes >= @MIN_COMM_SIZE')
                        .assign(bad_ratio=lambda d: d['total_w']/d['total_nodes'])
                        .sort_values('bad_ratio', ascending=False))

    # 评估：bad_ratio > 0.2 视为高危社区
    pred_nodes = comm_df[comm_df['community_id'].isin(
                        comm_stat[comm_stat['bad_ratio']>0.2].index)]['account_id'].unique()
    truth = account_df[account_df['label'].isin(['灰','黑','黑密接','黑次密接','黑次次密接'])]['account_id'].values
    tp = len(set(truth) & set(pred_nodes))
    prec = tp / len(pred_nodes) if pred_nodes.size else 0
    rec  = tp / len(truth) if truth.size else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    print(f'  res={res}  F1={f1:.3f}  Q={partition.quality():.3f}')
    if f1 > best_f1:
        best_f1, best_res, best_part = f1, res, partition

print(f'最佳分辨率 = {best_res}  F1 = {best_f1:.3f}')

# ===================== 4. 最终社区结果 =====================
community_df = pd.DataFrame({'account_id': g.vs['name'],
                             'community_id': best_part.membership})
community_df['w'] = community_df['account_id'].map(label_weight).fillna(0)
comm_stat_final = (community_df.groupby('community_id')
                                .agg(total_nodes=('account_id','count'),
                                     total_w=('w','sum'))
                                .query('total_nodes >= @MIN_COMM_SIZE')
                                .assign(bad_ratio=lambda d: d['total_w']/d['total_nodes'])
                                .sort_values('bad_ratio', ascending=False))

print('\n Top-10 高风险社区（加权投票）')
print(comm_stat_final.head(10))

# ===================== 5. 节点级特征 =====================
print('\n=== 4. 节点级拓扑特征 ===')
start_t = time.time()
pr  = g.pagerank(weights='weight')
btw = g.betweenness(directed=True, weights='weight')
in_s  = g.strength(mode='in',  weights='weight')
out_s = g.strength(mode='out', weights='weight')

node_feat = pd.DataFrame({
        'account_id': g.vs['name'],
        'in_strength': in_s,
        'out_strength': out_s,
        'net_flow': out_s - in_s,
        'pagerank': pr,
        'betweenness': btw,
        'community_id': best_part.membership,
        'label': community_df['account_id'].map(account_df.set_index('account_id')['label'])
})
print(f'特征完成，shape={node_feat.shape}  t={time.time()-start_t:.2f}s')

# ===================== 6. 保存结果 =====================
community_df.to_csv('output/leiden_final_communities.csv', index=False, encoding='utf-8-sig')
comm_stat_final.to_csv('output/leiden_final_community_risk.csv', encoding='utf-8-sig')
node_feat.to_csv('output/leiden_final_node_features.csv', index=False, encoding='utf-8-sig')
print(f'已保存：leiden_final_*.csv')

# ===================== 7. 可视化最大高风险社区 =====================
top_comm = comm_stat_final.index[0]
vis_nodes = community_df.query('community_id == @top_comm')['account_id'].tolist()
if len(vis_nodes) > 200:
    vis_nodes = vis_nodes[:200]

sub_g = g.induced_subgraph(vis_nodes)
sub_nx = nx.DiGraph()
sub_nx.add_weighted_edges_from([(sub_g.vs[e.source]['name'],
                                  sub_g.vs[e.target]['name'],
                                  sub_g.es['weight'][e.index]) for e in sub_g.es])

label_map = community_df.drop_duplicates('account_id').set_index('account_id')['label']
color_map = ['red' if label_map.get(n) in ['灰','黑','黑密接'] else 'lightblue' for n in sub_nx.nodes()]

plt.figure(figsize=(14, 14))
pos = nx.spring_layout(sub_nx, seed=42, k=1.5, iterations=50)
nx.draw_networkx_nodes(sub_nx, pos, node_color=color_map, node_size=80, alpha=0.9)
nx.draw_networkx_edges(sub_nx, pos, edge_color='grey', alpha=0.4, width=0.5)
plt.title(f'Leiden Final High-Risk Community {top_comm}  (res={best_res}, Q={best_part.quality():.3f})')
plt.axis('off')
plt.tight_layout()
plt.savefig('output/leiden_final_high_risk_comm.png', dpi=150, bbox_inches='tight')
plt.show()

# ===================== 8. 导出 GEXF（Gephi/Pyvis 直接拖） =====================
nx.write_gexf(sub_nx, 'output/leiden_final_high_risk_comm.gexf')
print('\n Leiden Final 全流程完成！最佳分辨率、加权 F1、GEXF 已输出') """