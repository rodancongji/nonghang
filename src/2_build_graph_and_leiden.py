# -*- coding: utf-8 -*-
"""
步骤2：全量图团伙发掘 + 训练图特征构建（科学可行版 - 修复中介识别 + 引入 balance 判断快进快出）
功能：
  - 构建全量图 H（团伙发掘）
  - 构建训练图 G_train（XGBoost 特征）
  - **正确识别中介账户：包含'灰'账户 + 高频未知对手方（opponent_name），基于快进快出 + 局部 Betweenness + PageRank + balance 无沉淀**
  - 保留高危对手方 Top100
  - **为交互式可视化系统保存全量图数据**
"""
import os, time, json, numpy as np, pandas as pd, networkx as nx, igraph as ig, leidenalg as la
import matplotlib
matplotlib.use('Agg')  # 避免 GUI 后端问题

print("=== 步骤2：全量图团伙发掘 + 训练图特征构建 ===")
start = time.time()

FULL_TXN_PATH = 'D:/Pycharm/Intermediaries_digging/data/cleaned_transactions.csv'
TRAIN_TXN_PATH = 'D:/Pycharm/Intermediaries_digging/data/train_transactions.csv'
TEST_IDS_PATH = 'D:/Pycharm/Intermediaries_digging/output/test_account_ids.csv'
OUTPUT_DIR = 'D:/Pycharm/Intermediaries_digging/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOISE = ['微信转账', '扫二维码付款', '微信红包', '还款', '手机充值', '零钱通', '生活缴费', '我的钱包', '待报解预算收入']

# ==================== 1. 加载数据 ====================
print("\n[1/8] 加载全量和训练交易数据...")
full_txn = pd.read_csv(FULL_TXN_PATH, encoding='utf-8-sig', low_memory=False)
train_txn = pd.read_csv(TRAIN_TXN_PATH, encoding='utf-8-sig', low_memory=False)
test_accounts = set(pd.read_csv(TEST_IDS_PATH, encoding='utf-8-sig')['account_id'])

full_txn['direction'] = full_txn['direction'].astype(str).str.strip()
train_txn['direction'] = train_txn['direction'].astype(str).str.strip()

if 'balance' in full_txn.columns:
    full_txn['balance'] = pd.to_numeric(full_txn['balance'], errors='coerce')
else:
    full_txn['balance'] = np.nan

for df in [full_txn, train_txn]:
    df.drop(columns=[col for col in ['account_label', 'counterparty_label'] if col in df.columns], inplace=True, errors='ignore')

ORIGINAL_ACCOUNT_PATH = 'D:/Pycharm/Intermediaries_digging/data/名单_脱敏后.csv'
account_df = pd.read_csv(ORIGINAL_ACCOUNT_PATH, encoding='UTF-8-SIG')
if '客户唯一id' in account_df.columns:
    account_df.rename(columns={'客户唯一id': 'account_id'}, inplace=True)
account_df['account_id'] = account_df['account_id'].astype(str)
full_txn['account_id'] = full_txn['account_id'].astype(str)
train_txn['account_id'] = train_txn['account_id'].astype(str)
test_accounts = {str(x) for x in test_accounts}

account_label_map = account_df.set_index('account_id')['label'].to_dict()

# === 合并标签（注意：此时 full_txn 还没有 src/tgt）===
full_txn = full_txn.merge(
    account_df[['account_id', 'label']].rename(columns={'label': 'account_label'}),
    on='account_id', how='left'
).merge(
    account_df[['account_id', 'label']].rename(columns={'account_id': 'counterparty_id', 'label': 'counterparty_label'}),
    on='counterparty_id', how='left'
)
full_txn['account_label'] = full_txn['account_label'].fillna('未知')
full_txn['counterparty_label'] = full_txn['counterparty_label'].fillna('未知')

train_txn = train_txn.merge(
    account_df[['account_id', 'label']].rename(columns={'label': 'account_label'}),
    on='account_id', how='left'
).merge(
    account_df[['account_id', 'label']].rename(columns={'account_id': 'counterparty_id', 'label': 'counterparty_label'}),
    on='counterparty_id', how='left'
)
train_txn['account_label'] = train_txn['account_label'].fillna('未知')
train_txn['counterparty_label'] = train_txn['counterparty_label'].fillna('未知')

official_account_ids = set(account_df['account_id'].unique())
RISK_LABELS = ['黑', '灰', '黑密接', '黑次密接', '黑次次密接', '黑次次次密接']
risk_account_ids_full = set(full_txn[full_txn['account_label'].isin(RISK_LABELS)]['account_id'].unique())
risk_account_ids_train = set(train_txn[train_txn['account_label'].isin(RISK_LABELS)]['account_id'].unique())

print(f"原始名单账户: {len(official_account_ids):,}")
print(f"全量黑产账户: {len(risk_account_ids_full):,}, 训练黑产账户: {len(risk_account_ids_train):,}")
print("交易数据 account_label 分布:")
print(full_txn['account_label'].value_counts().head(10))

# ✅✅✅ 关键修复：在所有 merge 完成后，再添加 src/tgt 列 ✅✅✅
full_txn['src'] = np.where(full_txn['direction'] == '2', full_txn['account_id'], full_txn['opponent_name'])
full_txn['tgt'] = np.where(full_txn['direction'] == '2', full_txn['opponent_name'], full_txn['account_id'])

# ==================== 2. 构建全量图 H ====================
print("\n[2/8] 构建全量图 H")
full_txn_clean = full_txn[~full_txn['opponent_name'].isin(NOISE)]
edges_full = full_txn_clean.copy()
edges_full['weight'] = np.log1p(edges_full['amount'])
edges_full['src'] = np.where(edges_full['direction'] == '2', edges_full['account_id'], edges_full['opponent_name'])
edges_full['tgt'] = np.where(edges_full['direction'] == '2', edges_full['opponent_name'], edges_full['account_id'])

H = nx.from_pandas_edgelist(edges_full, source='src', target='tgt', edge_attr=['weight', 'amount'], create_using=nx.DiGraph())
print(f"全量图 H：{H.number_of_nodes():,} 节点, {H.number_of_edges():,} 边")

# ==================== 3. 2-hop 子图 + 团伙发掘 ====================
print("\n[3/8] 全量图 2-hop 子图 + 团伙发掘...")
risk_nodes = list(risk_account_ids_full & set(H.nodes()))
all_nodes = set()
if risk_nodes:
    for node in risk_nodes:
        all_nodes.add(node)
        neighbors = set(H.successors(node)) | set(H.predecessors(node))
        all_nodes.update(neighbors)
        for nb in neighbors:
            if nb in H:
                nb_neighbors = set(H.successors(nb)) | set(H.predecessors(nb))
                all_nodes.update(nb_neighbors)
H_sub = H.subgraph(all_nodes).copy()
print(f"2-hop 子图：{H_sub.number_of_nodes():,} 节点, {H_sub.number_of_edges():,} 边")

# === 预计算 H_sub 的全局中心性（供后续复用，只计算一次）===
H_sub_pagerank = {}
if H_sub.number_of_nodes() > 0:
    try:
        print("  预计算 H_sub 的 PageRank 和 Betweenness...")
        H_sub_pagerank = nx.pagerank(H_sub, weight='amount')
        print(f"PageRank 覆盖 {len(H_sub_pagerank)} 节点")
    except Exception as e:
        print(f"警告：H_sub 中心性计算失败: {e}")

community_df = pd.DataFrame()
if H_sub.number_of_nodes() > 0:
    g_ig = ig.Graph.from_networkx(H_sub)
    if 'weight' not in g_ig.edge_attributes():
        g_ig.es['weight'] = [1.0] * g_ig.ecount()
    best_mod = -1
    best_part = None
    for res in [0.1, 0.5, 1.0, 1.5, 2.0]:
        try:
            part = la.find_partition(g_ig, la.RBConfigurationVertexPartition, resolution_parameter=res, weights=g_ig.es['weight'])
            if part.modularity > best_mod:
                best_mod, best_part = part.modularity, part
        except:
            continue
    if best_part:
        node_list = list(H_sub.nodes())
        comm_map = dict(zip(node_list, best_part.membership))
        community_df = pd.DataFrame([{'account_id': node, 'community_id': cid} for node, cid in comm_map.items()])

    node_label_map = full_txn.set_index('account_id')['account_label'].to_dict()
    opponent_label_map = full_txn.set_index('opponent_name')['counterparty_label'].to_dict()
    community_df['account_label'] = community_df['account_id'].map(
        lambda x: node_label_map.get(x, opponent_label_map.get(x, '未知'))
    )

    community_stats = community_df.groupby('community_id').agg(
        total_nodes=('account_id', 'count'),
        risk_nodes=('account_label', lambda x: x.isin(RISK_LABELS).sum()),
        risk_ratio=('account_label', lambda x: x.isin(RISK_LABELS).mean())
    ).sort_values('risk_ratio', ascending=False)

    community_df.to_csv(os.path.join(OUTPUT_DIR, 'gang_communities.csv'), index=False, encoding='utf-8-sig')
    community_stats.to_csv(os.path.join(OUTPUT_DIR, 'gang_community_stats.csv'), index=False, encoding='utf-8-sig')
    print(f"团伙发掘完成，共 {len(community_stats)} 个社区")

# ==================== 4. 构建训练图 G_train ====================
print("\n[4/8] 构建训练图 G_train（用于 XGBoost 特征）...")
train_txn_clean = train_txn[~train_txn['opponent_name'].isin(NOISE)]
train_txn_clean = train_txn_clean[
    (train_txn_clean['account_label'].isin(RISK_LABELS)) |
    (train_txn_clean['counterparty_label'].isin(RISK_LABELS))
]

edges_train = train_txn_clean.copy()
edges_train['weight'] = np.log1p(edges_train['amount'])
edges_train['src'] = np.where(edges_train['direction'] == '2', edges_train['account_id'], edges_train['opponent_name'])
edges_train['tgt'] = np.where(edges_train['direction'] == '2', edges_train['opponent_name'], edges_train['account_id'])

G_train = nx.from_pandas_edgelist(edges_train, source='src', target='tgt', edge_attr=['weight', 'amount'], create_using=nx.DiGraph())
print(f"训练图 G_train：{G_train.number_of_nodes():,} 节点, {G_train.number_of_edges():,} 边")

# ==================== 5. 识别中介账户 ====================
print("\n[5/8] 科学识别中介账户（基于2-hop子图 + 高效向量化快进快出 + balance 无沉淀）...")

candidate_nodes = set(H_sub.nodes()) if 'H_sub' in locals() and H_sub.number_of_nodes() > 0 else set()
if not candidate_nodes:
    print("警告：2-hop子图为空，跳过中介识别")
    intermediaries_df = pd.DataFrame(columns=['account_id', 'fast_out_ratio', 'betweenness', 'pagerank', 'intermediary_score'])
else:
    print(f"候选中介节点总数（2-hop内）: {len(candidate_nodes):,}")

    valid_mask = (
        full_txn['timestamp'].notna() &
        pd.to_numeric(full_txn['amount'], errors='coerce').notna() &
        (pd.to_numeric(full_txn['amount'], errors='coerce') > 0) &
        full_txn['direction'].isin(['1', '2'])
    )
    required_cols = ['account_id', 'opponent_name', 'direction', 'amount', 'timestamp', 'balance']
    CHUNK_SIZE = 500000

    all_summary_chunks = []
    candidate_mask = (
        valid_mask &
        (
            full_txn['account_id'].isin(candidate_nodes) |
            full_txn['opponent_name'].isin(candidate_nodes)
        )
    )
    candidate_indices = full_txn.index[candidate_mask].tolist()

    if not candidate_indices:
        print("警告：候选交易为空！")
        intermediaries_df = pd.DataFrame(columns=['account_id', 'fast_out_ratio', 'betweenness', 'pagerank', 'intermediary_score'])
    else:
        print(f"候选交易行数: {len(candidate_indices):,}")

        for i in range(0, len(candidate_indices), CHUNK_SIZE):
            chunk_indices = candidate_indices[i:i + CHUNK_SIZE]
            chunk = full_txn.loc[chunk_indices, required_cols].copy()

            mask_opponent_in = chunk['opponent_name'].isin(candidate_nodes)
            if mask_opponent_in.any():
                temp = chunk.loc[mask_opponent_in, 'account_id'].copy()
                chunk.loc[mask_opponent_in, 'account_id'] = chunk.loc[mask_opponent_in, 'opponent_name']
                chunk.loc[mask_opponent_in, 'opponent_name'] = temp
                chunk.loc[mask_opponent_in, 'direction'] = chunk.loc[mask_opponent_in, 'direction'].map({'1': '2', '2': '1'})

            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
            chunk = chunk[chunk['timestamp'].notna()]
            if chunk.empty:
                continue

            in_flows = chunk[chunk['direction'] == '1'][['account_id', 'timestamp', 'amount', 'balance']].rename(columns={'timestamp': 'time', 'amount': 'in_amount'})
            out_flows = chunk[chunk['direction'] == '2'][['account_id', 'timestamp', 'amount']].rename(columns={'timestamp': 'time', 'amount': 'out_amount'})
            if in_flows.empty or out_flows.empty:
                continue

            in_flows['out_amount'] = 0.0
            out_flows['in_amount'] = 0.0
            in_flows['type'] = 'in'
            out_flows['type'] = 'out'
            all_flows = pd.concat([in_flows, out_flows], ignore_index=True)
            all_flows = all_flows.sort_values(['account_id', 'time']).reset_index(drop=True)

            def fast_out_ratio_with_balance(group):
                group = group.sort_values('time').reset_index(drop=True)
                if len(group) < 2:
                    return pd.Series({'total_in': 0, 'fast_out_amt': 0, 'low_balance_count': 0})
                in_mask = group['type'] == 'in'
                in_rows = group[in_mask]
                if in_rows.empty or in_rows['in_amount'].sum() == 0:
                    return pd.Series({'total_in': 0, 'fast_out_amt': 0, 'low_balance_count': 0})
                total_in = in_rows['in_amount'].sum()
                group['next_type'] = group['type'].shift(-1)
                group['next_time'] = group['time'].shift(-1)
                group['next_out_amount'] = group['out_amount'].shift(-1).fillna(0)
                valid_in = group[
                    (group['type'] == 'in') &
                    (group['next_type'] == 'out') &
                    ((group['next_time'] - group['time']) <= pd.Timedelta(hours=24))
                ]
                fast_out_amt = 0.0
                if not valid_in.empty:
                    valid_in = valid_in.copy()
                    valid_in['fast_out_amt'] = np.minimum(valid_in['in_amount'], valid_in['next_out_amount'])
                    fast_out_amt = valid_in['fast_out_amt'].sum()
                low_balance_count = ((in_rows['balance'].notna()) & (in_rows['balance'] < 0.2 * in_rows['in_amount'])).sum()
                return pd.Series({'total_in': total_in, 'fast_out_amt': fast_out_amt, 'low_balance_count': low_balance_count})

            chunk_summary = all_flows.groupby('account_id', group_keys=False).apply(fast_out_ratio_with_balance).reset_index()
            all_summary_chunks.append(chunk_summary)

        if not all_summary_chunks:
            print("警告：无有效交易块！")
            intermediaries_df = pd.DataFrame(columns=['account_id', 'fast_out_ratio', 'betweenness', 'pagerank', 'intermediary_score'])
        else:
            summary = pd.concat(all_summary_chunks, ignore_index=True)
            summary = summary.groupby('account_id').agg({
                'total_in': 'sum',
                'fast_out_amt': 'sum',
                'low_balance_count': 'sum'
            }).reset_index()
            summary['fast_out_ratio'] = summary['fast_out_amt'] / summary['total_in']
            in_counts = full_txn[
                valid_mask &
                (full_txn['account_id'].isin(candidate_nodes)) &
                (full_txn['direction'] == '1')
            ].groupby('account_id').size()
            summary = summary.set_index('account_id')
            summary['in_count'] = in_counts
            summary = summary.reset_index()
            summary['low_balance_retention'] = summary['low_balance_count'] / summary['in_count'].fillna(1)
            summary['fast_out_ratio'] = summary['fast_out_amt'] / summary['total_in']
            summary['low_balance_retention'] = summary['low_balance_count'] / (
                all_flows[all_flows['type'] == 'in'].groupby('account_id').size().reindex(summary['account_id'], fill_value=1).values
            )

            RUN_SENSITIVITY_ANALYSIS = False
            if RUN_SENSITIVITY_ANALYSIS:
                print("\n[DEBUG] 正在进行阈值敏感性分析...")
                import matplotlib.pyplot as plt

                fast_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
                low_balances = [0.3, 0.4, 0.5, 0.6, 0.7]
                total_ins = [50, 100, 200, 500, 1000]

                results = []
                base_f, base_lb, base_ti = 0.8, 0.5, 100

                for f in fast_ratios:
                    cnt = len(summary[
                        (summary['fast_out_ratio'] >= f) &
                        (summary['low_balance_retention'] >= base_lb) &
                        (summary['total_in'] >= base_ti) &
                        (summary['fast_out_ratio'] > 0)
                    ])
                    results.append({'param': 'fast_out_ratio', 'value': f, 'count': cnt})

                for lb in low_balances:
                    cnt = len(summary[
                        (summary['fast_out_ratio'] >= base_f) &
                        (summary['low_balance_retention'] >= lb) &
                        (summary['total_in'] >= base_ti) &
                        (summary['fast_out_ratio'] > 0)
                    ])
                    results.append({'param': 'low_balance_retention', 'value': lb, 'count': cnt})

                for ti in total_ins:
                    cnt = len(summary[
                        (summary['fast_out_ratio'] >= base_f) &
                        (summary['low_balance_retention'] >= base_lb) &
                        (summary['total_in'] >= ti) &
                        (summary['fast_out_ratio'] > 0)
                    ])
                    results.append({'param': 'total_in', 'value': ti, 'count': cnt})

                sens_df = pd.DataFrame(results)
                sens_df.to_csv(os.path.join(OUTPUT_DIR, 'threshold_sensitivity_analysis.csv'), index=False, encoding='utf-8-sig')
                print("  敏感性分析结果已保存至: threshold_sensitivity_analysis.csv")

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for i, param in enumerate(['fast_out_ratio', 'low_balance_retention', 'total_in']):
                    sub = sens_df[sens_df['param'] == param]
                    axes[i].plot(sub['value'], sub['count'], 'o-', linewidth=2, markersize=6)
                    axes[i].set_title(f'Impact of {param}')
                    axes[i].set_xlabel(param)
                    axes[i].set_ylabel('Candidate Account Count')
                    axes[i].grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_sensitivity_curves.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print("  敏感性曲线图已保存至: threshold_sensitivity_curves.png")
                print("  [DEBUG] 敏感性分析完成，程序终止。")
                exit(0)

            summary = summary[
                (summary['fast_out_ratio'] >= 0.6) &
                (summary['low_balance_retention'] >= 0.5) &
                (summary['total_in'] >= 50) &
                (summary['fast_out_ratio'] > 0)
            ]

            if summary.empty:
                intermediaries_df = pd.DataFrame(columns=['account_id', 'fast_out_ratio', 'betweenness', 'pagerank', 'intermediary_score'])
            else:
                print(f"有快出行为的账户数: {len(summary):,}")
                initial_candidates = summary['account_id'].tolist()
                print(f"初筛候选账户数: {len(initial_candidates)}")

                H_local = H_sub.subgraph(set(initial_candidates) | {nb for node in initial_candidates for nb in H_sub.neighbors(node)}).copy()
                print("计算局部 Betweenness 和 PageRank...")
                betweenness = nx.betweenness_centrality(H_local, weight='amount')
                pagerank = nx.pagerank(H_local, weight='amount')

                candidate_df = pd.DataFrame({'account_id': initial_candidates})
                candidate_df = candidate_df.merge(summary[['account_id', 'fast_out_ratio', 'total_in', 'low_balance_retention']], on='account_id')
                candidate_df['betweenness'] = candidate_df['account_id'].map(betweenness).fillna(0)
                candidate_df['pagerank'] = candidate_df['account_id'].map(pagerank).fillna(0)

                for col in ['fast_out_ratio', 'betweenness', 'pagerank', 'low_balance_retention']:
                    col_min, col_max = candidate_df[col].min(), candidate_df[col].max()
                    if col_max > col_min:
                        candidate_df[f'{col}_norm'] = (candidate_df[col] - col_min) / (col_max - col_min)
                    else:
                        candidate_df[f'{col}_norm'] = 0.0

                candidate_df['is_gray'] = candidate_df['account_id'].map(lambda x: 1 if account_label_map.get(x) == '灰' else 0)
                candidate_df['intermediary_score'] = (
                    candidate_df['fast_out_ratio_norm'] * 0.3 +
                    candidate_df['betweenness_norm'] * 0.3 +
                    candidate_df['pagerank_norm'] * 0.2 +
                    candidate_df['is_gray'] * 0.1 +
                    candidate_df['low_balance_retention_norm'] * 0.1
                )

                candidate_df = candidate_df.sort_values('intermediary_score', ascending=False).reset_index(drop=True)

                if len(candidate_df) < 3:
                    print("  候选中介账户不足3个，保留全部作为中介")
                    intermediaries_df = candidate_df[['account_id', 'fast_out_ratio', 'betweenness', 'pagerank', 'intermediary_score']].copy()
                else:
                    scores = candidate_df['intermediary_score'].values.astype(float)
                    n_candidates = len(scores)
                    x = np.arange(n_candidates)

                    from scipy.ndimage import gaussian_filter1d
                    scores_smooth = gaussian_filter1d(scores, sigma=2)

                    elbow_idx = None
                    try:
                        from kneed import KneeLocator
                        kn = KneeLocator(
                            x=x,
                            y=scores_smooth,
                            S=1.0,
                            curve='convex',
                            direction='decreasing'
                        )
                        elbow_idx = kn.knee
                        method_used = "Kneedle"
                    except ImportError:
                        method_used = "Second derivative"
                        print("  未安装 kneed 库，使用二阶导数拐点法")
                        dy = np.gradient(scores_smooth, x)
                        d2y = np.gradient(dy, x)
                        elbow_idx = np.argmin(d2y)
                    except Exception as e:
                        method_used = f"回退：异常 {e}"
                        print(f"  Kneedle 计算失败: {e}")
                        dy = np.gradient(scores_smooth, x)
                        d2y = np.gradient(dy, x)
                        elbow_idx = np.argmin(d2y)

                    if elbow_idx is None:
                        dy = np.gradient(scores_smooth, x)
                        d2y = np.gradient(dy, x)
                        elbow_idx = np.argmin(d2y)

                    max_allowed = int(n_candidates * 0.8)
                    if elbow_idx > max_allowed:
                        elbow_idx = max_allowed

                    min_selected = min(5, n_candidates)
                    selected_count = max(min_selected, elbow_idx + 1)

                    intermediaries_df = candidate_df.head(selected_count)[
                        ['account_id', 'fast_out_ratio', 'betweenness', 'pagerank', 'intermediary_score']
                    ].copy()

                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(10, 6))
                        plt.plot(x, scores, 'o-', alpha=0.5, label='Raw Score', markersize=3, color='steelblue')
                        plt.plot(x, scores_smooth, '-', color='red', linewidth=2, label='Smoothed Score')
                        plt.axvline(x=elbow_idx, color='green', linestyle='--', linewidth=2,
                                    label=f'Selected Threshold (n={selected_count})')
                        plt.title(f'Intermediary Score Distribution ({method_used})', fontsize=14)
                        plt.xlabel('Account Rank (by Score, Descending)')
                        plt.ylabel('Intermediary Score')
                        plt.legend()
                        plt.grid(True, linestyle='--', alpha=0.6)
                        plt.tight_layout()
                        plt.savefig(os.path.join(OUTPUT_DIR, 'intermediary_score_distribution.png'), dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"  已保存中介得分分布图至: intermediary_score_distribution.png")
                    except Exception as e:
                        print(f"  警告：绘图失败: {e}")

                    print(f"  最终选取中介账户: {selected_count} 个（总候选 {n_candidates}）")

if intermediaries_df.empty:
    intermediaries_df = pd.DataFrame(columns=['account_id', 'fast_out_ratio', 'betweenness', 'pagerank', 'intermediary_score'])
intermediaries_df.to_csv(os.path.join(OUTPUT_DIR, 'intermediary_accounts.csv'), index=False, encoding='utf-8-sig')
print(f"  识别中介账户 {len(intermediaries_df)} 个")

# ==================== 5.5. 保存2-hop风险子图（复用预计算的全局中心性） ====================
print("\n[5.5/8] 保存2-hop风险子图（含中介分数、中心性指标）...")

if 'H_sub' in locals() and H_sub.number_of_nodes() > 0:
    sub_txn = full_txn[
        (full_txn['account_id'].isin(H_sub.nodes())) |
        (full_txn['opponent_name'].isin(H_sub.nodes()))
    ]
    beh_summary_sub = sub_txn.groupby('account_id').agg(
        total_txn=('account_id', 'count'),
        total_amt=('amount', 'sum'),
        first_txn=('timestamp', 'min'),
        last_txn=('timestamp', 'max')
    ).reset_index()

    opp_beh = sub_txn.groupby('opponent_name').agg(
        total_txn=('opponent_name', 'count'),
        total_amt=('amount', 'sum'),
        first_txn=('timestamp', 'min'),
        last_txn=('timestamp', 'max')
    ).reset_index()
    opp_beh = opp_beh.rename(columns={'opponent_name': 'account_id'})

    all_beh = pd.concat([beh_summary_sub, opp_beh], ignore_index=True)
    all_beh = all_beh.groupby('account_id').agg({
        'total_txn': 'sum',
        'total_amt': 'sum',
        'first_txn': 'min',
        'last_txn': 'max'
    }).reset_index()

    nodes_sub = pd.DataFrame({'id': list(H_sub.nodes())})
    nodes_sub['is_account'] = nodes_sub['id'].isin(node_label_map)
    nodes_sub['account_label'] = np.where(
        nodes_sub['is_account'],
        nodes_sub['id'].map(node_label_map),
        nodes_sub['id'].map(opponent_label_map).fillna('未知')
    )
    nodes_sub = nodes_sub.merge(all_beh, left_on='id', right_on='account_id', how='left')
    nodes_sub['total_txn'] = nodes_sub['total_txn'].fillna(0).astype(int)
    nodes_sub['total_amt'] = nodes_sub['total_amt'].fillna(0.0)
    nodes_sub['first_txn'] = nodes_sub['first_txn'].fillna('').astype(str)
    nodes_sub['last_txn'] = nodes_sub['last_txn'].fillna('').astype(str)
    nodes_sub['community_id'] = nodes_sub['id'].map(community_df.set_index('account_id')['community_id']).fillna(-1).astype(int)
    nodes_sub['node_type'] = np.where(nodes_sub['is_account'], 'account', 'external')

    nodes_sub = nodes_sub.sort_values(['community_id', 'id']).reset_index(drop=True)
    nodes_sub['node_rank_in_community'] = nodes_sub.groupby('community_id').cumcount() + 1

    cid = nodes_sub['community_id'].astype(int)
    rank = nodes_sub['node_rank_in_community'].astype(int)
    node_type_str = np.where(nodes_sub['is_account'], '本行', '对手方')

    base_name = '社区' + np.where(cid == -1, '-1', cid.astype(str)) + ' 节点' + rank.astype(str) + ' ' + node_type_str
    nodes_sub['name'] = base_name

    intermediary_score_map = intermediaries_df.set_index('account_id')['intermediary_score'].to_dict() if not intermediaries_df.empty else {}
    pagerank_map = H_sub_pagerank

    nodes_sub['intermediary_score'] = nodes_sub['id'].map(intermediary_score_map).fillna(0.0)
    nodes_sub['pagerank'] = nodes_sub['id'].map(pagerank_map).fillna(0.0)
    nodes_sub['betweenness'] = 0.0

    for col in ['id', 'name', 'account_label', 'node_type', 'first_txn', 'last_txn']:
        nodes_sub[col] = nodes_sub[col].astype(str)
    for col in ['community_id', 'total_txn']:
        nodes_sub[col] = nodes_sub[col].astype(int)
    for col in ['total_amt', 'intermediary_score', 'pagerank', 'betweenness']:
        nodes_sub[col] = nodes_sub[col].astype(float)

    # 此时 full_txn 已有 src/tgt，不再报错
    # 先提取必要列，再筛选，避免大内存 copy
    needed_cols = ['src', 'tgt', 'amount', 'timestamp', 'direction']
    # 转为 set 提升 isin 效率
    sub_nodes_set = set(H_sub.nodes())

    edge_mask = (
            full_txn['src'].isin(sub_nodes_set) &
            full_txn['tgt'].isin(sub_nodes_set)
    )

    # 只选需要的列 + 应用 mask
    filtered_edges = full_txn.loc[edge_mask, needed_cols].copy()
    edges_sub = filtered_edges.rename(columns={'src': 'source', 'tgt': 'target'})

    edges_sub['weight'] = np.log1p(edges_sub['amount'])
    edges_sub['timestamp'] = edges_sub['timestamp'].astype(str).fillna('')
    edges_sub = edges_sub.to_dict('records')

    graph_data = {
        "nodes": nodes_sub[[
            'id', 'name', 'account_label', 'node_type', 'community_id',
            'intermediary_score', 'pagerank',
            'total_txn', 'total_amt', 'first_txn', 'last_txn'
        ]].to_dict('records'),
        "edges": edges_sub
    }

    with open(os.path.join(OUTPUT_DIR, 'risk_2hop_subgraph.json'), 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print("  2-hop风险子图已保存为: risk_2hop_subgraph.json")
else:
    print("  跳过2-hop子图保存（H_sub为空）")

# ==================== 6. 保存训练图结构 ====================
print("\n[6/8] 保存训练图结构（用于 XGBoost）...")
if G_train.number_of_nodes() > 0:
    community_id_map = community_df.set_index('account_id')['community_id'].to_dict() if not community_df.empty else {}
    node_label_map = account_label_map
    opponent_to_id = train_txn[['opponent_name', 'counterparty_id']].drop_duplicates()
    opponent_label_map = {
        row['opponent_name']: account_label_map.get(row['counterparty_id'], '未知')
        for _, row in opponent_to_id.iterrows()
    }
    risk_set = risk_account_ids_train

    all_nodes_list = pd.Series(list(G_train.nodes()), name='id')
    pred_df = pd.DataFrame([(v, u) for v in G_train.nodes() for u in G_train.predecessors(v)], columns=['id', 'pred'])
    succ_df = pd.DataFrame([(v, w) for v in G_train.nodes() for w in G_train.successors(v)], columns=['id', 'succ'])

    in_from_risk = pred_df[pred_df['pred'].isin(risk_set)].groupby('id').size()
    out_to_risk = succ_df[succ_df['succ'].isin(risk_set)].groupby('id').size()
    path_count = (in_from_risk + out_to_risk).fillna(0).astype(int)

    weighted_in_degree = pd.Series(dict(G_train.in_degree(weight='weight')), name='weighted_in_degree')
    pagerank_g = pd.Series(nx.pagerank(G_train, weight='weight') if G_train.number_of_nodes() > 0 else {}, name='pagerank')

    nodes_df = pd.DataFrame({'id': list(G_train.nodes())})
    nodes_df['account_label'] = nodes_df['id'].map(lambda x: node_label_map.get(x, opponent_label_map.get(x, '未知')))
    nodes_df['node_type'] = np.where(nodes_df['id'].isin(risk_account_ids_train), 'account', 'external')
    nodes_df['community_id'] = nodes_df['id'].map(community_id_map).fillna(-1).astype(int)
    nodes_df['intermediary_score'] = nodes_df['id'].map(intermediaries_df.set_index('account_id')['intermediary_score']).fillna(0.0)
    nodes_df = nodes_df.merge(weighted_in_degree, left_on='id', right_index=True, how='left')
    nodes_df = nodes_df.merge(pagerank_g, left_on='id', right_index=True, how='left')
    nodes_df['path_count'] = nodes_df['id'].map(path_count).fillna(0).astype(int)
    nodes_df = nodes_df.fillna({'weighted_in_degree': 0.0, 'pagerank': 0.0})

    nodes_data = nodes_df.to_dict('records')
    edges_data = [{'source': u, 'target': v, 'weight': float(d.get('weight', 0.0)), 'amount': float(d.get('amount', 0.0))}
                  for u, v, d in G_train.edges(data=True)]

    with open(os.path.join(OUTPUT_DIR, 'risk_propagation_graph_train.json'), 'w', encoding='utf-8') as f:
        json.dump({"nodes": nodes_data, "edges": edges_data}, f, ensure_ascii=False, indent=2)
    print("训练图结构已保存.")

# ==================== 7. 输出高危对手方 Top 100 ====================
print("\n[7/8] 输出高危对手方 Top 100...")
risk_list = list(risk_account_ids_train)
risk_flows = train_txn_clean.query("account_id in @risk_list or counterparty_id in @risk_list")

top_opponents = risk_flows.groupby('opponent_name').agg(
    transaction_count=('opponent_name', 'count'),
    total_amount=('amount', 'sum'),
    unique_risk_accounts=('account_id', 'nunique')
).sort_values('transaction_count', ascending=False).head(100)

top_opponents.to_csv(os.path.join(OUTPUT_DIR, 'high_risk_opponents.csv'), index=True, encoding='utf-8-sig')
print("高危对手方Top100已保存.")

# ==================== 8. 为交互式可视化系统保存全量图数据 ====================
print("\n[8/8] 为交互式可视化系统保存全量图数据...")

CHUNK_SIZE = 500000
edge_output_path = os.path.join(OUTPUT_DIR, 'full_graph_edges.csv')
pd.DataFrame(columns=['source', 'target', 'amount', 'timestamp', 'direction']).to_csv(
    edge_output_path, index=False, encoding='utf-8-sig'
)

print("  正在分块处理边数据...")
for chunk in pd.read_csv(FULL_TXN_PATH, encoding='utf-8-sig', low_memory=False, chunksize=CHUNK_SIZE):
    chunk = chunk[~chunk['opponent_name'].isin(NOISE)]
    if chunk.empty:
        continue
    chunk['source'] = np.where(chunk['direction'] == '2', chunk['account_id'], chunk['opponent_name'])
    chunk['target'] = np.where(chunk['direction'] == '2', chunk['opponent_name'], chunk['account_id'])
    edge_chunk = chunk[['source', 'target', 'amount', 'timestamp', 'direction']]
    edge_chunk.to_csv(edge_output_path, mode='a', header=False, index=False, encoding='utf-8-sig')

print("  边数据已保存至: full_graph_edges.csv")

print("  正在提取节点列表...")
all_nodes_set = set()
for chunk in pd.read_csv(edge_output_path, encoding='utf-8-sig', usecols=['source', 'target'], chunksize=CHUNK_SIZE):
    all_nodes_set.update(chunk['source'].dropna().astype(str))
    all_nodes_set.update(chunk['target'].dropna().astype(str))

nodes_df = pd.DataFrame({'id': list(all_nodes_set)})
nodes_df['is_account'] = nodes_df['id'].isin(account_label_map)
nodes_df['label'] = np.where(
    nodes_df['is_account'],
    nodes_df['id'].map(account_label_map),
    '未知'
)
nodes_df['node_type'] = np.where(nodes_df['is_account'], 'account', 'external')

print("  正在分块计算节点行为统计...")
beh_summary = pd.DataFrame()
for chunk in pd.read_csv(FULL_TXN_PATH, encoding='utf-8-sig', low_memory=False, chunksize=CHUNK_SIZE):
    chunk = chunk[~chunk['opponent_name'].isin(NOISE)]
    if chunk.empty:
        continue
    summary_chunk = chunk.groupby('account_id').agg(
        total_txn=('account_id', 'count'),
        total_amt=('amount', 'sum'),
        first_txn=('timestamp', 'min'),
        last_txn=('timestamp', 'max')
    ).reset_index()
    beh_summary = pd.concat([beh_summary, summary_chunk], ignore_index=True)

if not beh_summary.empty:
    beh_summary = beh_summary.groupby('account_id').agg({
        'total_txn': 'sum',
        'total_amt': 'sum',
        'first_txn': 'min',
        'last_txn': 'max'
    }).reset_index()
    nodes_df = nodes_df.merge(beh_summary.rename(columns={'account_id': 'id'}), on='id', how='left')

nodes_df['total_txn'] = nodes_df['total_txn'].fillna(0).astype(int)
nodes_df['total_amt'] = nodes_df['total_amt'].fillna(0.0)
nodes_df['first_txn'] = nodes_df['first_txn'].fillna('').astype(str)
nodes_df['last_txn'] = nodes_df['last_txn'].fillna('').astype(str)

node_output_path = os.path.join(OUTPUT_DIR, 'full_graph_nodes.csv')
nodes_df[['id', 'node_type', 'label', 'total_txn', 'total_amt', 'first_txn', 'last_txn']].to_csv(
    node_output_path, index=False, encoding='utf-8-sig'
)
print("  节点数据已保存至: full_graph_nodes.csv")

if 'community_stats' in locals() and not community_stats.empty:
    gang_summary = []
    for cid, row in community_stats.iterrows():
        members = community_df[community_df['community_id'] == cid]['account_id'].tolist()
        rep_nodes = members[:3]
        gang_summary.append({
            'community_id': int(cid),
            'total_nodes': int(row['total_nodes']),
            'risk_ratio': float(row['risk_ratio']),
            'risk_nodes': int(row['risk_nodes']),
            'representative_nodes': rep_nodes
        })
    with open(os.path.join(OUTPUT_DIR, 'gang_communities_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(gang_summary, f, ensure_ascii=False, indent=2)

print("交互式可视化系统数据准备完成！")
print(f"步骤2完成！总耗时 {time.time() - start:.2f} s")