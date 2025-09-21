# -*- coding: utf-8 -*-
"""
步骤1：数据探索与清洗（终极修复版）
功能：
  - 加载账户、交易、对手方对照表
  - 清洗无效交易 & 标准化字段
  - 用 '对收方名称' 初步映射，再按 counterparty_id 稳定化对手方名称（避免节点分裂）
  - 合并账户/对手方风险标签
  - 分析对手方与账户ID对应关系（为图结构做准备）
  - 保存清洗后数据，供后续图构建使用
"""

import pandas as pd
from datetime import datetime

print("=== 步骤1：数据探索与清洗===")

# ==================== 1. 加载数据 ====================
print("\n[1/7] 正在加载基础数据...")

# 账户名单
account_df = pd.read_csv('D:/Pycharm/Intermediaries_digging/data/名单_脱敏后.csv', encoding='UTF-8-SIG')
print(f"账户表加载完成：{len(account_df)} 行，字段：{list(account_df.columns)}")

# 交易流水
txn_df = pd.read_csv(
    'D:/Pycharm/Intermediaries_digging/data/流水_脱敏后.csv',
    encoding='UTF-8-SIG',
    dtype={'代理人唯一id': str},
    low_memory=False
)
print(f"交易表加载完成：{len(txn_df)} 行，字段：{list(txn_df.columns)}")

# 对手方对照表
opponent_df = pd.read_excel('D:/Pycharm/Intermediaries_digging/data/对手方对照.xlsx')
print(f"对手方表加载完成：{len(opponent_df)} 行，字段：{list(opponent_df.columns)}")


# ==================== 2. 基础统计 ====================
print("\n[2/7] 基础统计信息：")

print(f"账户总数：{len(account_df)}")
print(f"唯一卡号数：{account_df['卡号'].nunique()}")
print(f"标签分布：\n{account_df['label'].value_counts()}")

print(f"\n交易总数：{len(txn_df)}")
print(f" 时间范围：{txn_df['时间'].min()} 到 {txn_df['时间'].max()}")


# ==================== 3. 数据清洗与标准化 ====================
print("\n[3/7] 数据清洗与字段标准化（保留 direction=1 和 2）...")

# 重命名关键字段
txn_df.rename(columns={
    '时间': 'timestamp',
    '客户唯一id': 'account_id',
    '卡号': 'card_number',
    '现金标志（00现金，01转账）': 'cash',
    '收付标志（01收，02付）': 'direction',
    '交易金额': 'amount',
    '交易后余额': 'balance',
    '交易渠道（1柜面，2网银，4atm，5pos，6手机银行，iwl网联，9其他）': 'channel',
    '对手账号': 'counterparty_id',
    '对收方名称':'counterparty_name',
    '备注': 'remark',
    '对手方所在地区': 'region',
    '代理人唯一id': 'agent_id'
}, inplace=True)

# 统一 direction 为字符串
txn_df['direction'] = txn_df['direction'].astype(str).str.strip()
print(f"direction 唯一值：{txn_df['direction'].unique()}")
print("收款交易（direction='1'）示例：")
print(txn_df[txn_df['direction']=='1'][['counterparty_id','counterparty_name']].head())

# 时间 & 金额标准化
txn_df['timestamp'] = pd.to_datetime(txn_df['timestamp'], errors='coerce')
txn_df['amount'] = pd.to_numeric(txn_df['amount'], errors='coerce')

# 清洗无效记录
txn_df = txn_df[
    (txn_df['amount'] > 0) &
    (txn_df['account_id'].notna()) &
    (txn_df['counterparty_id'].notna()) &
    (txn_df['timestamp'].notna()) &
    (txn_df['direction'].isin(['1', '2']))
].copy()

print(f"清洗后有效交易数：{len(txn_df):,}")
print(f"付款交易（direction=2）：{len(txn_df[txn_df['direction']=='2']):,}")
print(f"收款交易（direction=1）：{len(txn_df[txn_df['direction']=='1']):,}")


# ==================== 4. 初步合并对手方名称（使用 'counterparty_name' → 'Encrypted'） ====================
print("\n[4/7] 初步合并对手方名称（使用 'counterparty_name' → 'Encrypted'）...")

# 标准化对手方表字段
if 'Encrypted' in opponent_df.columns and 'TCNM' in opponent_df.columns:
    print("重命名对手方对照表字段...")
    opponent_df = opponent_df.rename(columns={'Encrypted': 'counterparty_id', 'TCNM': 'opponent_name'})
elif 'counterparty_id' not in opponent_df.columns or 'opponent_name' not in opponent_df.columns:
    raise ValueError("对手方表缺少必要字段，请检查！")

opponent_df = opponent_df[['counterparty_id', 'opponent_name']].drop_duplicates().reset_index(drop=True)
print(f"对手方表处理完成，共 {len(opponent_df):,} 行")

# 保留原始字段
txn_df['counterparty_name_hash'] = txn_df['counterparty_name'].copy()

# 核心：用 '对收方名称' 匹配 opponent_df['counterparty_id']
print("执行初步映射合并...")
txn_df = txn_df.merge(
    opponent_df,
    left_on='counterparty_name',
    right_on='counterparty_id',
    how='left',
    suffixes=('', '_mapped')
)

if 'opponent_name' not in txn_df.columns:
    raise KeyError("合并失败：未生成 opponent_name 字段")

txn_df.rename(columns={'opponent_name': 'mapped_opponent_name'}, inplace=True)

# 统计映射结果
mapped_count = txn_df['mapped_opponent_name'].notna().sum()
print(f"\n初步成功映射可读名称的交易数：{mapped_count:,} ({mapped_count/len(txn_df)*100:.2f}%)")

# 展示映射示例
print("\n=== 初步映射成功示例（前5条） ===")
print(txn_df[txn_df['mapped_opponent_name'].notna()][['counterparty_name', 'mapped_opponent_name','account_id']].head())


# ==================== 4.5/7 修复：重建稳定对手方名称映射（按 counterparty_id 取众数） ====================
print("\n[4.5/7] 重建稳定对手方名称映射（按 counterparty_id 取最频繁名称，避免图节点分裂）...")

# 步骤1：收集所有已知映射（包括初步映射和原始字段）
# 创建一个映射字典：counterparty_id -> 最常出现的 opponent_name
all_name_mappings = txn_df[['counterparty_id', 'mapped_opponent_name']].copy()
all_name_mappings['mapped_opponent_name'] = all_name_mappings['mapped_opponent_name'].fillna('未知')

# 按 counterparty_id 分组，取出现频率最高的名称
stable_name_map = (
    all_name_mappings.groupby('counterparty_id')['mapped_opponent_name']
    .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else '未知')
    .to_dict()
)

# 步骤2：应用稳定映射
txn_df['opponent_name'] = txn_df['counterparty_id'].map(stable_name_map).fillna('未知')

print(f" 稳定映射完成。唯一对手方ID数：{len(stable_name_map):,}")

# 步骤3：为“未知”类生成可区分名称（避免所有未知合并成一个节点）
print(">>> 为 '未知' 类对手方生成区分ID（基于 counterparty_id 前8位）...")
txn_df.loc[txn_df['opponent_name'] == '未知', 'opponent_name'] = \
    'Unknown_' + txn_df.loc[txn_df['opponent_name'] == '未知', 'counterparty_id'].str[:8]

print(" 对手方名称已稳定化并区分匿名实体，可安全用于图构建。")

# 打印最终 opponent_name 分布（前15）
print("\n最终 opponent_name 分布（前15）：")
print(txn_df['opponent_name'].value_counts().head(15))


# ==================== 5. 合并账户风险标签 ====================
print("\n[5/7] 合并账户风险标签...")

if '客户唯一id' in account_df.columns:
    account_df.rename(columns={'客户唯一id': 'account_id'}, inplace=True)

# 合并发起方标签
txn_df = txn_df.merge(
    account_df[['account_id', 'label']].rename(columns={'label': 'account_label'}),
    on='account_id',
    how='left'
)

# 合并对手方标签（如果对手方也是名单内账户）
txn_df = txn_df.merge(
    account_df[['account_id', 'label']].rename(columns={'account_id': 'counterparty_id', 'label': 'counterparty_label'}),
    on='counterparty_id',
    how='left'
)

txn_df['account_label'] = txn_df['account_label'].fillna('未知')
txn_df['counterparty_label'] = txn_df['counterparty_label'].fillna('未知')

print(f"标签合并完成：")
print(f"账户方灰产数：{(txn_df['account_label'] == '灰').sum():,}")
print(f"账户方黑次次次密接数：{(txn_df['account_label'] == '黑次次次密接').sum():,}")
print(f"账户方黑次次密接数：{(txn_df['account_label'] == '黑次次密接').sum():,}")
print(f"账户方黑次密接数：{(txn_df['account_label'] == '黑次密接').sum():,}")
print(f"账户方黑密接数：{(txn_df['account_label'] == '黑密接').sum():,}")
print(f"账户方黑产数：{(txn_df['account_label'] == '黑').sum():,}")
print(f"账户方未知数：{(txn_df['account_label'] == '未知').sum():,}")

print(f"对手方灰产数：{(txn_df['counterparty_label'] == '灰').sum():,}")
print(f"对手方黑次次次密接数：{(txn_df['counterparty_label'] == '黑次次次密接').sum():,}")
print(f"对手方黑次次密接数：{(txn_df['counterparty_label'] == '黑次次密接').sum():,}")
print(f"对手方黑次密接数：{(txn_df['counterparty_label'] == '黑次密接').sum():,}")
print(f"对手方黑密接数：{(txn_df['counterparty_label'] == '黑密接').sum():,}")
print(f"对手方黑产数：{(txn_df['counterparty_label'] == '黑').sum():,}")
print(f"对手方未知数：{(txn_df['counterparty_label'] == '未知').sum():,}")


# ==================== 5.5/7 分析 counterparty_name, counterparty_id, account_id 对应关系 ====================
print("\n[5.5/7] 对手方与账户ID对应关系分析（为图结构做准备）...")

# 1. 检查有多少对手方 ID 同时出现在 account_id 中（即：对手方是否也是“名单内账户”）
internal_counterparties = txn_df[
    txn_df['counterparty_id'].isin(account_df['account_id'])
]
print(f"\n>>> 内部交易（对手方也是名单内账户）交易数：{len(internal_counterparties):,} ({len(internal_counterparties)/len(txn_df)*100:.2f}%)")

# 展示部分内部交易示例
print("\n=== 内部交易示例（前5条）===")
if len(internal_counterparties) > 0:
    print(internal_counterparties[['account_id', 'counterparty_id', 'opponent_name', 'amount', 'direction']].head())
else:
    print("无内部交易记录。")

# 2. 检查 account_id == counterparty_id 的自环交易（自己转给自己）
self_loops = txn_df[txn_df['account_id'] == txn_df['counterparty_id']]
print(f"\n>>> 自环交易（account_id == counterparty_id）数：{len(self_loops):,}")
if len(self_loops) > 0:
    print("自环交易示例：")
    print(self_loops[['account_id', 'counterparty_id', 'opponent_name', 'amount', 'direction']].head())

# 3. 分析每个 account_id 最常交易的 counterparty_name（Top 3）—— 内存安全版
print(f"\n>>> 每个账户最频繁交易的对手方（Top 3）示例（展示前5个账户）：")

# 手动遍历分组，避免 Pandas apply + dict 崩溃
top_counterparties_list = []
grouped = txn_df.groupby('account_id')['opponent_name']

# 只取前 1000 个账户避免性能问题
sample_accounts = txn_df['account_id'].unique()[:1000]

for account in sample_accounts:
    name_series = grouped.get_group(account)
    top3_dict = name_series.value_counts().head(3).to_dict()
    top_counterparties_list.append({
        'account_id': account,
        'top_counterparties': top3_dict
    })

# 转为 DataFrame
top_counterparties_per_account = pd.DataFrame(top_counterparties_list)

# 展示前5个
for _, row in top_counterparties_per_account.head(5).iterrows():
    print(f"账户 {row['account_id']} → {row['top_counterparties']}")

# 4. 统计高频对手方名称及其关联的 counterparty_id 数量（检查名称是否唯一映射ID）
print(f"\n>>> 高频对手方名称及其关联的不同 counterparty_id 数量（检查歧义）：")
name_to_id_mapping = (
    txn_df.groupby('opponent_name')['counterparty_id']
    .nunique()
    .sort_values(ascending=False)
    .head(15)
)
print(name_to_id_mapping)

# 5. 反向：一个 counterparty_id 是否对应多个 opponent_name？（理论上不应该，除非映射错误）
# → 现在应该基本为1，因为我们已经稳定化了映射
print(f"\n>>> 检查稳定化后：一个 counterparty_id 是否仍对应多个 opponent_name？")
id_to_name_mapping = (
    txn_df.groupby('counterparty_id')['opponent_name']
    .nunique()
    .sort_values(ascending=False)
)
multi_name_ids = id_to_name_mapping[id_to_name_mapping > 1]
if len(multi_name_ids) > 0:
    print(f" 仍发现 {len(multi_name_ids)} 个 counterparty_id 对应多个名称（应极少），示例：")
    example_id = multi_name_ids.index[0]
    print(txn_df[txn_df['counterparty_id'] == example_id][['counterparty_id', 'opponent_name']].drop_duplicates())
else:
    print(" 所有 counterparty_id 均唯一对应一个 opponent_name（映射稳定）。")

# 6. 查看灰产账户是否频繁与特定对手方交易
print(f"\n>>> 灰产账户（黑/灰）最频繁交易的 Top 5 对手方：")
risk_labels = ['黑', '黑密接', '黑次密接', '黑次次密接', '黑次次次密接', '灰']
risk_flows = txn_df[txn_df['account_label'].isin(risk_labels)]
top_risk_opponents = risk_flows['opponent_name'].value_counts().head(5)
print(top_risk_opponents)


# ==================== 6. 保存清洗后数据 ====================
print("\n[6/7] 保存清洗后数据...")

# 清理临时字段
temp_cols = ['counterparty_name_hash', 'mapped_opponent_name']
drop_cols = [col for col in temp_cols if col in txn_df.columns]
if drop_cols:
    txn_df = txn_df.drop(columns=drop_cols)
    print(f"已删除临时字段：{drop_cols}")

# 保存
output_txn = 'D:/Pycharm/Intermediaries_digging/data/cleaned_transactions.csv'
output_acc = 'D:/Pycharm/Intermediaries_digging/data/cleaned_accounts.csv'

txn_df.to_csv(output_txn, index=False, encoding='utf-8-sig')
account_df.drop_duplicates('account_id').to_csv(output_acc, index=False, encoding='utf-8-sig')

print(f"交易数据已保存至：{output_txn}")
print(f"账户数据已保存至：{output_acc}")


# ==================== 7. 字段说明 ====================
print("\n[7/7] 重要字段说明（图构建用）：")
print("边方向：")
print("   - direction == '2'（付款）: account_id → counterparty_id")
print("   - direction == '1'（收款）: counterparty_id → account_id")
print("边属性：amount, direction, timestamp, channel, opponent_name")
print("节点属性：account_label, counterparty_label")
print("节点类型：")
print("   - 若 counterparty_id 在 account_df 中 → 节点类型 = 'account'")
print("   - 否则 → 节点类型 = 'external'")

print("\n 数据预处理完成！图结构已优化，可进入图分析阶段。")


# ==================== 8. 补充：灰产账户主要交易对手 Top 10（按类别） ====================
print("\n>>> 各类灰产账户主要交易对手 Top 10：")
for label in ['黑', '黑密接', '黑次密接', '黑次次密接', '黑次次次密接', '灰']:
    flows = txn_df[txn_df['account_label'] == label]['opponent_name'].value_counts().head(10)
    print(f"\n【{label}】账户 Top 10 交易对手：")

    print(flows)
  """D:\Anaconda\envs\RAG_Learning\python.exe D:\Pycharm\Intermediaries_digging\src\1_eda_and_preprocess.py 
=== 步骤1：数据探索与清洗===

[1/7] 正在加载基础数据...
账户表加载完成：54166 行，字段：['客户唯一id', '卡号', 'label']
交易表加载完成：10293076 行，字段：['时间', '客户唯一id', '卡号', '现金标志（00现金，01转账）', '收付标志（01收，02付）', '交易金额', '交易后余额', '交易渠道（1柜面，2网银，4atm，5pos，6手机银行，iwl网联，9其他）', '对手账号', '对收方名称', '备注', '对手方所在地区', '代理人唯一id']
对手方表加载完成：385 行，字段：['TCNM', 'Encrypted']

[2/7] 基础统计信息：
账户总数：54166
唯一卡号数：53761
标签分布：
label
无关        47183
灰          6057
黑次次次密接      365
黑           218
黑次次密接       185
黑次密接        115
黑密接          43
Name: count, dtype: int64

交易总数：10293076
 时间范围：2023-10-05 11:24:18 到 2025-07-07 23:59:42

[3/7] 数据清洗与字段标准化（保留 direction=1 和 2）...
direction 唯一值：['2' '1']
收款交易（direction='1'）示例：
                                      counterparty_id                                  counterparty_name
27  8efa690df7f19b42a328d9f2a612f6c9e396957400c2c4...  50f2e8bcef5d1044f7d4ad85d3c1a5f98b6d74a7769ec4...
28  0b7b7d10a5b08a913486f2bbcbf61aec83b547daf964d2...  3c8da0c99f7ad402d7126dc1348c06a1138d8ab30ef5cc...
34  0287067ebc5d7080b93d8ad8d45eab7465bf64a42a0c7e...  c8c9bf511d23b46e73302a24393c05d14fe2e7f1224d6d...
61  e35b7b14784a97f890ff0d81b84ecbc21d37d0e803b007...  47bd88b1df58741b4c0d01ec0fe30b9b4ff19f989d5d43...
62  e35b7b14784a97f890ff0d81b84ecbc21d37d0e803b007...  47bd88b1df58741b4c0d01ec0fe30b9b4ff19f989d5d43...
清洗后有效交易数：10,289,784
付款交易（direction=2）：8,443,602
收款交易（direction=1）：1,846,182

[4/7] 初步合并对手方名称（使用 'counterparty_name' → 'Encrypted'）...
重命名对手方对照表字段...
对手方表处理完成，共 385 行
执行初步映射合并...

初步成功映射可读名称的交易数：5,346,698 (51.96%)

=== 初步映射成功示例（前5条） ===
                                    counterparty_name  ...                                         account_id
0   8712439ff4dc7a03d9a2b70d37a46eb44f3c253e1e4b07...  ...  b9897fc4620c5a518177c1a220e489dc0e37b2b3970af1...
2   1b4a47a818469a86d4dbe727a6591f627a4bbad5b0b5a3...  ...  c5b8beeec0e9c9d93c063715a536ac73b0bc69fe7a2ca6...
3   1b4a47a818469a86d4dbe727a6591f627a4bbad5b0b5a3...  ...  c5b8beeec0e9c9d93c063715a536ac73b0bc69fe7a2ca6...
4   8712439ff4dc7a03d9a2b70d37a46eb44f3c253e1e4b07...  ...  c5b8beeec0e9c9d93c063715a536ac73b0bc69fe7a2ca6...
17  65b00e0405651a9561afe6c746210b6bf66f981a82f71c...  ...  74a2b2204a9586952a90ad3ec6e7a8c414f48bf6409777...

[5 rows x 3 columns]

[4.5/7] 重建稳定对手方名称映射（按 counterparty_id 取最频繁名称，避免图节点分裂）...
 稳定映射完成。唯一对手方ID数：1,036,725
>>> 为 '未知' 类对手方生成区分ID（基于 counterparty_id 前8位）...
 对手方名称已稳定化并区分匿名实体，可安全用于图构建。

最终 opponent_name 分布（前15）：
opponent_name
微信转账                1386684
扫二维码付款              1324772
拼多多平台商户              332147
Unknown_0621202f     330377
微信红包                 169755
微信零钱充值账户             163641
上海格物致品网络科技有限公司       149120
Unknown_3f3e38fb     114892
美团平台商户               102676
蚂蚁（杭州）基金销售有限公司        88304
抖店平台商户                74727
手机充值                  71294
还款                    67309
抖音支付科技有限公司            43604
零钱通                   43499
Name: count, dtype: int64

[5/7] 合并账户风险标签...
标签合并完成：
账户方灰产数：5,084,482
账户方黑次次次密接数：156,331
账户方黑次次密接数：71,890
账户方黑次密接数：50,371
账户方黑密接数：18,952
账户方黑产数：57,473
账户方未知数：1
对手方灰产数：0
对手方黑次次次密接数：0
对手方黑次次密接数：0
对手方黑次密接数：0
对手方黑密接数：0
对手方黑产数：0
对手方未知数：13,504,591

[5.5/7] 对手方与账户ID对应关系分析（为图结构做准备）...

>>> 内部交易（对手方也是名单内账户）交易数：2 (0.00%)

=== 内部交易示例（前5条）===
                                                account_id  ... direction
6124492  aa77e2c05436c8ae0a997207c502ae133e7c43073b74ad...  ...         2
6124494  aa77e2c05436c8ae0a997207c502ae133e7c43073b74ad...  ...         1

[2 rows x 5 columns]

>>> 自环交易（account_id == counterparty_id）数：2
自环交易示例：
                                                account_id  ... direction
6124492  aa77e2c05436c8ae0a997207c502ae133e7c43073b74ad...  ...         2
6124494  aa77e2c05436c8ae0a997207c502ae133e7c43073b74ad...  ...         1

[2 rows x 5 columns]

>>> 每个账户最频繁交易的对手方（Top 3）示例（展示前5个账户）：
账户 b9897fc4620c5a518177c1a220e489dc0e37b2b3970af1bca4cf1195c5fef3a7 → {'微信转账': 454, '扫二维码付款': 140, 'Unknown_12105a1a': 55}
账户 a9d06b83413e8e98b8ce9f7328c8aae2e4b118968ec8ac04e684a6c1ce058f88 → {'扫二维码付款': 844, '微信转账': 584, 'Unknown_b8b2021a': 278}
账户 c5b8beeec0e9c9d93c063715a536ac73b0bc69fe7a2ca615d7475d79e1c0d249 → {'微信转账': 730, 'Unknown_3a25a656': 544, '扫二维码付款': 470}
账户 25397865ae3b30fc29ded9cb70838ce152102cef9810252d7d7b4227e22385a7 → {'Unknown_e35b7b14': 50, '微信转账': 33, '扫二维码付款': 28}
账户 74a2b2204a9586952a90ad3ec6e7a8c414f48bf640977702678dba65279036ff → {'Unknown_60a1a1a0': 455, '微信转账': 392, 'Unknown_cbcf7190': 345}

>>> 高频对手方名称及其关联的不同 counterparty_id 数量（检查歧义）：
opponent_name
扫二维码付款            247272
微信零钱充值账户            7717
淘宝平台                5267
视频号小店               3722
快手小店平台商户            1700
特约商户                1453
网商银行数字人民币            608
我的钱包                 579
支付宝小荷包-自动攒           450
支付宝小荷包-转入            335
蚂蚁（杭州）基金销售有限公司       208
北京微播视界科技有限公司         173
生活缴费                 101
待报解预算收入               84
无人售货机                 74
Name: counterparty_id, dtype: int64

>>> 检查稳定化后：一个 counterparty_id 是否仍对应多个 opponent_name？
 所有 counterparty_id 均唯一对应一个 opponent_name（映射稳定）。

>>> 灰产账户（黑/灰）最频繁交易的 Top 5 对手方：
opponent_name
微信转账                884914
扫二维码付款              442510
Unknown_0621202f    149589
拼多多平台商户              78250
微信红包                 74432
Name: count, dtype: int64

[6/7] 保存清洗后数据...
已删除临时字段：['counterparty_name_hash', 'mapped_opponent_name']
交易数据已保存至：D:/Pycharm/Intermediaries_digging/data/cleaned_transactions.csv
账户数据已保存至：D:/Pycharm/Intermediaries_digging/data/cleaned_accounts.csv

[7/7] 重要字段说明（图构建用）：
边方向：
   - direction == '2'（付款）: account_id → counterparty_id
   - direction == '1'（收款）: counterparty_id → account_id
边属性：amount, direction, timestamp, channel, opponent_name
节点属性：account_label, counterparty_label
节点类型：
   - 若 counterparty_id 在 account_df 中 → 节点类型 = 'account'
   - 否则 → 节点类型 = 'external'

 数据预处理完成！图结构已优化，可进入图分析阶段。

>>> 各类灰产账户主要交易对手 Top 10：

【黑】账户 Top 10 交易对手：
opponent_name
微信转账                9107
扫二维码付款              4776
Unknown_0621202f    1747
微信红包                1450
微信零钱充值账户            1199
上海格物致品网络科技有限公司      1124
北京快手科技有限公司          1031
拼多多平台商户              964
Unknown_1aa22eb9     897
美团平台商户               714
Name: count, dtype: int64

【黑密接】账户 Top 10 交易对手：
opponent_name
微信转账                2636
扫二维码付款              1097
Unknown_0621202f     801
Unknown_0f33c0ab     623
Unknown_3b156edb     412
Unknown_23acc243     366
上海格物致品网络科技有限公司       337
拼多多平台商户              336
美团平台商户               302
微信红包                 295
Name: count, dtype: int64

【黑次密接】账户 Top 10 交易对手：
opponent_name
微信转账                       6082
扫二维码付款                     5480
拼多多平台商户                    1712
Unknown_0621202f           1686
上海格物致品网络科技有限公司             1028
蚂蚁（杭州）基金销售有限公司              576
微信红包                        529
平安银行电子商务交易资金待清算专户（３１６４）     464
Unknown_0d2711a2            457
天弘基金管理有限公司                  415
Name: count, dtype: int64

【黑次次密接】账户 Top 10 交易对手：
opponent_name
微信转账                9757
扫二维码付款              6110
Unknown_0621202f    2569
上海格物致品网络科技有限公司      1058
拼多多平台商户             1045
微信红包                 924
Unknown_3f3e38fb     903
微信零钱充值账户             758
Unknown_0f33c0ab     623
美团平台商户               600
Name: count, dtype: int64

【黑次次次密接】账户 Top 10 交易对手：
opponent_name
微信转账                22989
扫二维码付款              15122
Unknown_0621202f     6347
拼多多平台商户              3333
上海格物致品网络科技有限公司       2283
蚂蚁（杭州）基金销售有限公司       1710
微信红包                 1673
Unknown_3f3e38fb     1424
微信零钱充值账户             1238
美团平台商户               1223
Name: count, dtype: int64

【灰】账户 Top 10 交易对手：
opponent_name
微信转账                834343
扫二维码付款              409925
Unknown_0621202f    136439
拼多多平台商户              70860
微信红包                 69561
Unknown_3f3e38fb     45690
微信零钱充值账户             42138
蚂蚁（杭州）基金销售有限公司       39439
美团平台商户               36001
上海格物致品网络科技有限公司       29906
Name: count, dtype: int64

Process finished with exit code 0"""
