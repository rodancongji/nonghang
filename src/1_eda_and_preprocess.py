# -*- coding: utf-8 -*-
"""
步骤1：数据探索与清洗（终极正确版）
功能：
  - 加载账户、交易、对手方对照表
  - 清洗无效交易 & 标准化字段
  - 用 '对收方名称' 正确映射对手方对照表，获取可读名称
  - 合并账户/对手方风险标签
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
    '对手账号': 'counterparty_id',
    '交易金额': 'amount',
    '收付标志（01收，02付）': 'direction',
    '交易渠道（1柜面，2网银，4atm，5pos，6手机银行，iwl网联，9其他）': 'channel',
    '备注': 'remark',
    '对手方所在地区': 'region',
    '代理人唯一id': 'agent_id'
}, inplace=True)

# 统一 direction 为字符串
txn_df['direction'] = txn_df['direction'].astype(str).str.strip()
print(f"direction 唯一值：{txn_df['direction'].unique()}")
print("收款交易（direction='1'）示例：")
print(txn_df[txn_df['direction']=='1'][['counterparty_id','对收方名称']].head())

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


# ==================== 4. 合并对手方名称（核心：用“对收方名称”匹配） ====================
print("\n[4/7] 合并对手方名称（使用 '对收方名称' → 'Encrypted'）...")

# 标准化对手方表字段
if 'Encrypted' in opponent_df.columns and 'TCNM' in opponent_df.columns:
    print("重命名对手方对照表字段...")
    opponent_df = opponent_df.rename(columns={'Encrypted': 'counterparty_id', 'TCNM': 'opponent_name'})
elif 'counterparty_id' not in opponent_df.columns or 'opponent_name' not in opponent_df.columns:
    raise ValueError("对手方表缺少必要字段，请检查！")

opponent_df = opponent_df[['counterparty_id', 'opponent_name']].drop_duplicates().reset_index(drop=True)
print(f"对手方表处理完成，共 {len(opponent_df):,} 行")

# 保留原始字段（'对收方名称' 实际是加密哈希）
txn_df['counterparty_name_hash'] = txn_df['对收方名称'].copy()

# 核心：用 '对收方名称' 匹配 opponent_df['counterparty_id']
print("执行映射合并...")
txn_df = txn_df.merge(
    opponent_df,
    left_on='对收方名称',
    right_on='counterparty_id',
    how='left',
    suffixes=('', '_mapped')
)

if 'opponent_name' not in txn_df.columns:
    raise KeyError("合并失败：未生成 opponent_name 字段")

txn_df.rename(columns={'opponent_name': 'mapped_opponent_name'}, inplace=True)

# 统计映射结果（无需比较名称，join 已确保键匹配）
mapped_count = txn_df['mapped_opponent_name'].notna().sum()
print(f"\n成功映射可读名称的交易数：{mapped_count:,} ({mapped_count/len(txn_df)*100:.2f}%)")

# 展示映射示例
print("\n=== 映射成功示例（前5条） ===")
print(txn_df[txn_df['mapped_opponent_name'].notna()][['对收方名称', 'mapped_opponent_name']].head())

# 填充未映射记录 & 创建最终字段
txn_df['mapped_opponent_name'] = txn_df['mapped_opponent_name'].fillna('未知')
txn_df['opponent_name'] = txn_df['mapped_opponent_name'].copy()

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

print(f"🏷标签合并完成：")
print(f"账户方灰产数：{(txn_df['account_label'] == '灰').sum():,}")
print(f"对手方灰产数：{(txn_df['counterparty_label'] == '灰').sum():,}")


# ==================== 6. 保存清洗后数据 ====================
print("\n[6/7] 保存清洗后数据...")

# 清理临时字段
temp_cols = ['counterparty_name_hash', 'mapped_opponent_name', '对收方名称']
drop_cols = [col for col in temp_cols if col in txn_df.columns]
if drop_cols:
    txn_df = txn_df.drop(columns=drop_cols)
    print(f"已删除临时字段：{drop_cols}")

# 保存
output_txn = 'D:/Pycharm/Intermediaries_digging/output/cleaned_transactions.csv'
output_acc = 'D:/Pycharm/Intermediaries_digging/output/cleaned_accounts.csv'

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

print("\n 数据预处理完成！可进入图分析阶段。")

# 快速查看灰产账户主要交易对手
gray_flows = txn_df[txn_df['account_label'] == '灰']['opponent_name'].value_counts().head(10)
print("灰产账户主要交易对手：")
print(gray_flows)