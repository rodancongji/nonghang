# src/1_eda_and_preprocess.py
import pandas as pd
from datetime import datetime

print("=== 步骤1：数据探索与清洗（终极正确版：保留双向资金流）===")

# ==================== 1. 加载数据 ====================
print("\n正在加载数据...")

# 账户名单
account_df = pd.read_csv('D:/Pycharm/Intermediaries_digging/data/名单_脱敏后.csv', encoding='UTF-8-SIG')
print(f"账户表加载完成：{len(account_df)} 行，字段：{list(account_df.columns)}")

# 交易流水
txn_df = pd.read_csv('D:/Pycharm/Intermediaries_digging/data/流水_脱敏后.csv', encoding='UTF-8-SIG')
print(f"交易表加载完成：{len(txn_df)} 行，字段：{list(txn_df.columns)}")

# 对手方对照表
opponent_df = pd.read_excel('D:/Pycharm/Intermediaries_digging/data/对手方对照.xlsx')
print(f"对手方表加载完成：{len(opponent_df)} 行，字段：{list(opponent_df.columns)}")

# ==================== 2. 基础统计 ====================
print("\n基础统计信息：")

print(f"账户总数：{len(account_df)}")
print(f"唯一卡号数：{account_df['卡号'].nunique()}")
print(f"标签分布：\n{account_df['label'].value_counts()}")

print(f"\n交易总数：{len(txn_df)}")
print(f"时间范围：{txn_df['时间'].min()} 到 {txn_df['时间'].max()}")

# ==================== 3. 数据清洗与标准化（保留双向交易！） ====================
print("\n数据清洗与字段标准化（保留 direction=1 和 2）...")

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

print(f"清洗后有效交易数：{len(txn_df)}")
print(f"其中付款交易（direction=2）：{len(txn_df[txn_df['direction']=='2'])}")
print(f"其中收款交易（direction=1）：{len(txn_df[txn_df['direction']=='1'])}")

# ==================== 4. 合并对手方名称 ====================
print("\n🔗 合并对手方名称...")

opponent_df.rename(columns={'Encrypted': 'counterparty_id', 'TCNM': 'opponent_name'}, inplace=True)
txn_df = txn_df.merge(opponent_df[['counterparty_id', 'opponent_name']], on='counterparty_id', how='left')
txn_df['opponent_name'] = txn_df['opponent_name'].fillna('未知')

# ==================== 5. 合并账户风险标签 ====================
print("\n合并账户风险标签...")

account_df.rename(columns={'客户唯一id': 'account_id'}, inplace=True)

# 合并 account_id 的标签
txn_df = txn_df.merge(
    account_df[['account_id', 'label']].rename(columns={'label': 'account_label'}),
    on='account_id', how='left'
)

# 合并 counterparty_id 的标签（如果存在）
txn_df = txn_df.merge(
    account_df[['account_id', 'label']].rename(columns={'account_id': 'counterparty_id', 'label': 'counterparty_label'}),
    on='counterparty_id', how='left'
)

txn_df['account_label'] = txn_df['account_label'].fillna('未知')
txn_df['counterparty_label'] = txn_df['counterparty_label'].fillna('未知')

print(f"标签合并完成：")
print(f"账户方灰产数：{(txn_df['account_label'] == '灰').sum()}")
print(f"对手方灰产数：{(txn_df['counterparty_label'] == '灰').sum()}")

# ==================== 6. 保存清洗后数据 ====================
print("\n保存清洗后数据（含双向交易）...")
txn_df = pd.read_csv('D:/Pycharm/Intermediaries_digging/data/流水_脱敏后.csv',encoding='UTF-8-SIG',dtype={'代理人唯一id': str},low_memory=False)
account_df.to_csv('D:/Pycharm/Intermediaries_digging/output/cleaned_accounts.csv', index=False, encoding='utf-8-sig')

print(f"交易数据已保存至：D:/Pycharm/Intermediaries_digging/output/cleaned_transactions.csv")
print(f"账户数据已保存至：D:/Pycharm/Intermediaries_digging/output/cleaned_accounts.csv")

# ==================== 7. 字段映射说明 ====================
print("\n重要字段说明（后续图构建关键！）：")
print("在构建图时：")
print("- 若 direction == '2'（付款） → 边方向：account_id → counterparty_id")
print("- 若 direction == '1'（收款） → 边方向：counterparty_id → account_id （资金流入方向）")
print("边属性应包含：amount, direction, timestamp, channel, opponent_name")
print("节点属性应包含：account_label, counterparty_label")
