import warnings
warnings.simplefilter('ignore')

import os
import gc

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error

import lightgbm as lgb

from matplotlib.pyplot import plot, show

# 玩家角色表

roles = pd.read_csv('datasets/role_id.csv')

# 共七天, roles 表填充完整
dfs = []
for i in range(2, 9):
    tmp = roles.copy()
    tmp['day'] = i
    dfs.append(tmp)
data = pd.concat(dfs).reset_index(drop=True)


# 货币消耗表
consume = pd.read_csv('datasets/role_consume_op.csv')
consume['dt'] = pd.to_datetime(consume['dt'])
consume['day'] = consume['dt'].dt.day

# 货币消耗按天合并
# TODO: mtime 可以做时差衍生特征, 其他表也是一样的
for i in range(1, 5):
    for m in ['count', 'sum']:
        tmp = consume.groupby(['role_id', 'day'])[f'use_t{i}'].agg(m).to_frame(name=f'use_t{i}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

# 升级表

evolve = pd.read_csv('datasets/role_evolve_op.csv')
evolve['dt'] = pd.to_datetime(evolve['dt'])
evolve['day'] = evolve['dt'].dt.day
evolve['n_level_up'] = evolve['new_lv'] - evolve['old_lv']
evolve = evolve.rename(columns={'num': 'lv_consume_item_num'})

for col in ['type', 'item_id']:
    for m in ['count', 'nunique']:
        tmp = evolve.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')
for col in ['lv_consume_item_num', 'n_level_up']:
    for m in ['sum', 'mean']:
        tmp = evolve.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

# 副本表
# TODO: 这个表信息比较多, 可以多挖掘

fb = pd.read_csv('datasets/role_fb_op.csv')
fb['dt'] = pd.to_datetime(fb['dt'])
fb['day'] = fb['dt'].dt.day
fb['fb_used_time'] = fb['finish_time'] - fb['start_time']

for col in ['fb_id', 'fb_type']:
    for m in ['count', 'nunique']:
        tmp = fb.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')
for col in ['fb_used_time', 'exp']:
    for m in ['sum', 'mean']:
        tmp = fb.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

tmp = fb.groupby(['role_id', 'day'])['fb_result'].value_counts().reset_index(name='fb_result_count')
for i in [0, 1, 2]:
    tt = tmp[tmp['fb_result'] == i]
    tt.columns = list(tt.columns[:-1]) + ['fb_result%d_count' % i]
    data = data.merge(tt[['role_id', 'day', 'fb_result%d_count' % i]], on=['role_id', 'day'], how='left')


# 任务系统表

mission = pd.read_csv('datasets/role_mission_op.csv')
mission['dt'] = pd.to_datetime(mission['dt'])
mission['day'] = mission['dt'].dt.day

for col in ['mission_id', 'mission_type']:
    for m in ['count', 'nunique']:
        tmp = mission.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

# 玩家离线表
# TODO: 可以做很多时间、坐标方面的特征

offline = pd.read_csv('datasets/role_offline_op.csv')
offline['dt'] = pd.to_datetime(mission['dt'])
offline['day'] = offline['dt'].dt.day
offline['online_durations'] = offline['offline'] - offline['online']

for col in ['reason', 'map_id']:
    for m in ['count', 'nunique']:
        tmp = offline.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

for col in ['online_durations']:
    for m in ['mean', 'sum']:
        tmp = offline.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')


# 付费表

pay = pd.read_csv('datasets/role_pay.csv')
pay['dt'] = pd.to_datetime(pay['dt'])
pay['day'] = pay['dt'].dt.day
tmp = pay.groupby(['role_id', 'day'])['pay'].agg('sum').to_frame(name='pay_sum_day').reset_index()
data = data.merge(tmp, on=['role_id', 'day'], how='left')
data['pay_sum_day'].fillna(0., inplace=True)

# 验证集设置
# 现在我们可以把问题转为用前n天的行为来预测第n+1天的付费

# 训练集 day 2,3,4,5,6 -> 标签 day 7 pay_sum
df_train = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(2, 7)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    df_train = df_train.merge(tmp, on='role_id')
    # 假设df_train是你的训练集数据
    df_train.to_csv('train.csv', index=False)

# 验证集 day 3,4,5,6,7 -> 标签 day 8 pay_sum
df_valid = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(3, 8)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    df_valid = df_valid.merge(tmp, on='role_id')
    # 假设df_train是你的训练集数据
    df_valid.to_csv('valid.csv', index=False)

# 测试集 day 4,5,6,7,8
df_test = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(4, 9)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    df_test = df_test.merge(tmp, on='role_id')
    # 假设df_train是你的训练集数据
    df_test.to_csv('test.csv', index=False)

# 标签构造

# 训练集 day == 7 pay_sum
# 验证集 day == 8 pay_sum

day7_pay = pay[pay.day == 7].copy().reset_index(drop=True)
tmp = day7_pay.groupby('role_id')['pay'].agg('sum').to_frame(name='pay').reset_index()
df_train = df_train.merge(tmp, on='role_id', how='left')
df_train['pay'].fillna(0., inplace=True)

day8_pay = pay[pay.day == 8].copy().reset_index(drop=True)
tmp = day8_pay.groupby('role_id')['pay'].agg('sum').to_frame(name='pay').reset_index()
df_valid = df_valid.merge(tmp, on='role_id', how='left')
df_valid['pay'].fillna(0., inplace=True)


df = pd.concat([df_train, df_valid, df_test]).reset_index(drop=True)

df['pay_log'] = np.log1p(df['pay'])


df_train = df[:len(df_train)].reset_index(drop=True)
df_valid = df[len(df_train):len(df_train)+len(df_valid)].reset_index(drop=True)
df_test = df[-len(df_test):].reset_index(drop=True)


# params = {
#     'objective': 'regression',
#     'metric': {'rmse'},
#     'boosting_type' : 'gbdt',
#     'learning_rate': 0.05,
#     'max_depth' : 7,
#     'num_leaves' : 31,
#     'feature_fraction' : 0.8,
#     'subsample' : 0.8,
#     'seed' : 114,
#     'num_iterations' : 3000,
#     'nthread' : -1,
#     'verbose' : -1
# }
params = {
    'objective': 'regression',
    'metric': {'rmse'},
    'boosting_type' : 'gbdt',
    'learning_rate': 0.05,
    'max_depth' : 8,
    'num_leaves' : 40,
    'feature_fraction' : 0.8,
    'subsample' : 0.8,
    'min_child_samples': 25,
    'seed' : 114,
    'num_iterations' : 3000,
    'nthread' : -1,
    'verbose' : -1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}
features = [col for col in df_train.columns if col not in ['role_id', 'pay', 'pay_log']]
len(features)


def train(df_train, df_valid, label, params, features):
    '''
    训练函数
    '''
    train_label = df_train[label].values
    train_feat = df_train[features]

    valid_label = df_valid[label].values
    valid_feat = df_valid[features]
    gc.collect()

    trn_data = lgb.Dataset(train_feat, label=train_label)
    val_data = lgb.Dataset(valid_feat, label=valid_label)
    params = {
        'verbose_eval': 50,  # 设置为1或其他数值来控制输出详细程度
        'early_stopping_rounds': 100
        # 其他参数...
    }
    clf = lgb.train(params,
                    trn_data,
                    valid_sets=[trn_data, val_data])

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    fold_importance_df = fold_importance_df.sort_values(by='importance', ascending=False)
    print(fold_importance_df[:30])
    #     fold_importance_df.to_csv(f"importance_df.csv", index=None)
    df_valid['{}_preds'.format(label)] = clf.predict(valid_feat, num_iteration=clf.best_iteration)
    # 负值修正
    df_valid['{}_preds'.format(label)] = df_valid['{}_preds'.format(label)].clip(lower=0.)

    result = mean_squared_log_error(np.expm1(df_valid[label]),
                                    np.expm1(df_valid['{}_preds'.format(label)]))

    #     plot(df_valid[label])
    #     plot(df_valid['{}_preds'.format(label)])
    #     show()
    plot(np.expm1(df_valid[label]))
    plot(np.expm1(df_valid['{}_preds'.format(label)]))
    show()

    return clf, result


clf_valid, result_valid = train(df_train, df_valid, 'pay_log', params, features)

print('########################rmsle score', np.sqrt(result_valid))


# 用 4,5,6,7,8 重新训练模型

params['num_iterations'] = clf_valid.best_iteration
clf_test, _ = train(df_valid, df_valid, 'pay_log', params, features)


df_test['pay'] = np.expm1(clf_test.predict(df_test[features]))
df_test['pay'] = df_test['pay'].clip(lower=0.)
df_test['pay'].describe()


sub = pd.read_csv('result/submission_sample.csv')
sub_df = df_test[['role_id', 'pay']].copy()
sub = sub[['role_id']].merge(sub_df, on='role_id', how='left')
sub[['role_id', 'pay']].to_csv('submission.csv', index=False)