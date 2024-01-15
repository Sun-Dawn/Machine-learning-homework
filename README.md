# README
1.
利用训练集和验证集数据训练LightGBM模型使用的参数为:
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

2.
之后利用上述训练过程中的最佳轮次对应的参数，只使用验证集数据进行模型重训练。

4.
利用训练好的模型对用户第8天的充值金额进行预测。
