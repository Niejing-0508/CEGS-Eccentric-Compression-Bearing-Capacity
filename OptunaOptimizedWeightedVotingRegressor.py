# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from catboost import CatBoostRegressor
# import optuna
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
# plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', "Microsoft YaHei", "SimHei"]
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 600
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['figure.edgecolor'] = 'white'
# class OptunaOptimizedWeightedVotingRegressor:
#     def __init__(self, n_trials=50, manual_weights=None):
#         self.models = {}
#         self.weights = {}
#         self.manual_weights = manual_weights
#         self.svr_scaler = StandardScaler()
#         self.target_scaler = StandardScaler()
#         self.is_fitted = False
#         self.best_params_ = {}
#         self.n_trials = n_trials
#         self.X_train_fit_ = None
#         self.y_train_fit_ = None
#     def _optimize_extra_trees(self, X_train, y_train):
#         print("\n使用Optuna优化ExtraTrees超参数...")
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#                 'max_depth': trial.suggest_int('max_depth', 10, 50),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
#                 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
#                 'random_state': 1443,
#                 'n_jobs': -1
#             }
#             model = ExtraTreesRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=self.n_trials)
#         print(f"ExtraTrees最佳参数: {study.best_params}")
#         print(f"ExtraTrees最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = 1443
#         best_params['n_jobs'] = -1
#         return ExtraTreesRegressor(**best_params)
#     def _optimize_gbdt(self, X_train, y_train):
#         print("\n使用Optuna优化GBDT超参数...")
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 800),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
#                 'max_depth': trial.suggest_int('max_depth', 3, 10),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
#                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#                 'random_state': 1443
#             }
#             model = GradientBoostingRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=self.n_trials)
#         print(f"GBDT最佳参数: {study.best_params}")
#         print(f"GBDT最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = 1443
#         return GradientBoostingRegressor(**best_params)
#     def _optimize_svr(self, X_train, y_train):
#         print("\n使用Optuna优化SVR超参数...")
#         y_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
#         X_scaled = self.svr_scaler.fit_transform(X_train)
#         def objective(trial):
#             params = {
#                 'C': trial.suggest_float('C', 0.1, 1000, log=True),
#                 'gamma': trial.suggest_float('gamma', 0.001, 1, log=True),
#                 'epsilon': trial.suggest_float('epsilon', 0.001, 0.5),
#                 'kernel': 'rbf',
#                 'cache_size': 1000
#             }
#             model = SVR(**params)
#             scores = cross_val_score(model, X_scaled, y_scaled, cv=10, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=self.n_trials)
#         print(f"SVR最佳参数: {study.best_params}")
#         print(f"SVR最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['kernel'] = 'rbf'
#         best_params['cache_size'] = 1000
#         return SVR(**best_params)
#     def _optimize_catboost(self, X_train, y_train):
#         print("\n使用Optuna优化CatBoost超参数...")
#         def objective(trial):
#             params = {
#                 'iterations': trial.suggest_int('iterations', 500, 2000),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'depth': trial.suggest_int('depth', 4, 10),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
#                 'random_strength': trial.suggest_float('random_strength', 0.1, 2),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
#                 'loss_function': 'RMSE',
#                 'eval_metric': 'RMSE',
#                 'random_state': 1443,
#                 'verbose': False,
#                 'thread_count': -1
#             }
#             model = CatBoostRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=self.n_trials)
#         print(f"CatBoost最佳参数: {study.best_params}")
#         print(f"CatBoost最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params.update({
#             'loss_function': 'RMSE',
#             'eval_metric': 'RMSE',
#             'random_state': 1443,
#             'verbose': False,
#             'thread_count': -1
#         })
#         return CatBoostRegressor(**best_params)
#     def _initialize_optimized_models(self, X_train, y_train):
#         print("开始Optuna超参数优化...")
#         optimized_et = self._optimize_extra_trees(X_train, y_train)
#         optimized_gbdt = self._optimize_gbdt(X_train, y_train)
#         optimized_svr = self._optimize_svr(X_train, y_train)
#         optimized_catboost = self._optimize_catboost(X_train, y_train)
#         self.models = {
#             'ExtraTrees': optimized_et,
#             'GBDT': optimized_gbdt,
#             'SVR': optimized_svr,
#             'CatBoost': optimized_catboost
#         }
#         self.best_params_ = {
#             'ExtraTrees': optimized_et.get_params(),
#             'GBDT': optimized_gbdt.get_params(),
#             'SVR': optimized_svr.get_params(),
#             'CatBoost': optimized_catboost.get_params()
#         }
#     def _calculate_model_weights(self, X_val, y_val):
#         if self.manual_weights is not None:
#             required_models = ['ExtraTrees', 'GBDT', 'SVR', 'CatBoost']
#             if not all(model in self.manual_weights for model in required_models):
#                 raise ValueError(f"手动权重字典必须包含以下所有模型: {required_models}")
#             if not np.isclose(sum(self.manual_weights.values()), 1.0, atol=1e-3):
#                 raise ValueError(f"手动权重的总和必须为 1.0")
#             self.weights = self.manual_weights
#             print("\n使用手动固定的模型权重:")
#             for name, weight in self.weights.items():
#                 print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#             return
#         print("\n未检测到手动权重，将基于验证集性能自动计算权重...")
#         performance_scores = {}
#         individual_metrics = {}
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_val_scaled = self.svr_scaler.transform(X_val)
#                 y_pred_scaled = model.predict(X_val_scaled)
#                 y_pred_original = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 y_pred_original = model.predict(X_val)
#             r2 = r2_score(y_val, y_pred_original)
#             rmse = np.sqrt(mean_squared_error(y_val, y_pred_original))
#             mae = mean_absolute_error(y_val, y_pred_original)
#             mape = mean_absolute_percentage_error(y_val, y_pred_original)
#             individual_metrics[name] = {
#                 'R2': r2,
#                 'RMSE': rmse,
#                 'MAE': mae,
#                 'MAPE': mape
#             }
#             if r2 < 0:
#                 print(f"警告: {name} 的R²为负值 ({r2:.4f})，将使用保守权重")
#                 performance_scores[name] = 0.01
#             else:
#                 performance_scores[name] = max(r2, 0.05)
#         print("\n各个模型在验证集上的表现:")
#         for name, metrics in individual_metrics.items():
#             print(f"  {name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
#         total_score = sum(performance_scores.values())
#         self.weights = {name: score / total_score for name, score in performance_scores.items()}
#         print("\n基于验证集性能自动计算的最终模型权重分配:")
#         for name, weight in self.weights.items():
#             print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#     def fit(self, X, y, val_ratio=0.3):
#         print("开始训练Optuna优化加权的投票集成模型...")
#         self.X_train_fit_ = X
#         self.y_train_fit_ = y
#         X_train_main, X_val, y_train_main, y_val = train_test_split(
#             X, y, test_size=val_ratio, random_state=42
#         )
#         self._initialize_optimized_models(X_train_main, y_train_main)
#         print("\n在完整训练集上重新训练优化后的模型...")
#         for name, model in self.models.items():
#             print(f"重新训练 {name}...")
#             if name == 'SVR':
#                 y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
#                 X_scaled = self.svr_scaler.fit_transform(X)
#                 model.fit(X_scaled, y_scaled)
#             else:
#                 model.fit(X, y)
#         self._calculate_model_weights(X_val, y_val)
#         self.is_fitted = True
#         print("\nOptuna优化集成模型训练完成！")
#     def predict(self, X):
#         if not self.is_fitted:
#             raise ValueError("模型尚未训练，请先调用fit方法")
#         predictions = np.zeros(len(X))
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.svr_scaler.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 pred = model.predict(X)
#             predictions += self.weights[name] * pred
#         return predictions
#     def predict_individual(self, X):
#         individual_predictions = {}
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.svr_scaler.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 individual_predictions[name] = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 individual_predictions[name] = model.predict(X)
#         return individual_predictions
#     def evaluate(self, X, y):
#         predictions = self.predict(X)
#         metrics = {
#             'MSE': mean_squared_error(y, predictions),
#             'RMSE': np.sqrt(mean_squared_error(y, predictions)),
#             'MAE': mean_absolute_error(y, predictions),
#             'R2': r2_score(y, predictions),
#             'MAPE': mean_absolute_percentage_error(y, predictions)
#         }
#         return metrics, predictions
#     def calculate_cov_and_mean(self, X, y):
#         y_pred = self.predict(X)
#         mean_actual = np.mean(y)
#         mean_pred = np.mean(y_pred)
#         std_actual = np.std(y, ddof=1)
#         std_pred = np.std(y_pred, ddof=1)
#         absolute_errors = np.abs(y - y_pred)
#         mean_absolute_error_val = np.mean(absolute_errors)
#         std_absolute_error = np.std(absolute_errors, ddof=1)
#         ratio = y_pred / y
#         mean_ratio = np.mean(ratio)
#         std_ratio = np.std(ratio, ddof=1)
#         cov_actual = (std_actual / mean_actual) * 100 if mean_actual != 0 else 0
#         cov_pred = (std_pred / mean_pred) * 100 if mean_pred != 0 else 0
#         cov_absolute_error = (std_absolute_error / mean_absolute_error_val) * 100 if mean_absolute_error_val != 0 else 0
#         cov_ratio = (std_ratio / mean_ratio) * 100 if mean_ratio != 0 else 0
#         return {
#             'MEAN_Actual': mean_actual,
#             'MEAN_Predicted': mean_pred,
#             'MEAN_Absolute_Error': mean_absolute_error_val,
#             'MEAN_Ratio': mean_ratio,
#             'STD_Actual': std_actual,
#             'STD_Predicted': std_pred,
#             'STD_Absolute_Error': std_absolute_error,
#             'STD_Ratio': std_ratio,
#             'COV_Actual(%)': cov_actual,
#             'COV_Predicted(%)': cov_pred,
#             'COV_Absolute_Error(%)': cov_absolute_error,
#             'COV_Ratio(%)': cov_ratio
#         }
#     def plot_comparison(self, X_test, y_test):
#         individual_preds = self.predict_individual(X_test)
#         ensemble_pred = self.predict(X_test)
#         model_r2 = {name: r2_score(y_test, pred) for name, pred in individual_preds.items()}
#         ensemble_r2 = r2_score(y_test, ensemble_pred)
#         plt.figure(figsize=(12, 7))
#         models_list = list(model_r2.keys()) + ['Optuna Ensemble']
#         r2_scores = list(model_r2.values()) + [ensemble_r2]
#         colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#FF6B6B']
#         bars = plt.bar(models_list, r2_scores, color=colors, alpha=1.0)
#         plt.ylabel('R² Score', fontsize=16, fontweight='bold')
#         plt.title('Performance Comparison Between Base Models and Ensemble Model', fontsize=18, fontweight='bold')
#         plt.xticks(rotation=45, fontsize=14, fontweight='bold')
#         plt.yticks(fontsize=14, fontweight='bold')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=6)
#         ax.tick_params(axis='both', which='minor', width=1.5, length=3)
#         plt.grid(True, alpha=0.3, axis='y')
#         for bar, score in zip(bars, r2_scores):
#             plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{score:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
#         plt.axhline(y=ensemble_r2, color='r', linestyle='--', alpha=0.7,
#                     label=f'Ensemble R² ({ensemble_r2:.4f})')
#         plt.legend(fontsize=14, prop={'weight': 'bold', 'family': 'Times New Roman'})
#         plt.tight_layout()
#         plt.savefig('CEGS模型性能对比.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         return model_r2, ensemble_r2
#     def plot_predictions_vs_actual(self, X_train, X_test, y_train, y_test):
#         y_train_pred = self.predict(X_train)
#         y_test_pred = self.predict(X_test)
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
#         plt.figure(figsize=(10, 8))
#         plt.scatter(y_train, y_train_pred, color='#64B5CD', s=100, marker='o', alpha=1,
#                     label=f'Train List')
#         plt.scatter(y_test, y_test_pred, color='#D4BE83', s=100, marker='s', alpha=1,
#                     label=f'Test List')
#         plt.xlabel('True ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         plt.ylabel('Predicted ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=8)
#         ax.tick_params(axis='both', which='minor', width=2.0, length=4)
#         text_box = f"R²: {test_r2:.4f}\nMAE: {test_mae:.4f} kN\nRMSE: {test_rmse:.4f} kN\nMAPE: {test_mape:.2%}"
#         plt.text(70, 2400, text_box, fontsize=20, fontfamily='Times New Roman',
#                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#         plt.plot([0, 3500], [0, 3500], "--k")
#         plt.plot([0, 3500], [0 * 1.15, 3500 * 1.15], "--b", alpha=0.7, label='±15% Error Bound')
#         plt.plot([0, 3500], [0 * 0.85, 3500 * 0.85], "--b", alpha=0.7)
#         plt.xlim(0, 3500)
#         plt.ylim(0, 3500)
#         plt.legend(fontsize=20, loc='upper left', prop={'family': 'Times New Roman'})
#         plt.xticks(fontsize=20, fontfamily='Times New Roman')
#         plt.yticks(fontsize=20, fontfamily='Times New Roman')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('CEGS Regression.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
#         plt.close()
#         print(f"\n训练集指标:")
#         print(f"R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
#         print(f"\n测试集指标:")
#         print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")
#         train_cov_metrics = self.calculate_cov_and_mean(X_train, y_train)
#         test_cov_metrics = self.calculate_cov_and_mean(X_test, y_test)
#         print(f"\n训练集COV和MEAN统计:")
#         for key, value in train_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         print(f"\n测试集COV和MEAN统计:")
#         for key, value in test_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         return y_train_pred, y_test_pred, train_cov_metrics, test_cov_metrics


# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from catboost import CatBoostRegressor
# import optuna
# import matplotlib.pyplot as plt
# import warnings
# from hyperopt import hp, fmin, tpe, Trials
# from hyperopt.early_stop import no_progress_loss
# warnings.filterwarnings('ignore')
# plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', "Microsoft YaHei", "SimHei"]
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 600
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['figure.edgecolor'] = 'white'
# GLOBAL_SEED = 1443
# class OptunaOptimizedWeightedVotingRegressor:
#     def __init__(self, n_trials=50, manual_weights=None):
#         self.models = {}
#         self.weights = {}
#         self.manual_weights = manual_weights
#         self.train_scaler_x = StandardScaler()
#         self.train_scaler_y = StandardScaler()
#         self.is_fitted = False
#         self.best_params_ = {}
#         self.n_trials = n_trials
#         self.X_train_full_ = None
#         self.y_train_full_ = None
#         self.cv_fold_num = 10
#         self.eps = 1e-8
#     def _optimize_extra_trees(self, X_train, y_train):
#         print("\n使用Optuna优化ExtraTrees超参数...")
#         cv_kfold = KFold(n_splits=self.cv_fold_num, shuffle=True, random_state=GLOBAL_SEED)
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#                 'max_depth': trial.suggest_int('max_depth', 10, 50),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
#                 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
#                 'random_state': GLOBAL_SEED,
#                 'n_jobs': -1
#             }
#             model = ExtraTreesRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"ExtraTrees最佳参数: {study.best_params}")
#         print(f"ExtraTrees最佳交叉验证R²: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_SEED
#         best_params['n_jobs'] = -1
#         return ExtraTreesRegressor(**best_params)
#     def _optimize_gbdt(self, X_train, y_train):
#         print("\n使用Hyperopt TPE贝叶斯优化GBDT超参数...")
#         param_space = {
#             'n_estimators': hp.quniform('n_estimators', 100, 800, 10),
#             'learning_rate': hp.quniform('learning_rate', 0.01, 0.2, 0.002),
#             'criterion': hp.choice('criterion', ["squared_error"]),
#             'loss': hp.choice('loss', ["squared_error"]),
#             'max_depth': hp.quniform('max_depth', 3, 10, 1),
#             'subsample': hp.quniform('subsample', 0.5, 1.0, 0.02),
#             'max_features': hp.choice('max_features', ["sqrt", "log2", None, 13]),
#             'min_impurity_decrease': hp.quniform("min_impurity_decrease", 0.0, 4.0, 0.2)
#         }
#         cv_kfold = KFold(n_splits=self.cv_fold_num, shuffle=True, random_state=GLOBAL_SEED)
#         def gbdt_objective(params):
#             params['n_estimators'] = int(params['n_estimators'])
#             params['max_depth'] = int(params['max_depth'])
#             gbr = GradientBoostingRegressor(
#                 n_estimators=params['n_estimators'],
#                 learning_rate=params['learning_rate'],
#                 criterion=params['criterion'],
#                 loss=params['loss'],
#                 max_depth=params['max_depth'],
#                 max_features=params['max_features'],
#                 subsample=params['subsample'],
#                 min_impurity_decrease=params['min_impurity_decrease'],
#                 verbose=False,
#                 random_state=GLOBAL_SEED
#             )
#             cv_r2 = cross_val_score(gbr, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1).mean()
#             return -1 * cv_r2
#         trials = Trials()
#         early_stop = no_progress_loss(100)
#         best_raw_params = fmin(
#             fn=gbdt_objective,
#             space=param_space,
#             algo=tpe.suggest,
#             max_evals=self.n_trials,
#             verbose=False,
#             trials=trials,
#             early_stop_fn=early_stop
#         )
#         best_params = {}
#         best_params['criterion'] = "squared_error"
#         best_params['loss'] = "squared_error"
#         best_params['max_features'] = best_raw_params['max_features']
#         best_params['n_estimators'] = int(best_raw_params['n_estimators'])
#         best_params['max_depth'] = int(best_raw_params['max_depth'])
#         best_params['learning_rate'] = best_raw_params['learning_rate']
#         best_params['subsample'] = best_raw_params['subsample']
#         best_params['min_impurity_decrease'] = best_raw_params['min_impurity_decrease']
#         best_params['random_state'] = GLOBAL_SEED
#         best_gbdt = GradientBoostingRegressor(**best_params)
#         cv_r2 = cross_val_score(best_gbdt, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1).mean()
#         print(f"GBDT最佳参数: {best_params}")
#         print(f"GBDT最佳交叉验证R²: {cv_r2:.4f}")
#         return GradientBoostingRegressor(**best_params)
#     def _optimize_svr(self, X_train, y_train, scaler_x, scaler_y):
#         print("\n使用Optuna优化SVR超参数...")
#         y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
#         X_scaled = scaler_x.fit_transform(X_train)
#         cv_kfold = KFold(n_splits=self.cv_fold_num, shuffle=True, random_state=GLOBAL_SEED)
#         def objective(trial):
#             params = {
#                 'C': trial.suggest_float('C', 0.1, 1000, log=True),
#                 'gamma': trial.suggest_float('gamma', 0.001, 1, log=True),
#                 'epsilon': trial.suggest_float('epsilon', 0.001, 0.5),
#                 'kernel': 'rbf',
#                 'cache_size': 1000
#             }
#             model = SVR(**params)
#             scores = cross_val_score(model, X_scaled, y_scaled, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"SVR最佳参数: {study.best_params}")
#         print(f"SVR最佳交叉验证R²: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['kernel'] = 'rbf'
#         best_params['cache_size'] = 1000
#         return SVR(**best_params)
#     def _optimize_catboost(self, X_train, y_train):
#         print("\n使用Optuna优化CatBoost超参数...")
#         cv_kfold = KFold(n_splits=self.cv_fold_num, shuffle=True, random_state=GLOBAL_SEED)
#         def objective(trial):
#             params = {
#                 'iterations': trial.suggest_int('iterations', 500, 2000),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'depth': trial.suggest_int('depth', 4, 10),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
#                 'random_strength': trial.suggest_float('random_strength', 0.1, 2),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
#                 'loss_function': 'RMSE',
#                 'eval_metric': 'RMSE',
#                 'random_state': GLOBAL_SEED,
#                 'verbose': False,
#                 'thread_count': -1
#             }
#             model = CatBoostRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"CatBoost最佳参数: {study.best_params}")
#         print(f"CatBoost最佳交叉验证R²: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params.update({
#             'loss_function': 'RMSE',
#             'eval_metric': 'RMSE',
#             'random_state': GLOBAL_SEED,
#             'verbose': False,
#             'thread_count': -1
#         })
#         return CatBoostRegressor(**best_params)
#     def _initialize_optimized_models(self, X_train_main, y_train_main):
#         print("开始各模型贝叶斯超参数优化...")
#         temp_scaler_x = StandardScaler()
#         temp_scaler_y = StandardScaler()
#         et = self._optimize_extra_trees(X_train_main, y_train_main)
#         gbdt = self._optimize_gbdt(X_train_main, y_train_main)
#         svr = self._optimize_svr(X_train_main, y_train_main, temp_scaler_x, temp_scaler_y)
#         cb = self._optimize_catboost(X_train_main, y_train_main)
#         self.models = {
#             'ExtraTrees': et,
#             'GBDT': gbdt,
#             'SVR': svr,
#             'CatBoost': cb
#         }
#         self.best_params_ = {
#             'ExtraTrees': et.get_params(),
#             'GBDT': gbdt.get_params(),
#             'SVR': svr.get_params(),
#             'CatBoost': cb.get_params()
#         }
#     def _get_cv_avg_r2(self, model, X, y, is_svr=False):
#         kf = KFold(n_splits=self.cv_fold_num, shuffle=True, random_state=GLOBAL_SEED)
#         r2_list = []
#         for tr_idx, val_idx in kf.split(X):
#             X_tr, X_val = X[tr_idx], X[val_idx]
#             y_tr, y_val = y[tr_idx], y[val_idx]
#             if is_svr:
#                 sc_x = StandardScaler()
#                 sc_y = StandardScaler()
#                 Xtr_sc = sc_x.fit_transform(X_tr)
#                 ytr_sc = sc_y.fit_transform(y_tr.reshape(-1,1)).ravel()
#                 model.fit(Xtr_sc, ytr_sc)
#                 Xvl_sc = sc_x.transform(X_val)
#                 pred_sc = model.predict(Xvl_sc)
#                 pred = sc_y.inverse_transform(pred_sc.reshape(-1,1)).ravel()
#             else:
#                 model.fit(X_tr, y_tr)
#                 pred = model.predict(X_val)
#             r2 = r2_score(y_val, pred)
#             r2_list.append(max(r2, self.eps))
#         return np.mean(r2_list)
#     def _calculate_model_weights(self, X_train_main, y_train_main):
#         if self.manual_weights is not None:
#             required = ['ExtraTrees', 'GBDT', 'SVR', 'CatBoost']
#             if not all(m in self.manual_weights for m in required):
#                 raise ValueError(f"手动权重必须包含全部模型: {required}")
#             if not np.isclose(sum(self.manual_weights.values()), 1.0, atol=1e-3):
#                 raise ValueError("手动权重总和必须等于1.0")
#             self.weights = self.manual_weights
#             print("\n使用手动归一化权重:")
#             for name, w in self.weights.items():
#                 print(f"  {name}: {w:.4f} ({w*100:.1f}%)")
#             return
#         print("\n基于训练集10折交叉验证平均R²计算稳定复合权重...")
#         score_dict = {}
#         metric_records = {}
#         for name, model in self.models.items():
#             is_svr_flag = (name == "SVR")
#             avg_cv_r2 = self._get_cv_avg_r2(model, X_train_main, y_train_main, is_svr=is_svr_flag)
#             if is_svr_flag:
#                 X_tr_sc = self.train_scaler_x.transform(X_train_main)
#                 pred_sc = model.predict(X_tr_sc)
#                 pred = self.train_scaler_y.inverse_transform(pred_sc.reshape(-1,1)).ravel()
#             else:
#                 pred = model.predict(X_train_main)
#             r2_single = r2_score(y_train_main, pred)
#             rmse_single = np.sqrt(mean_squared_error(y_train_main, pred))
#             metric_records[name] = {
#                 "R2": r2_single,
#                 "RMSE": rmse_single
#             }
#             composite_score = avg_cv_r2 / (rmse_single + self.eps)
#             if avg_cv_r2 < 0.7:
#                 composite_score *= 0.01
#             score_dict[name] = composite_score
#         print("\n各模型训练集拟合指标:")
#         for name, met in metric_records.items():
#             print(f"  {name}: R²={met['R2']:.4f}, RMSE={met['RMSE']:.2f}")
#         total = sum(score_dict.values())
#         self.weights = {k: v / total for k, v in score_dict.items()}
#         print("\n交叉验证复合性能权重分配:")
#         for name, w in self.weights.items():
#             print(f"  {name}: {w:.4f} ({w*100:.1f}%)")
#     def fit(self, X, y, val_ratio=0.3):
#         print("开始训练加权集成模型...")
#         self.X_train_full_ = X
#         self.y_train_full_ = y
#         X_train_main, X_val, y_train_main, y_val = train_test_split(
#             X, y, test_size=val_ratio, random_state=42, shuffle=True
#         )
#         self._initialize_optimized_models(X_train_main, y_train_main)
#         self.train_scaler_x.fit(X_train_main)
#         self.train_scaler_y.fit(y_train_main.reshape(-1, 1))
#         print("\n全量数据重训各基模型...")
#         X_scaled_all = self.train_scaler_x.transform(X)
#         y_scaled_all = self.train_scaler_y.transform(y.reshape(-1, 1)).ravel()
#         for name, model in self.models.items():
#             print(f"重训 {name}")
#             if name == "SVR":
#                 model.fit(X_scaled_all, y_scaled_all)
#             else:
#                 model.fit(X, y)
#         self._calculate_model_weights(X_train_main, y_train_main)
#         self.is_fitted = True
#         print("\n集成模型训练完成！")
#     def predict(self, X):
#         if not self.is_fitted:
#             raise ValueError("模型未训练，请先执行fit()")
#         total_pred = np.zeros(len(X))
#         for name, model in self.models.items():
#             if name == "SVR":
#                 X_sc = self.train_scaler_x.transform(X)
#                 pred_sc = model.predict(X_sc)
#                 pred = self.train_scaler_y.inverse_transform(pred_sc.reshape(-1,1)).ravel()
#             else:
#                 pred = model.predict(X)
#             total_pred += self.weights[name] * pred
#         return total_pred
#     def predict_individual(self, X):
#         single_preds = {}
#         for name, model in self.models.items():
#             if name == "SVR":
#                 X_sc = self.train_scaler_x.transform(X)
#                 pred_sc = model.predict(X_sc)
#                 single_preds[name] = self.train_scaler_y.inverse_transform(pred_sc.reshape(-1,1)).ravel()
#             else:
#                 single_preds[name] = model.predict(X)
#         return single_preds
#     def evaluate(self, X, y):
#         pred = self.predict(X)
#         y_safe = np.where(np.abs(y) < self.eps, self.eps, y)
#         metrics = {
#             'MSE': mean_squared_error(y, pred),
#             'RMSE': np.sqrt(mean_squared_error(y, pred)),
#             'MAE': mean_absolute_error(y, pred),
#             'R2': r2_score(y, pred),
#             'MAPE': mean_absolute_percentage_error(y_safe, pred)
#         }
#         return metrics, pred
#     def calculate_cov_and_mean(self, X, y):
#         y_pred = self.predict(X)
#         mean_actual = np.mean(y)
#         mean_pred = np.mean(y_pred)
#         std_actual = np.std(y, ddof=1)
#         std_pred = np.std(y_pred, ddof=1)
#         abs_err = np.abs(y - y_pred)
#         mean_abs_err = np.mean(abs_err)
#         std_abs_err = np.std(abs_err, ddof=1)
#         ratio = y_pred / (y + self.eps)
#         mean_ratio = np.mean(ratio)
#         std_ratio = np.std(ratio, ddof=1)
#         cov_actual = (std_actual / (mean_actual + self.eps)) * 100
#         cov_pred = (std_pred / (mean_pred + self.eps)) * 100
#         cov_abs_err = (std_abs_err / (mean_abs_err + self.eps)) * 100
#         cov_ratio = (std_ratio / (mean_ratio + self.eps)) * 100
#         return {
#             'MEAN_Actual': mean_actual,
#             'MEAN_Predicted': mean_pred,
#             'MEAN_Absolute_Error': mean_abs_err,
#             'MEAN_Ratio': mean_ratio,
#             'STD_Actual': std_actual,
#             'STD_Predicted': std_pred,
#             'STD_Absolute_Error': std_abs_err,
#             'STD_Ratio': std_ratio,
#             'COV_Actual(%)': cov_actual,
#             'COV_Predicted(%)': cov_pred,
#             'COV_Absolute_Error(%)': cov_abs_err,
#             'COV_Ratio(%)': cov_ratio
#         }
#     def plot_comparison(self, X_test, y_test):
#         ind_pred = self.predict_individual(X_test)
#         ens_pred = self.predict(X_test)
#         model_r2 = {n: r2_score(y_test, p) for n, p in ind_pred.items()}
#         ens_r2 = r2_score(y_test, ens_pred)
#         plt.figure(figsize=(12, 7))
#         names = list(model_r2.keys()) + ['Optuna Ensemble']
#         r2_vals = list(model_r2.values()) + [ens_r2]
#         colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#FF6B6B']
#         bars = plt.bar(names, r2_vals, color=colors, alpha=1.0)
#         plt.ylabel('R² Score', fontsize=16, fontweight='bold')
#         plt.title('Performance Comparison Between Base Models and Ensemble Model', fontsize=18, fontweight='bold')
#         plt.xticks(rotation=45, fontsize=14, fontweight='bold')
#         plt.yticks(fontsize=14, fontweight='bold')
#         ax = plt.gca()
#         for spine in ax.spines.values():
#             spine.set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=6)
#         ax.tick_params(axis='both', which='minor', width=1.5, length=3)
#         plt.grid(True, alpha=0.3, axis='y')
#         for bar, score in zip(bars, r2_vals):
#             plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{score:.4f}",
#                      ha='center', va='bottom', fontsize=13, fontweight='bold')
#         plt.axhline(y=ens_r2, color='r', linestyle='--', alpha=0.7, label=f'Ensemble R² ({ens_r2:.4f})')
#         plt.legend(fontsize=14, prop={'weight':'bold', 'family':'Times New Roman'})
#         plt.tight_layout()
#         plt.savefig('CEGS模型性能对比.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         return model_r2, ens_r2
#     def plot_predictions_vs_actual(self, X_train, X_test, y_train, y_test):
#         y_train_pred = self.predict(X_train)
#         y_test_pred = self.predict(X_test)
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_mape = mean_absolute_percentage_error(np.where(np.abs(y_train)<self.eps,self.eps,y_train), y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_mape = mean_absolute_percentage_error(np.where(np.abs(y_test)<self.eps,self.eps,y_test), y_test_pred)
#         plt.figure(figsize=(10, 8))
#         plt.scatter(y_train, y_train_pred, color='#64B5CD', s=100, marker='o', alpha=1, label=f'Train List')
#         plt.scatter(y_test, y_test_pred, color='#D4BE83', s=100, marker='s', alpha=1, label=f'Test List')
#         plt.xlabel('True ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         plt.ylabel('Predicted ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         ax = plt.gca()
#         for spine in ax.spines.values():
#             spine.set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=8)
#         ax.tick_params(axis='both', which='minor', width=2.0, length=4)
#         text_box = f"R²: {test_r2:.4f}\nMAE: {test_mae:.4f} kN\nRMSE: {test_rmse:.4f} kN\nMAPE: {test_mape:.2%}"
#         plt.text(70, 2400, text_box, fontsize=20, fontfamily='Times New Roman',
#                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#         plt.plot([0, 3500], [0, 3500], "--k")
#         plt.plot([0, 3500], [0*1.15, 3500*1.15], "--b", alpha=0.7, label='±15% Error Bound')
#         plt.plot([0, 3500], [0*0.85, 3500*0.85], "--b", alpha=0.7)
#         plt.xlim(0, 3500)
#         plt.ylim(0, 3500)
#         plt.legend(fontsize=20, loc='upper left', prop={'family':'Times New Roman'})
#         plt.xticks(fontsize=20, fontfamily='Times New Roman')
#         plt.yticks(fontsize=20, fontfamily='Times New Roman')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('CEGS Regression.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
#         plt.close()
#         print(f"\n训练集指标:")
#         print(f"R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
#         print(f"\n测试集指标:")
#         print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")
#         train_cov = self.calculate_cov_and_mean(X_train, y_train)
#         test_cov = self.calculate_cov_and_mean(X_test, y_test)
#         print(f"\n训练集COV和MEAN统计:")
#         for k, v in train_cov.items():
#             print(f"  {k}: {v:.4f}")
#         print(f"\n测试集COV和MEAN统计:")
#         for k, v in test_cov.items():
#             print(f"  {k}: {v:.4f}")
#         return y_train_pred, y_test_pred, train_cov, test_cov



# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from catboost import CatBoostRegressor
# import optuna
# from optuna.pruners import MedianPruner
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
# plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', "Microsoft YaHei", "SimHei"]
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 600
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['figure.edgecolor'] = 'white'
#
# GLOBAL_RANDOM_SEED = 1443
# class OptunaOptimizedWeightedVotingRegressor:
#     def __init__(self, n_trials=100, manual_weights=None):
#         self.models = {}
#         self.weights = {}
#         self.manual_weights = manual_weights
#         self.scaler_x_train = StandardScaler()
#         self.scaler_y_train = StandardScaler()
#         self.is_fitted = False
#         self.best_params_ = {}
#         self.n_trials = n_trials
#         self.X_train_fit_ = None
#         self.y_train_fit_ = None
#         self.eps = 1e-8
#         self.cv_folds = 10
#
#     def _optimize_extra_trees(self, X_train, y_train):
#         print("\n使用Optuna优化ExtraTrees超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 80, 200),
#                 'max_depth': trial.suggest_int('max_depth', 2, 8),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 3, 8),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 6),
#                 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
#                 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'n_jobs': -1
#             }
#             model = ExtraTreesRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"ExtraTrees最佳参数: {study.best_params}")
#         print(f"ExtraTrees最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         best_params['n_jobs'] = -1
#         return ExtraTreesRegressor(**best_params)
#
#     def _optimize_gbdt(self, X_train, y_train):
#         print("\n使用Optuna优化GBDT超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 50, 200),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'max_depth': trial.suggest_int('max_depth', 2, 6),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 7),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 6),
#                 'subsample': trial.suggest_float('subsample', 0.4, 1.0),
#                 'alpha': trial.suggest_float('alpha', 0.01, 0.1),
#                 'random_state': GLOBAL_RANDOM_SEED
#             }
#             model = GradientBoostingRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"GBDT最佳参数: {study.best_params}")
#         print(f"GBDT最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         return GradientBoostingRegressor(**best_params)
#
#     def _optimize_svr(self, X_train, y_train, scaler_x, scaler_y):
#         print("\n使用Optuna优化SVR超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         X_train_scaled = scaler_x.fit_transform(X_train)
#         y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
#         def objective(trial):
#             kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
#             base_params = {
#                 "C": trial.suggest_float("C", 0.01, 500, log=True),
#                 "epsilon": trial.suggest_float("epsilon", 0.001, 0.5),
#                 "cache_size": 1000
#             }
#             if kernel == "rbf":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_rbf", ["scale", "auto", 0.001, 0.01, 0.1, 1, 10, 100])
#             elif kernel == "poly":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_poly", ["scale", "auto", 0.001, 0.01, 0.1, 1])
#                 base_params["degree"] = trial.suggest_int("degree", 2, 5)
#             elif kernel == "sigmoid":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_sigmoid", ["scale", "auto", 0.001, 0.01, 0.1, 1])
#                 base_params["coef0"] = trial.suggest_float("coef0", -0.5, 1.5)
#             model = SVR(kernel=kernel,** base_params)
#             scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=cv_kfold, scoring="r2", n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         best_raw_params = study.best_params
#         print(f"SVR最佳参数: {best_raw_params}")
#         print(f"SVR最佳交叉验证R²分数: {study.best_value:.4f}")
#         best_params = {}
#         kernel = best_raw_params["kernel"]
#         best_params["kernel"] = kernel
#         best_params["C"] = best_raw_params["C"]
#         best_params["epsilon"] = best_raw_params["epsilon"]
#         best_params["cache_size"] = 1000
#         if kernel == "rbf":
#             best_params["gamma"] = best_raw_params["gamma_rbf"]
#         elif kernel == "poly":
#             best_params["gamma"] = best_raw_params["gamma_poly"]
#             best_params["degree"] = best_raw_params["degree"]
#         elif kernel == "sigmoid":
#             best_params["gamma"] = best_raw_params["gamma_sigmoid"]
#             best_params["coef0"] = best_raw_params["coef0"]
#         return SVR(**best_params)
#
#     def _optimize_catboost(self, X_train, y_train):
#         print("\n使用Optuna优化CatBoost超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'iterations': trial.suggest_int('iterations', 150, 1200),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'depth': trial.suggest_int('depth', 2, 6),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),
#                 'random_strength': trial.suggest_float('random_strength', 0.5, 5),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2.0),
#                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#                 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
#                 'loss_function': 'RMSE',
#                 'eval_metric': 'RMSE',
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'verbose': False,
#                 'thread_count': -1,
#                 'early_stopping_rounds': 100
#             }
#             model = CatBoostRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"CatBoost最佳参数: {study.best_params}")
#         print(f"CatBoost最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params.update({
#             'loss_function': 'RMSE',
#             'eval_metric': 'RMSE',
#             'random_state': GLOBAL_RANDOM_SEED,
#             'verbose': False,
#             'thread_count': -1,
#             'early_stopping_rounds': 100
#         })
#         return CatBoostRegressor(**best_params)
#
#     def _initialize_optimized_models(self, X_train_main, y_train_main):
#         print("开始Optuna超参数优化...")
#         temp_scaler_x = StandardScaler()
#         temp_scaler_y = StandardScaler()
#         optimized_et = self._optimize_extra_trees(X_train_main, y_train_main)
#         optimized_gbdt = self._optimize_gbdt(X_train_main, y_train_main)
#         optimized_svr = self._optimize_svr(X_train_main, y_train_main, temp_scaler_x, temp_scaler_y)
#         optimized_catboost = self._optimize_catboost(X_train_main, y_train_main)
#         self.models = {
#             'ExtraTrees': optimized_et,
#             'GBDT': optimized_gbdt,
#             'SVR': optimized_svr,
#             'CatBoost': optimized_catboost
#         }
#         self.best_params_ = {
#             'ExtraTrees': optimized_et.get_params(),
#             'GBDT': optimized_gbdt.get_params(),
#             'SVR': optimized_svr.get_params(),
#             'CatBoost': optimized_catboost.get_params()
#         }
#
#     def _get_cv_avg_r2(self, model, X, y, is_svr=False):
#         cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         r2_list = []
#         for train_idx, val_idx in cv.split(X):
#             X_cv_train, X_cv_val = X[train_idx], X[val_idx]
#             y_cv_train, y_cv_val = y[train_idx], y[val_idx]
#             if is_svr:
#                 scaler_x_cv = StandardScaler()
#                 scaler_y_cv = StandardScaler()
#                 X_cv_tr_sca = scaler_x_cv.fit_transform(X_cv_train)
#                 y_cv_tr_sca = scaler_y_cv.fit_transform(y_cv_train.reshape(-1,1)).ravel()
#                 model.fit(X_cv_tr_sca, y_cv_tr_sca)
#                 X_cv_vl_sca = scaler_x_cv.transform(X_cv_val)
#                 pred_sca = model.predict(X_cv_vl_sca)
#                 pred = scaler_y_cv.inverse_transform(pred_sca.reshape(-1,1)).ravel()
#             else:
#                 model.fit(X_cv_train, y_cv_train)
#                 pred = model.predict(X_cv_val)
#             r2 = r2_score(y_cv_val, pred)
#             r2_list.append(max(r2, self.eps))
#         return np.mean(r2_list)
#
#     def _calculate_model_weights(self, X_train_main, y_train_main):
#         required_models = ['ExtraTrees', 'GBDT', 'SVR', 'CatBoost']
#         if self.manual_weights is not None:
#             if not all(model in self.manual_weights for model in required_models):
#                 raise ValueError(f"手动权重字典必须包含以下所有模型: {required_models}")
#             weight_sum = sum(self.manual_weights.values())
#             norm_weights = {k: v / (weight_sum + self.eps) for k, v in self.manual_weights.items()}
#             self.weights = norm_weights
#             print("\n使用手动固定并归一化后的模型权重:")
#             for name, weight in self.weights.items():
#                 print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#             return
#         print("\n未检测到手动权重，基于10折交叉验证平均R²复合指标自动计算权重...")
#         performance_scores = {}
#         individual_metrics = {}
#         for name, model in self.models.items():
#             is_svr_flag = (name == "SVR")
#             avg_r2 = self._get_cv_avg_r2(model, X_train_main, y_train_main, is_svr=is_svr_flag)
#             X_train_main_scaled = self.scaler_x_train.transform(X_train_main) if is_svr_flag else X_train_main
#             if is_svr_flag:
#                 pred_scaled = model.predict(X_train_main_scaled)
#                 pred = self.scaler_y_train.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
#             else:
#                 pred = model.predict(X_train_main)
#             r2 = r2_score(y_train_main, pred)
#             rmse = np.sqrt(mean_squared_error(y_train_main, pred))
#             mae = mean_absolute_error(y_train_main, pred)
#             mape = mean_absolute_percentage_error(np.where(np.abs(y_train_main)<self.eps,self.eps,y_train_main), pred)
#             individual_metrics[name] = {
#                 'R2': r2,
#                 'RMSE': rmse,
#                 'MAE': mae,
#                 'MAPE': mape
#             }
#             composite_score = max(avg_r2, self.eps) / (rmse + self.eps)
#             performance_scores[name] = composite_score
#         print("\n各个模型在训练集交叉验证表现:")
#         for name, metrics in individual_metrics.items():
#             print(f"  {name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
#         total_score = sum(performance_scores.values())
#         self.weights = {name: score / total_score for name, score in performance_scores.items()}
#         print("\n基于交叉验证复合性能自动计算的最终模型权重分配:")
#         for name, weight in self.weights.items():
#             print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#
#     def fit(self, X, y, val_ratio=0.3):
#         print("开始训练Optuna优化加权的投票集成模型...")
#         self.X_train_fit_ = X
#         self.y_train_fit_ = y
#         X_train_main, X_val, y_train_main, y_val = train_test_split(
#             X, y, test_size=val_ratio, random_state=GLOBAL_RANDOM_SEED, shuffle=True
#         )
#         self._initialize_optimized_models(X_train_main, y_train_main)
#         print("\n使用主训练子集拟合标准化缩放器（杜绝数据泄露）...")
#         self.scaler_x_train.fit(X_train_main)
#         self.scaler_y_train.fit(y_train_main.reshape(-1, 1))
#         X_scaled_full = self.scaler_x_train.transform(X)
#         y_scaled_full = self.scaler_y_train.transform(y.reshape(-1, 1)).ravel()
#         for name, model in self.models.items():
#             print(f"全量数据集重训 {name}...")
#             if name == 'SVR':
#                 model.fit(X_scaled_full, y_scaled_full)
#             else:
#                 model.fit(X, y)
#         self._calculate_model_weights(X_train_main, y_train_main)
#         self.is_fitted = True
#         print("\nOptuna优化集成模型训练完成！")
#
#     def predict(self, X):
#         if not self.is_fitted:
#             raise ValueError("模型尚未训练，请先调用fit方法")
#         predictions = np.zeros(len(X))
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.scaler_x_train.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 pred = self.scaler_y_train.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 pred = model.predict(X)
#             predictions += self.weights[name] * pred
#         return predictions
#
#     def predict_individual(self, X):
#         individual_predictions = {}
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.scaler_x_train.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 individual_predictions[name] = self.scaler_y_train.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 individual_predictions[name] = model.predict(X)
#         return individual_predictions
#
#     def evaluate(self, X, y):
#         predictions = self.predict(X)
#         y_safe = np.where(np.abs(y) < self.eps, self.eps, y)
#         mape = mean_absolute_percentage_error(y_safe, predictions)
#         metrics = {
#             'MSE': mean_squared_error(y, predictions),
#             'RMSE': np.sqrt(mean_squared_error(y, predictions)),
#             'MAE': mean_absolute_error(y, predictions),
#             'R2': r2_score(y, predictions),
#             'MAPE': mape
#         }
#         return metrics, predictions
#
#     def calculate_cov_and_mean(self, X, y):
#         y_pred = self.predict(X)
#         mean_actual = np.mean(y)
#         mean_pred = np.mean(y_pred)
#         std_actual = np.std(y, ddof=1)
#         std_pred = np.std(y_pred, ddof=1)
#         absolute_errors = np.abs(y - y_pred)
#         mean_absolute_error_val = np.mean(absolute_errors)
#         std_absolute_error = np.std(absolute_errors, ddof=1)
#         ratio = y_pred / (y + self.eps)
#         mean_ratio = np.mean(ratio)
#         std_ratio = np.std(ratio, ddof=1)
#         cov_actual = (std_actual / (mean_actual + self.eps)) * 100
#         cov_pred = (std_pred / (mean_pred + self.eps)) * 100
#         cov_absolute_error = (std_absolute_error / (mean_absolute_error_val + self.eps)) * 100
#         cov_ratio = (std_ratio / (mean_ratio + self.eps)) * 100
#         return {
#             'MEAN_Actual': mean_actual,
#             'MEAN_Predicted': mean_pred,
#             'MEAN_Absolute_Error': mean_absolute_error_val,
#             'MEAN_Ratio': mean_ratio,
#             'STD_Actual': std_actual,
#             'STD_Predicted': std_pred,
#             'STD_Ratio': std_ratio,
#             'COV_Actual(%)': cov_actual,
#             'COV_Predicted(%)': cov_pred,
#             'COV_Absolute_Error(%)': cov_absolute_error,
#             'COV_Ratio(%)': cov_ratio
#         }
#
#     def plot_comparison(self, X_test, y_test):
#         individual_preds = self.predict_individual(X_test)
#         ensemble_pred = self.predict(X_test)
#         model_r2 = {name: r2_score(y_test, pred) for name, pred in individual_preds.items()}
#         ensemble_r2 = r2_score(y_test, ensemble_pred)
#         plt.figure(figsize=(12, 7))
#         models_list = list(model_r2.keys()) + ['Optuna Ensemble']
#         r2_scores = list(model_r2.values()) + [ensemble_r2]
#         colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#FF6B6B']
#         bars = plt.bar(models_list, r2_scores, color=colors, alpha=1.0)
#         plt.ylabel('R² Score', fontsize=16, fontweight='bold')
#         plt.title('Performance Comparison Between Base Models and Ensemble Model', fontsize=18, fontweight='bold')
#         plt.xticks(rotation=45, fontsize=14, fontweight='bold')
#         plt.yticks(fontsize=14, fontweight='bold')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=6)
#         ax.tick_params(axis='both', which='minor', width=1.5, length=3)
#         plt.grid(True, alpha=0.3, axis='y')
#         for bar, score in zip(bars, r2_scores):
#             plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{score:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
#         plt.axhline(y=ensemble_r2, color='r', linestyle='--', alpha=0.7,
#                     label=f'Ensemble R² ({ensemble_r2:.4f})')
#         plt.legend(fontsize=14, prop={'weight': 'bold', 'family': 'Times New Roman'})
#         plt.tight_layout()
#         plt.savefig('CEGS模型性能对比.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         return model_r2, ensemble_r2
#
#     def plot_predictions_vs_actual(self, X_train, X_test, y_train, y_test):
#         y_train_pred = self.predict(X_train)
#         y_test_pred = self.predict(X_test)
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_mape = mean_absolute_percentage_error(np.where(np.abs(y_train)<self.eps,self.eps,y_train), y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_mape = mean_absolute_percentage_error(np.where(np.abs(y_test)<self.eps,self.eps,y_test), y_test_pred)
#         plt.figure(figsize=(10, 8))
#         plt.scatter(y_train, y_train_pred, color='#64B5CD', s=100, marker='o', alpha=1,
#                     label=f'Train List')
#         plt.scatter(y_test, y_test_pred, color='#D4BE83', s=100, marker='s', alpha=1,
#                     label=f'Test List')
#         plt.xlabel('True ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         plt.ylabel('Predicted ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=8)
#         ax.tick_params(axis='both', which='minor', width=2.0, length=4)
#         text_box = f"R²: {test_r2:.4f}\nMAE: {test_mae:.4f} kN\nRMSE: {test_rmse:.4f} kN\nMAPE: {test_mape:.2%}"
#         plt.text(70, 2400, text_box, fontsize=20, fontfamily='Times New Roman',
#                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#         plt.plot([0, 3500], [0, 3500], "--k")
#         plt.plot([0, 3500], [0 * 1.15, 3500 * 1.15], "--b", alpha=0.7, label='±15% Error Bound')
#         plt.plot([0, 3500], [0 * 0.85, 3500 * 0.85], "--b", alpha=0.7)
#         plt.xlim(0, 3500)
#         plt.ylim(0, 3500)
#         plt.legend(fontsize=20, loc='upper left', prop={'family': 'Times New Roman'})
#         plt.xticks(fontsize=20, fontfamily='Times New Roman')
#         plt.yticks(fontsize=20, fontfamily='Times New Roman')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('CEGS Regression.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
#         plt.close()
#         print(f"\n训练集指标:")
#         print(f"R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
#         print(f"\n测试集指标:")
#         print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")
#         train_cov_metrics = self.calculate_cov_and_mean(X_train, y_train)
#         test_cov_metrics = self.calculate_cov_and_mean(X_test, y_test)
#         print(f"\n训练集COV和MEAN统计:")
#         for key, value in train_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         print(f"\n测试集COV和MEAN统计:")
#         for key, value in test_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         return y_train_pred, y_test_pred, train_cov_metrics, test_cov_metrics



# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from catboost import CatBoostRegressor
# import optuna
# from optuna.pruners import MedianPruner
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
# plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', "Microsoft YaHei", "SimHei"]
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 600
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['figure.edgecolor'] = 'white'
#
# GLOBAL_RANDOM_SEED = 1443
#
# class OptunaOptimizedWeightedVotingRegressor:
#     def __init__(self, n_trials=100, manual_weights=None):
#         self.models = {}
#         self.weights = {}
#         self.manual_weights = manual_weights
#         self.scaler_x_train = StandardScaler()
#         self.scaler_y_train = StandardScaler()
#         self.is_fitted = False
#         self.best_params_ = {}
#         self.n_trials = n_trials
#         self.X_train_fit_ = None
#         self.y_train_fit_ = None
#         self.eps = 1e-8
#         self.cv_folds = 10
#
#     def _optimize_extra_trees(self, X_train, y_train):
#         print("\n使用Optuna优化ExtraTrees超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 80, 200),
#                 'max_depth': trial.suggest_int('max_depth', 2, 8),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 3, 8),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 6),
#                 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
#                 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'n_jobs': -1
#             }
#             model = ExtraTreesRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"ExtraTrees最佳参数: {study.best_params}")
#         print(f"ExtraTrees最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         best_params['n_jobs'] = -1
#         return ExtraTreesRegressor(**best_params)
#
#     def _optimize_gbdt(self, X_train, y_train):
#         print("\n使用Optuna优化GBDT超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 50, 200),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'max_depth': trial.suggest_int('max_depth', 2, 6),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 7),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 6),
#                 'subsample': trial.suggest_float('subsample', 0.4, 1.0),
#                 'alpha': trial.suggest_float('alpha', 0.01, 0.1),
#                 'random_state': GLOBAL_RANDOM_SEED
#             }
#             model = GradientBoostingRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"GBDT最佳参数: {study.best_params}")
#         print(f"GBDT最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         return GradientBoostingRegressor(**best_params)
#
#     def _optimize_svr(self, X_train, y_train, scaler_x, scaler_y):
#         print("\n使用Optuna优化SVR超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         X_train_scaled = scaler_x.fit_transform(X_train)
#         y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
#         def objective(trial):
#             kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
#             base_params = {
#                 "C": trial.suggest_float("C", 0.01, 500, log=True),
#                 "epsilon": trial.suggest_float("epsilon", 0.001, 0.5),
#                 "cache_size": 1000
#             }
#             if kernel == "rbf":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_rbf", ["scale", "auto", 0.001, 0.01, 0.1, 1, 10, 100])
#             elif kernel == "poly":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_poly", ["scale", "auto", 0.001, 0.01, 0.1, 1])
#                 base_params["degree"] = trial.suggest_int("degree", 2, 5)
#             elif kernel == "sigmoid":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_sigmoid", ["scale", "auto", 0.001, 0.01, 0.1, 1])
#                 base_params["coef0"] = trial.suggest_float("coef0", -0.5, 1.5)
#             model = SVR(kernel=kernel,** base_params)
#             scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=cv_kfold, scoring="r2", n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         best_raw_params = study.best_params
#         print(f"SVR最佳参数: {best_raw_params}")
#         print(f"SVR最佳交叉验证R²分数: {study.best_value:.4f}")
#         best_params = {}
#         kernel = best_raw_params["kernel"]
#         best_params["kernel"] = kernel
#         best_params["C"] = best_raw_params["C"]
#         best_params["epsilon"] = best_raw_params["epsilon"]
#         best_params["cache_size"] = 1000
#         if kernel == "rbf":
#             best_params["gamma"] = best_raw_params["gamma_rbf"]
#         elif kernel == "poly":
#             best_params["gamma"] = best_raw_params["gamma_poly"]
#             best_params["degree"] = best_raw_params["degree"]
#         elif kernel == "sigmoid":
#             best_params["gamma"] = best_raw_params["gamma_sigmoid"]
#             best_params["coef0"] = best_raw_params["coef0"]
#         return SVR(**best_params)
#
#     def _optimize_catboost(self, X_train, y_train):
#         print("\n使用Optuna优化CatBoost超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'iterations': trial.suggest_int('iterations', 150, 1200),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'depth': trial.suggest_int('depth', 2, 6),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),
#                 'random_strength': trial.suggest_float('random_strength', 0.5, 5),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2.0),
#                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#                 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
#                 'loss_function': 'RMSE',
#                 'eval_metric': 'RMSE',
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'verbose': False,
#                 'thread_count': -1,
#                 'early_stopping_rounds': 100
#             }
#             model = CatBoostRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"CatBoost最佳参数: {study.best_params}")
#         print(f"CatBoost最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params.update({
#             'loss_function': 'RMSE',
#             'eval_metric': 'RMSE',
#             'random_state': GLOBAL_RANDOM_SEED,
#             'verbose': False,
#             'thread_count': -1,
#             'early_stopping_rounds': 100
#         })
#         return CatBoostRegressor(**best_params)
#
#     def _initialize_optimized_models(self, X_train_main, y_train_main):
#         print("开始Optuna超参数优化...")
#         temp_scaler_x = StandardScaler()
#         temp_scaler_y = StandardScaler()
#         optimized_et = self._optimize_extra_trees(X_train_main, y_train_main)
#         optimized_gbdt = self._optimize_gbdt(X_train_main, y_train_main)
#         optimized_svr = self._optimize_svr(X_train_main, y_train_main, temp_scaler_x, temp_scaler_y)
#         optimized_catboost = self._optimize_catboost(X_train_main, y_train_main)
#         self.models = {
#             'ExtraTrees': optimized_et,
#             'GBDT': optimized_gbdt,
#             'SVR': optimized_svr,
#             'CatBoost': optimized_catboost
#         }
#         self.best_params_ = {
#             'ExtraTrees': optimized_et.get_params(),
#             'GBDT': optimized_gbdt.get_params(),
#             'SVR': optimized_svr.get_params(),
#             'CatBoost': optimized_catboost.get_params()
#         }
#
#     def _get_cv_avg_r2(self, model, X, y, is_svr=False):
#         cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         r2_list = []
#         for train_idx, val_idx in cv.split(X):
#             X_cv_train, X_cv_val = X[train_idx], X[val_idx]
#             y_cv_train, y_cv_val = y[train_idx], y[val_idx]
#             if is_svr:
#                 scaler_x_cv = StandardScaler()
#                 scaler_y_cv = StandardScaler()
#                 X_cv_tr_sca = scaler_x_cv.fit_transform(X_cv_train)
#                 y_cv_tr_sca = scaler_y_cv.fit_transform(y_cv_train.reshape(-1,1)).ravel()
#                 model.fit(X_cv_tr_sca, y_cv_tr_sca)
#                 X_cv_vl_sca = scaler_x_cv.transform(X_cv_val)
#                 pred_sca = model.predict(X_cv_vl_sca)
#                 pred = scaler_y_cv.inverse_transform(pred_sca.reshape(-1,1)).ravel()
#             else:
#                 model.fit(X_cv_train, y_cv_train)
#                 pred = model.predict(X_cv_val)
#             r2 = r2_score(y_cv_val, pred)
#             r2_list.append(max(r2, self.eps))
#         return np.mean(r2_list)
#
#     def _calculate_model_weights(self):
#         required_models = ['ExtraTrees', 'GBDT', 'SVR', 'CatBoost']
#         if self.manual_weights is not None:
#             if not all(model in self.manual_weights for model in required_models):
#                 raise ValueError(f"手动权重字典必须包含以下所有模型: {required_models}")
#             weight_sum = sum(self.manual_weights.values())
#             norm_weights = {k: v / (weight_sum + self.eps) for k, v in self.manual_weights.items()}
#             self.weights = norm_weights
#             print("\n使用手动固定并归一化后的模型权重:")
#             for name, weight in self.weights.items():
#                 print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#             return
#         print("\n未检测到手动权重，基于10折交叉验证平均R²自动计算权重...")
#         performance_scores = {}
#         for name, model in self.models.items():
#             is_svr_flag = (name == "SVR")
#             avg_r2 = self._get_cv_avg_r2(model, self.X_train_fit_, self.y_train_fit_, is_svr=is_svr_flag)
#             performance_scores[name] = max(avg_r2, self.eps)
#         total_score = sum(performance_scores.values())
#         self.weights = {name: score / total_score for name, score in performance_scores.items()}
#         print("\n基于交叉验证平均R²自动计算的最终模型权重分配:")
#         for name, weight in self.weights.items():
#             print(f"  {name}: CV R²={performance_scores[name]:.4f}, 权重={weight:.4f} ({weight*100:.1f}%)")
#
#     def fit(self, X, y, val_ratio=0.3):
#         print("开始训练Optuna优化加权的投票集成模型...")
#         self.X_train_fit_ = X
#         self.y_train_fit_ = y
#         # 直接使用全部数据进行超参数优化和训练
#         self._initialize_optimized_models(X, y)
#         print("\n使用全部数据拟合标准化缩放器...")
#         self.scaler_x_train.fit(X)
#         self.scaler_y_train.fit(y.reshape(-1, 1))
#         X_scaled_full = self.scaler_x_train.transform(X)
#         y_scaled_full = self.scaler_y_train.transform(y.reshape(-1, 1)).ravel()
#         for name, model in self.models.items():
#             print(f"全量数据集重训 {name}...")
#             if name == 'SVR':
#                 model.fit(X_scaled_full, y_scaled_full)
#             else:
#                 model.fit(X, y)
#         self._calculate_model_weights()
#         self.is_fitted = True
#         print("\nOptuna优化集成模型训练完成！")
#
#     def predict(self, X):
#         if not self.is_fitted:
#             raise ValueError("模型尚未训练，请先调用fit方法")
#         predictions = np.zeros(len(X))
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.scaler_x_train.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 pred = self.scaler_y_train.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 pred = model.predict(X)
#             predictions += self.weights[name] * pred
#         # 强制预测值非负（承载力不能为负），提高物理合理性和指标
#         predictions = np.maximum(predictions, 0)
#         return predictions
#
#     def predict_individual(self, X):
#         individual_predictions = {}
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.scaler_x_train.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 individual_predictions[name] = self.scaler_y_train.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 individual_predictions[name] = model.predict(X)
#         return individual_predictions
#
#     def evaluate(self, X, y):
#         predictions = self.predict(X)
#         y_safe = np.where(np.abs(y) < self.eps, self.eps, y)
#         mape = mean_absolute_percentage_error(y_safe, predictions)
#         metrics = {
#             'MSE': mean_squared_error(y, predictions),
#             'RMSE': np.sqrt(mean_squared_error(y, predictions)),
#             'MAE': mean_absolute_error(y, predictions),
#             'R2': r2_score(y, predictions),
#             'MAPE': mape
#         }
#         return metrics, predictions
#
#     def calculate_cov_and_mean(self, X, y):
#         y_pred = self.predict(X)
#         mean_actual = np.mean(y)
#         mean_pred = np.mean(y_pred)
#         std_actual = np.std(y, ddof=1)
#         std_pred = np.std(y_pred, ddof=1)
#         absolute_errors = np.abs(y - y_pred)
#         mean_absolute_error_val = np.mean(absolute_errors)
#         std_absolute_error = np.std(absolute_errors, ddof=1)
#         ratio = y_pred / (y + self.eps)
#         mean_ratio = np.mean(ratio)
#         std_ratio = np.std(ratio, ddof=1)
#         cov_actual = (std_actual / (mean_actual + self.eps)) * 100
#         cov_pred = (std_pred / (mean_pred + self.eps)) * 100
#         cov_absolute_error = (std_absolute_error / (mean_absolute_error_val + self.eps)) * 100
#         cov_ratio = (std_ratio / (mean_ratio + self.eps)) * 100
#         return {
#             'MEAN_Actual': mean_actual,
#             'MEAN_Predicted': mean_pred,
#             'MEAN_Absolute_Error': mean_absolute_error_val,
#             'MEAN_Ratio': mean_ratio,
#             'STD_Actual': std_actual,
#             'STD_Predicted': std_pred,
#             'STD_Ratio': std_ratio,
#             'COV_Actual(%)': cov_actual,
#             'COV_Predicted(%)': cov_pred,
#             'COV_Absolute_Error(%)': cov_absolute_error,
#             'COV_Ratio(%)': cov_ratio
#         }
#
#     def plot_comparison(self, X_test, y_test):
#         individual_preds = self.predict_individual(X_test)
#         ensemble_pred = self.predict(X_test)
#         model_r2 = {name: r2_score(y_test, pred) for name, pred in individual_preds.items()}
#         ensemble_r2 = r2_score(y_test, ensemble_pred)
#         plt.figure(figsize=(12, 7))
#         models_list = list(model_r2.keys()) + ['Optuna Ensemble']
#         r2_scores = list(model_r2.values()) + [ensemble_r2]
#         colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#FF6B6B']
#         bars = plt.bar(models_list, r2_scores, color=colors, alpha=1.0)
#         plt.ylabel('R² Score', fontsize=16, fontweight='bold')
#         plt.title('Performance Comparison Between Base Models and Ensemble Model', fontsize=18, fontweight='bold')
#         plt.xticks(rotation=45, fontsize=14, fontweight='bold')
#         plt.yticks(fontsize=14, fontweight='bold')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=6)
#         ax.tick_params(axis='both', which='minor', width=1.5, length=3)
#         plt.grid(True, alpha=0.3, axis='y')
#         for bar, score in zip(bars, r2_scores):
#             plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{score:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
#         plt.axhline(y=ensemble_r2, color='r', linestyle='--', alpha=0.7,
#                     label=f'Ensemble R² ({ensemble_r2:.4f})')
#         plt.legend(fontsize=14, prop={'weight': 'bold', 'family': 'Times New Roman'})
#         plt.tight_layout()
#         plt.savefig('CEGS模型性能对比.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         return model_r2, ensemble_r2
#
#     def plot_predictions_vs_actual(self, X_train, X_test, y_train, y_test):
#         y_train_pred = self.predict(X_train)
#         y_test_pred = self.predict(X_test)
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_mape = mean_absolute_percentage_error(np.where(np.abs(y_train)<self.eps,self.eps,y_train), y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_mape = mean_absolute_percentage_error(np.where(np.abs(y_test)<self.eps,self.eps,y_test), y_test_pred)
#         plt.figure(figsize=(10, 8))
#         plt.scatter(y_train, y_train_pred, color='#64B5CD', s=100, marker='o', alpha=1,
#                     label=f'Train List')
#         plt.scatter(y_test, y_test_pred, color='#D4BE83', s=100, marker='s', alpha=1,
#                     label=f'Test List')
#         plt.xlabel('True ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         plt.ylabel('Predicted ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=8)
#         ax.tick_params(axis='both', which='minor', width=2.0, length=4)
#         text_box = f"R²: {test_r2:.4f}\nMAE: {test_mae:.4f} kN\nRMSE: {test_rmse:.4f} kN\nMAPE: {test_mape:.2%}"
#         plt.text(70, 2400, text_box, fontsize=20, fontfamily='Times New Roman',
#                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#         plt.plot([0, 3500], [0, 3500], "--k")
#         plt.plot([0, 3500], [0 * 1.15, 3500 * 1.15], "--b", alpha=0.7, label='±15% Error Bound')
#         plt.plot([0, 3500], [0 * 0.85, 3500 * 0.85], "--b", alpha=0.7)
#         plt.xlim(0, 3500)
#         plt.ylim(0, 3500)
#         plt.legend(fontsize=20, loc='upper left', prop={'family': 'Times New Roman'})
#         plt.xticks(fontsize=20, fontfamily='Times New Roman')
#         plt.yticks(fontsize=20, fontfamily='Times New Roman')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('CEGS Regression.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
#         plt.close()
#         print(f"\n训练集指标:")
#         print(f"R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
#         print(f"\n测试集指标:")
#         print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")
#         train_cov_metrics = self.calculate_cov_and_mean(X_train, y_train)
#         test_cov_metrics = self.calculate_cov_and_mean(X_test, y_test)
#         print(f"\n训练集COV和MEAN统计:")
#         for key, value in train_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         print(f"\n测试集COV和MEAN统计:")
#         for key, value in test_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         return y_train_pred, y_test_pred, train_cov_metrics, test_cov_metrics


# False Best
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from catboost import CatBoostRegressor
# import optuna
# from optuna.pruners import MedianPruner
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
# plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', "Microsoft YaHei", "SimHei"]
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 600
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['figure.edgecolor'] = 'white'
#
# GLOBAL_RANDOM_SEED = 1443
#
# class OptunaOptimizedWeightedVotingRegressor:
#     def __init__(self, n_trials=200, manual_weights=None):
#         self.models = {}
#         self.weights = {}
#         self.manual_weights = manual_weights
#         self.scaler_x_train = StandardScaler()
#         self.scaler_y_train = StandardScaler()
#         self.is_fitted = False
#         self.best_params_ = {}
#         self.n_trials = n_trials
#         self.X_train_fit_ = None
#         self.y_train_fit_ = None
#         self.eps = 1e-8
#         self.cv_folds = 10
#
#     def _optimize_extra_trees(self, X_train, y_train):
#         print("\n使用Optuna优化ExtraTrees超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#                 'max_depth': trial.suggest_int('max_depth', 2, 10),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 3, 10),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
#                 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
#                 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'n_jobs': -1
#             }
#             model = ExtraTreesRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"ExtraTrees最佳参数: {study.best_params}")
#         print(f"ExtraTrees最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         best_params['n_jobs'] = -1
#         return ExtraTreesRegressor(**best_params)
#
#     def _optimize_gbdt(self, X_train, y_train):
#         print("\n使用Optuna优化GBDT超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 300),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'max_depth': trial.suggest_int('max_depth', 2, 8),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
#                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#                 'alpha': trial.suggest_float('alpha', 0.01, 0.2),
#                 'random_state': GLOBAL_RANDOM_SEED
#             }
#             model = GradientBoostingRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"GBDT最佳参数: {study.best_params}")
#         print(f"GBDT最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         return GradientBoostingRegressor(**best_params)
#
#     def _optimize_svr(self, X_train, y_train, scaler_x, scaler_y):
#         print("\n使用Optuna优化SVR超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         X_train_scaled = scaler_x.fit_transform(X_train)
#         y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
#         def objective(trial):
#             kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
#             base_params = {
#                 "C": trial.suggest_float("C", 0.01, 500, log=True),
#                 "epsilon": trial.suggest_float("epsilon", 0.001, 0.5),
#                 "cache_size": 1000
#             }
#             if kernel == "rbf":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_rbf", ["scale", "auto", 0.001, 0.01, 0.1, 1, 10, 100])
#             elif kernel == "poly":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_poly", ["scale", "auto", 0.001, 0.01, 0.1, 1])
#                 base_params["degree"] = trial.suggest_int("degree", 2, 5)
#             elif kernel == "sigmoid":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_sigmoid", ["scale", "auto", 0.001, 0.01, 0.1, 1])
#                 base_params["coef0"] = trial.suggest_float("coef0", -0.5, 1.5)
#             model = SVR(kernel=kernel,** base_params)
#             scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=cv_kfold, scoring="r2", n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         best_raw_params = study.best_params
#         print(f"SVR最佳参数: {best_raw_params}")
#         print(f"SVR最佳交叉验证R²分数: {study.best_value:.4f}")
#         best_params = {}
#         kernel = best_raw_params["kernel"]
#         best_params["kernel"] = kernel
#         best_params["C"] = best_raw_params["C"]
#         best_params["epsilon"] = best_raw_params["epsilon"]
#         best_params["cache_size"] = 1000
#         if kernel == "rbf":
#             best_params["gamma"] = best_raw_params["gamma_rbf"]
#         elif kernel == "poly":
#             best_params["gamma"] = best_raw_params["gamma_poly"]
#             best_params["degree"] = best_raw_params["degree"]
#         elif kernel == "sigmoid":
#             best_params["gamma"] = best_raw_params["gamma_sigmoid"]
#             best_params["coef0"] = best_raw_params["coef0"]
#         return SVR(**best_params)
#
#     def _optimize_catboost(self, X_train, y_train):
#         print("\n使用Optuna优化CatBoost超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'iterations': trial.suggest_int('iterations', 300, 1500),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'depth': trial.suggest_int('depth', 2, 8),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 15),
#                 'random_strength': trial.suggest_float('random_strength', 0.5, 8),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2.0),
#                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#                 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
#                 'loss_function': 'RMSE',
#                 'eval_metric': 'RMSE',
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'verbose': False,
#                 'thread_count': -1,
#                 'early_stopping_rounds': 100
#             }
#             model = CatBoostRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"CatBoost最佳参数: {study.best_params}")
#         print(f"CatBoost最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params.update({
#             'loss_function': 'RMSE',
#             'eval_metric': 'RMSE',
#             'random_state': GLOBAL_RANDOM_SEED,
#             'verbose': False,
#             'thread_count': -1,
#             'early_stopping_rounds': 100
#         })
#         return CatBoostRegressor(**best_params)
#
#     def _initialize_optimized_models(self, X_train_main, y_train_main):
#         print("开始Optuna超参数优化...")
#         temp_scaler_x = StandardScaler()
#         temp_scaler_y = StandardScaler()
#         optimized_et = self._optimize_extra_trees(X_train_main, y_train_main)
#         optimized_gbdt = self._optimize_gbdt(X_train_main, y_train_main)
#         optimized_svr = self._optimize_svr(X_train_main, y_train_main, temp_scaler_x, temp_scaler_y)
#         optimized_catboost = self._optimize_catboost(X_train_main, y_train_main)
#         self.models = {
#             'ExtraTrees': optimized_et,
#             'GBDT': optimized_gbdt,
#             'SVR': optimized_svr,
#             'CatBoost': optimized_catboost
#         }
#         self.best_params_ = {
#             'ExtraTrees': optimized_et.get_params(),
#             'GBDT': optimized_gbdt.get_params(),
#             'SVR': optimized_svr.get_params(),
#             'CatBoost': optimized_catboost.get_params()
#         }
#
#     def _get_cv_avg_r2(self, model, X, y, is_svr=False):
#         cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         r2_list = []
#         for train_idx, val_idx in cv.split(X):
#             X_cv_train, X_cv_val = X[train_idx], X[val_idx]
#             y_cv_train, y_cv_val = y[train_idx], y[val_idx]
#             if is_svr:
#                 scaler_x_cv = StandardScaler()
#                 scaler_y_cv = StandardScaler()
#                 X_cv_tr_sca = scaler_x_cv.fit_transform(X_cv_train)
#                 y_cv_tr_sca = scaler_y_cv.fit_transform(y_cv_train.reshape(-1,1)).ravel()
#                 model.fit(X_cv_tr_sca, y_cv_tr_sca)
#                 X_cv_vl_sca = scaler_x_cv.transform(X_cv_val)
#                 pred_sca = model.predict(X_cv_vl_sca)
#                 pred = scaler_y_cv.inverse_transform(pred_sca.reshape(-1,1)).ravel()
#             else:
#                 model.fit(X_cv_train, y_cv_train)
#                 pred = model.predict(X_cv_val)
#             r2 = r2_score(y_cv_val, pred)
#             r2_list.append(max(r2, self.eps))
#         return np.mean(r2_list)
#
#     def _calculate_model_weights(self, X_train_main, y_train_main):
#         required_models = ['ExtraTrees', 'GBDT', 'SVR', 'CatBoost']
#         if self.manual_weights is not None:
#             if not all(model in self.manual_weights for model in required_models):
#                 raise ValueError(f"手动权重字典必须包含以下所有模型: {required_models}")
#             weight_sum = sum(self.manual_weights.values())
#             norm_weights = {k: v / (weight_sum + self.eps) for k, v in self.manual_weights.items()}
#             self.weights = norm_weights
#             print("\n使用手动固定并归一化后的模型权重:")
#             for name, weight in self.weights.items():
#                 print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#             return
#         print("\n未检测到手动权重，基于10折交叉验证平均R²复合指标自动计算权重...")
#         performance_scores = {}
#         individual_metrics = {}
#         for name, model in self.models.items():
#             is_svr_flag = (name == "SVR")
#             avg_r2 = self._get_cv_avg_r2(model, X_train_main, y_train_main, is_svr=is_svr_flag)
#             X_train_main_scaled = self.scaler_x_train.transform(X_train_main) if is_svr_flag else X_train_main
#             if is_svr_flag:
#                 pred_scaled = model.predict(X_train_main_scaled)
#                 pred = self.scaler_y_train.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
#             else:
#                 pred = model.predict(X_train_main)
#             r2 = r2_score(y_train_main, pred)
#             rmse = np.sqrt(mean_squared_error(y_train_main, pred))
#             mae = mean_absolute_error(y_train_main, pred)
#             mape = mean_absolute_percentage_error(np.where(np.abs(y_train_main)<self.eps,self.eps,y_train_main), pred)
#             individual_metrics[name] = {
#                 'R2': r2,
#                 'RMSE': rmse,
#                 'MAE': mae,
#                 'MAPE': mape
#             }
#             composite_score = max(avg_r2, self.eps) / (rmse + self.eps)
#             performance_scores[name] = composite_score
#         print("\n各个模型在训练集交叉验证表现:")
#         for name, metrics in individual_metrics.items():
#             print(f"  {name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
#         total_score = sum(performance_scores.values())
#         self.weights = {name: score / total_score for name, score in performance_scores.items()}
#         print("\n基于交叉验证复合性能自动计算的最终模型权重分配:")
#         for name, weight in self.weights.items():
#             print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#
#     def fit(self, X, y, val_ratio=0.3):
#         print("开始训练Optuna优化加权的投票集成模型...")
#         self.X_train_fit_ = X
#         self.y_train_fit_ = y
#         # ★ 不再划分验证集，使用全部数据进行优化和训练
#         self._initialize_optimized_models(X, y)
#         print("\n使用全部数据拟合标准化缩放器...")
#         self.scaler_x_train.fit(X)
#         self.scaler_y_train.fit(y.reshape(-1, 1))
#         X_scaled_full = self.scaler_x_train.transform(X)
#         y_scaled_full = self.scaler_y_train.transform(y.reshape(-1, 1)).ravel()
#         for name, model in self.models.items():
#             print(f"全量数据集重训 {name}...")
#             if name == 'SVR':
#                 model.fit(X_scaled_full, y_scaled_full)
#             else:
#                 model.fit(X, y)
#         self._calculate_model_weights(X, y)   # 权重计算也基于全部数据
#         self.is_fitted = True
#         print("\nOptuna优化集成模型训练完成！")
#
#     def predict(self, X):
#         if not self.is_fitted:
#             raise ValueError("模型尚未训练，请先调用fit方法")
#         predictions = np.zeros(len(X))
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.scaler_x_train.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 pred = self.scaler_y_train.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 pred = model.predict(X)
#             predictions += self.weights[name] * pred
#         # 承载力不应为负，提升物理合理性和微小精度
#         predictions = np.maximum(predictions, 0)
#         return predictions
#
#     def predict_individual(self, X):
#         individual_predictions = {}
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.scaler_x_train.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 individual_predictions[name] = self.scaler_y_train.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 individual_predictions[name] = model.predict(X)
#         return individual_predictions
#
#     def evaluate(self, X, y):
#         predictions = self.predict(X)
#         y_safe = np.where(np.abs(y) < self.eps, self.eps, y)
#         mape = mean_absolute_percentage_error(y_safe, predictions)
#         metrics = {
#             'MSE': mean_squared_error(y, predictions),
#             'RMSE': np.sqrt(mean_squared_error(y, predictions)),
#             'MAE': mean_absolute_error(y, predictions),
#             'R2': r2_score(y, predictions),
#             'MAPE': mape
#         }
#         return metrics, predictions
#
#     def calculate_cov_and_mean(self, X, y):
#         y_pred = self.predict(X)
#         mean_actual = np.mean(y)
#         mean_pred = np.mean(y_pred)
#         std_actual = np.std(y, ddof=1)
#         std_pred = np.std(y_pred, ddof=1)
#         absolute_errors = np.abs(y - y_pred)
#         mean_absolute_error_val = np.mean(absolute_errors)
#         std_absolute_error = np.std(absolute_errors, ddof=1)
#         ratio = y_pred / (y + self.eps)
#         mean_ratio = np.mean(ratio)
#         std_ratio = np.std(ratio, ddof=1)
#         cov_actual = (std_actual / (mean_actual + self.eps)) * 100
#         cov_pred = (std_pred / (mean_pred + self.eps)) * 100
#         cov_absolute_error = (std_absolute_error / (mean_absolute_error_val + self.eps)) * 100
#         cov_ratio = (std_ratio / (mean_ratio + self.eps)) * 100
#         return {
#             'MEAN_Actual': mean_actual,
#             'MEAN_Predicted': mean_pred,
#             'MEAN_Absolute_Error': mean_absolute_error_val,
#             'MEAN_Ratio': mean_ratio,
#             'STD_Actual': std_actual,
#             'STD_Predicted': std_pred,
#             'STD_Ratio': std_ratio,
#             'COV_Actual(%)': cov_actual,
#             'COV_Predicted(%)': cov_pred,
#             'COV_Absolute_Error(%)': cov_absolute_error,
#             'COV_Ratio(%)': cov_ratio
#         }
#
#     def plot_comparison(self, X_test, y_test):
#         individual_preds = self.predict_individual(X_test)
#         ensemble_pred = self.predict(X_test)
#         model_r2 = {name: r2_score(y_test, pred) for name, pred in individual_preds.items()}
#         ensemble_r2 = r2_score(y_test, ensemble_pred)
#         plt.figure(figsize=(12, 7))
#         models_list = list(model_r2.keys()) + ['Optuna Ensemble']
#         r2_scores = list(model_r2.values()) + [ensemble_r2]
#         colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#FF6B6B']
#         bars = plt.bar(models_list, r2_scores, color=colors, alpha=1.0)
#         plt.ylabel('R² Score', fontsize=16, fontweight='bold')
#         plt.title('Performance Comparison Between Base Models and Ensemble Model', fontsize=18, fontweight='bold')
#         plt.xticks(rotation=45, fontsize=14, fontweight='bold')
#         plt.yticks(fontsize=14, fontweight='bold')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=6)
#         ax.tick_params(axis='both', which='minor', width=1.5, length=3)
#         plt.grid(True, alpha=0.3, axis='y')
#         for bar, score in zip(bars, r2_scores):
#             plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{score:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
#         plt.axhline(y=ensemble_r2, color='r', linestyle='--', alpha=0.7,
#                     label=f'Ensemble R² ({ensemble_r2:.4f})')
#         plt.legend(fontsize=14, prop={'weight': 'bold', 'family': 'Times New Roman'})
#         plt.tight_layout()
#         plt.savefig('CEGS模型性能对比.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         return model_r2, ensemble_r2
#
#     def plot_predictions_vs_actual(self, X_train, X_test, y_train, y_test):
#         y_train_pred = self.predict(X_train)
#         y_test_pred = self.predict(X_test)
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_mape = mean_absolute_percentage_error(np.where(np.abs(y_train)<self.eps,self.eps,y_train), y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_mape = mean_absolute_percentage_error(np.where(np.abs(y_test)<self.eps,self.eps,y_test), y_test_pred)
#         plt.figure(figsize=(10, 8))
#         plt.scatter(y_train, y_train_pred, color='#64B5CD', s=100, marker='o', alpha=1,
#                     label=f'Train List')
#         plt.scatter(y_test, y_test_pred, color='#D4BE83', s=100, marker='s', alpha=1,
#                     label=f'Test List')
#         plt.xlabel('True ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         plt.ylabel('Predicted ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=8)
#         ax.tick_params(axis='both', which='minor', width=2.0, length=4)
#         text_box = f"R²: {test_r2:.4f}\nMAE: {test_mae:.4f} kN\nRMSE: {test_rmse:.4f} kN\nMAPE: {test_mape:.2%}"
#         plt.text(70, 2400, text_box, fontsize=20, fontfamily='Times New Roman',
#                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#         plt.plot([0, 3500], [0, 3500], "--k")
#         plt.plot([0, 3500], [0 * 1.15, 3500 * 1.15], "--b", alpha=0.7, label='±15% Error Bound')
#         plt.plot([0, 3500], [0 * 0.85, 3500 * 0.85], "--b", alpha=0.7)
#         plt.xlim(0, 3500)
#         plt.ylim(0, 3500)
#         plt.legend(fontsize=20, loc='upper left', prop={'family': 'Times New Roman'})
#         plt.xticks(fontsize=20, fontfamily='Times New Roman')
#         plt.yticks(fontsize=20, fontfamily='Times New Roman')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('CEGS Regression.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
#         plt.close()
#         print(f"\n训练集指标:")
#         print(f"R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
#         print(f"\n测试集指标:")
#         print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")
#         train_cov_metrics = self.calculate_cov_and_mean(X_train, y_train)
#         test_cov_metrics = self.calculate_cov_and_mean(X_test, y_test)
#         print(f"\n训练集COV和MEAN统计:")
#         for key, value in train_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         print(f"\n测试集COV和MEAN统计:")
#         for key, value in test_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         return y_train_pred, y_test_pred, train_cov_metrics, test_cov_metrics


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
import optuna
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', "Microsoft YaHei", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.edgecolor'] = 'white'

GLOBAL_RANDOM_SEED = 1443

class OptunaOptimizedWeightedVotingRegressor:
    def __init__(self, n_trials=200, manual_weights=None):
        self.models = {}
        self.weights = {}
        self.manual_weights = manual_weights
        self.scaler_x_train = StandardScaler()
        self.scaler_y_train = StandardScaler()
        self.is_fitted = False
        self.best_params_ = {}
        self.n_trials = n_trials
        self.X_train_fit_ = None
        self.y_train_fit_ = None
        self.eps = 1e-8
        self.cv_folds = 10

    def _optimize_extra_trees(self, X_train, y_train):
        print("\n使用Optuna优化ExtraTrees超参数...")
        cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 3, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': GLOBAL_RANDOM_SEED,
                'n_jobs': -1
            }
            model = ExtraTreesRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
            return scores.mean()
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_warmup_steps=10),
            sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        print(f"ExtraTrees最佳参数: {study.best_params}")
        print(f"ExtraTrees最佳分数: {study.best_value:.4f}")
        best_params = study.best_params.copy()
        best_params['random_state'] = GLOBAL_RANDOM_SEED
        best_params['n_jobs'] = -1
        return ExtraTreesRegressor(**best_params)

    def _optimize_gbdt(self, X_train, y_train):
        print("\n使用Optuna优化GBDT超参数...")
        cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'alpha': trial.suggest_float('alpha', 0.01, 0.2),
                'random_state': GLOBAL_RANDOM_SEED
            }
            model = GradientBoostingRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
            return scores.mean()
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_warmup_steps=10),
            sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        print(f"GBDT最佳参数: {study.best_params}")
        print(f"GBDT最佳分数: {study.best_value:.4f}")
        best_params = study.best_params.copy()
        best_params['random_state'] = GLOBAL_RANDOM_SEED
        return GradientBoostingRegressor(**best_params)

    def _optimize_svr(self, X_train, y_train, scaler_x, scaler_y):
        print("\n使用Optuna优化SVR超参数...")
        cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
        X_train_scaled = scaler_x.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        def objective(trial):
            kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
            base_params = {
                "C": trial.suggest_float("C", 0.01, 500, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.001, 0.5),
                "cache_size": 1000
            }
            if kernel == "rbf":
                base_params["gamma"] = trial.suggest_categorical("gamma_rbf", ["scale", "auto", 0.001, 0.01, 0.1, 1, 10, 100])
            elif kernel == "poly":
                base_params["gamma"] = trial.suggest_categorical("gamma_poly", ["scale", "auto", 0.001, 0.01, 0.1, 1])
                base_params["degree"] = trial.suggest_int("degree", 2, 5)
            elif kernel == "sigmoid":
                base_params["gamma"] = trial.suggest_categorical("gamma_sigmoid", ["scale", "auto", 0.001, 0.01, 0.1, 1])
                base_params["coef0"] = trial.suggest_float("coef0", -0.5, 1.5)
            model = SVR(kernel=kernel,** base_params)
            scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=cv_kfold, scoring="r2", n_jobs=-1)
            return scores.mean()
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_warmup_steps=10),
            sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best_raw_params = study.best_params
        print(f"SVR最佳参数: {best_raw_params}")
        print(f"SVR最佳交叉验证R²分数: {study.best_value:.4f}")
        best_params = {}
        kernel = best_raw_params["kernel"]
        best_params["kernel"] = kernel
        best_params["C"] = best_raw_params["C"]
        best_params["epsilon"] = best_raw_params["epsilon"]
        best_params["cache_size"] = 1000
        if kernel == "rbf":
            best_params["gamma"] = best_raw_params["gamma_rbf"]
        elif kernel == "poly":
            best_params["gamma"] = best_raw_params["gamma_poly"]
            best_params["degree"] = best_raw_params["degree"]
        elif kernel == "sigmoid":
            best_params["gamma"] = best_raw_params["gamma_sigmoid"]
            best_params["coef0"] = best_raw_params["coef0"]
        return SVR(**best_params)

    def _optimize_catboost(self, X_train, y_train):
        print("\n使用Optuna优化CatBoost超参数...")
        cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 2, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 15),
                'random_strength': trial.suggest_float('random_strength', 0.5, 8),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'random_state': GLOBAL_RANDOM_SEED,
                'verbose': False,
                'thread_count': -1,
                'early_stopping_rounds': 100
            }
            model = CatBoostRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
            return scores.mean()
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_warmup_steps=10),
            sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        print(f"CatBoost最佳参数: {study.best_params}")
        print(f"CatBoost最佳分数: {study.best_value:.4f}")
        best_params = study.best_params.copy()
        best_params.update({
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_state': GLOBAL_RANDOM_SEED,
            'verbose': False,
            'thread_count': -1,
            'early_stopping_rounds': 100
        })
        return CatBoostRegressor(**best_params)

    def _initialize_optimized_models(self, X_train_main, y_train_main):
        print("开始Optuna超参数优化...")
        temp_scaler_x = StandardScaler()
        temp_scaler_y = StandardScaler()
        optimized_et = self._optimize_extra_trees(X_train_main, y_train_main)
        optimized_gbdt = self._optimize_gbdt(X_train_main, y_train_main)
        optimized_svr = self._optimize_svr(X_train_main, y_train_main, temp_scaler_x, temp_scaler_y)
        optimized_catboost = self._optimize_catboost(X_train_main, y_train_main)
        self.models = {
            'ExtraTrees': optimized_et,
            'GBDT': optimized_gbdt,
            'SVR': optimized_svr,
            'CatBoost': optimized_catboost
        }
        self.best_params_ = {
            'ExtraTrees': optimized_et.get_params(),
            'GBDT': optimized_gbdt.get_params(),
            'SVR': optimized_svr.get_params(),
            'CatBoost': optimized_catboost.get_params()
        }

    def _get_cv_avg_r2(self, model, X, y, is_svr=False):
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
        r2_list = []
        for train_idx, val_idx in cv.split(X):
            X_cv_train, X_cv_val = X[train_idx], X[val_idx]
            y_cv_train, y_cv_val = y[train_idx], y[val_idx]
            if is_svr:
                scaler_x_cv = StandardScaler()
                scaler_y_cv = StandardScaler()
                X_cv_tr_sca = scaler_x_cv.fit_transform(X_cv_train)
                y_cv_tr_sca = scaler_y_cv.fit_transform(y_cv_train.reshape(-1,1)).ravel()
                model.fit(X_cv_tr_sca, y_cv_tr_sca)
                X_cv_vl_sca = scaler_x_cv.transform(X_cv_val)
                pred_sca = model.predict(X_cv_vl_sca)
                pred = scaler_y_cv.inverse_transform(pred_sca.reshape(-1,1)).ravel()
            else:
                model.fit(X_cv_train, y_cv_train)
                pred = model.predict(X_cv_val)
            r2 = r2_score(y_cv_val, pred)
            r2_list.append(max(r2, self.eps))
        return np.mean(r2_list)

    def _calculate_model_weights(self, X_train_main, y_train_main):
        required_models = ['ExtraTrees', 'GBDT', 'SVR', 'CatBoost']
        if self.manual_weights is not None:
            if not all(model in self.manual_weights for model in required_models):
                raise ValueError(f"手动权重字典必须包含以下所有模型: {required_models}")
            weight_sum = sum(self.manual_weights.values())
            norm_weights = {k: v / (weight_sum + self.eps) for k, v in self.manual_weights.items()}
            self.weights = norm_weights
            print("\n使用手动固定并归一化后的模型权重:")
            for name, weight in self.weights.items():
                print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
            return
        print("\n未检测到手动权重，基于10折交叉验证平均R²复合指标自动计算权重...")
        performance_scores = {}
        individual_metrics = {}
        for name, model in self.models.items():
            is_svr_flag = (name == "SVR")
            avg_r2 = self._get_cv_avg_r2(model, X_train_main, y_train_main, is_svr=is_svr_flag)
            X_train_main_scaled = self.scaler_x_train.transform(X_train_main) if is_svr_flag else X_train_main
            if is_svr_flag:
                pred_scaled = model.predict(X_train_main_scaled)
                pred = self.scaler_y_train.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
            else:
                pred = model.predict(X_train_main)
            r2 = r2_score(y_train_main, pred)
            rmse = np.sqrt(mean_squared_error(y_train_main, pred))
            mae = mean_absolute_error(y_train_main, pred)
            mape = mean_absolute_percentage_error(np.where(np.abs(y_train_main)<self.eps,self.eps,y_train_main), pred)
            individual_metrics[name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            }
            composite_score = max(avg_r2, self.eps) / (rmse + self.eps)
            performance_scores[name] = composite_score
        print("\n各个模型在训练集交叉验证表现:")
        for name, metrics in individual_metrics.items():
            print(f"  {name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
        total_score = sum(performance_scores.values())
        self.weights = {name: score / total_score for name, score in performance_scores.items()}
        print("\n基于交叉验证复合性能自动计算的最终模型权重分配:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")

    def fit(self, X, y, val_ratio=0.3):
        print("开始训练Optuna优化加权的投票集成模型...")
        self.X_train_fit_ = X
        self.y_train_fit_ = y
        # ★ 不再划分验证集，使用全部数据进行优化和训练
        self._initialize_optimized_models(X, y)
        print("\n使用全部数据拟合标准化缩放器...")
        self.scaler_x_train.fit(X)
        self.scaler_y_train.fit(y.reshape(-1, 1))
        X_scaled_full = self.scaler_x_train.transform(X)
        y_scaled_full = self.scaler_y_train.transform(y.reshape(-1, 1)).ravel()
        for name, model in self.models.items():
            print(f"全量数据集重训 {name}...")
            if name == 'SVR':
                model.fit(X_scaled_full, y_scaled_full)
            else:
                model.fit(X, y)
        self._calculate_model_weights(X, y)   # 权重计算也基于全部数据
        self.is_fitted = True
        print("\nOptuna优化集成模型训练完成！")

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        predictions = np.zeros(len(X))
        for name, model in self.models.items():
            if name == 'SVR':
                X_scaled = self.scaler_x_train.transform(X)
                pred_scaled = model.predict(X_scaled)
                pred = self.scaler_y_train.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            else:
                pred = model.predict(X)
            predictions += self.weights[name] * pred
        # 承载力不应为负，提升物理合理性和微小精度
        predictions = np.maximum(predictions, 0)
        return predictions

    def predict_individual(self, X):
        individual_predictions = {}
        for name, model in self.models.items():
            if name == 'SVR':
                X_scaled = self.scaler_x_train.transform(X)
                pred_scaled = model.predict(X_scaled)
                individual_predictions[name] = self.scaler_y_train.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            else:
                individual_predictions[name] = model.predict(X)
        return individual_predictions

    def evaluate(self, X, y):
        predictions = self.predict(X)
        y_safe = np.where(np.abs(y) < self.eps, self.eps, y)
        mape = mean_absolute_percentage_error(y_safe, predictions)
        metrics = {
            'MSE': mean_squared_error(y, predictions),
            'RMSE': np.sqrt(mean_squared_error(y, predictions)),
            'MAE': mean_absolute_error(y, predictions),
            'R2': r2_score(y, predictions),
            'MAPE': mape
        }
        return metrics, predictions

    def calculate_cov_and_mean(self, X, y):
        y_pred = self.predict(X)
        mean_actual = np.mean(y)
        mean_pred = np.mean(y_pred)
        std_actual = np.std(y, ddof=1)
        std_pred = np.std(y_pred, ddof=1)
        absolute_errors = np.abs(y - y_pred)
        mean_absolute_error_val = np.mean(absolute_errors)
        std_absolute_error = np.std(absolute_errors, ddof=1)
        ratio = y_pred / (y + self.eps)
        mean_ratio = np.mean(ratio)
        std_ratio = np.std(ratio, ddof=1)
        cov_actual = (std_actual / (mean_actual + self.eps)) * 100
        cov_pred = (std_pred / (mean_pred + self.eps)) * 100
        cov_absolute_error = (std_absolute_error / (mean_absolute_error_val + self.eps)) * 100
        cov_ratio = (std_ratio / (mean_ratio + self.eps)) * 100
        return {
            'MEAN_Actual': mean_actual,
            'MEAN_Predicted': mean_pred,
            'MEAN_Absolute_Error': mean_absolute_error_val,
            'MEAN_Ratio': mean_ratio,
            'STD_Actual': std_actual,
            'STD_Predicted': std_pred,
            'STD_Ratio': std_ratio,
            'COV_Actual(%)': cov_actual,
            'COV_Predicted(%)': cov_pred,
            'COV_Absolute_Error(%)': cov_absolute_error,
            'COV_Ratio(%)': cov_ratio
        }

    def plot_comparison(self, X_test, y_test):
        individual_preds = self.predict_individual(X_test)
        ensemble_pred = self.predict(X_test)
        model_r2 = {name: r2_score(y_test, pred) for name, pred in individual_preds.items()}
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        plt.figure(figsize=(12, 7))
        models_list = list(model_r2.keys()) + ['Optuna Ensemble']
        r2_scores = list(model_r2.values()) + [ensemble_r2]
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#FF6B6B']
        bars = plt.bar(models_list, r2_scores, color=colors, alpha=1.0)
        plt.ylabel('R² Score', fontsize=16, fontweight='bold')
        plt.title('Performance Comparison Between Base Models and Ensemble Model', fontsize=18, fontweight='bold')
        plt.xticks(rotation=45, fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')
        ax = plt.gca()
        ax.spines['left'].set_linewidth(3.0)
        ax.spines['bottom'].set_linewidth(3.0)
        ax.spines['right'].set_linewidth(3.0)
        ax.spines['top'].set_linewidth(3.0)
        ax.tick_params(axis='both', which='major', width=2.0, length=6)
        ax.tick_params(axis='both', which='minor', width=1.5, length=3)
        plt.grid(True, alpha=0.3, axis='y')
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        plt.axhline(y=ensemble_r2, color='r', linestyle='--', alpha=0.7,
                    label=f'Ensemble R² ({ensemble_r2:.4f})')
        plt.legend(fontsize=14, prop={'weight': 'bold', 'family': 'Times New Roman'})
        plt.tight_layout()
        plt.savefig('CEGS模型性能对比.png', dpi=300, bbox_inches='tight')
        plt.close()
        return model_r2, ensemble_r2

    def plot_predictions_vs_actual(self, X_train, X_test, y_train, y_test):
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mape = mean_absolute_percentage_error(np.where(np.abs(y_train)<self.eps,self.eps,y_train), y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mape = mean_absolute_percentage_error(np.where(np.abs(y_test)<self.eps,self.eps,y_test), y_test_pred)

        print(f"\n训练集指标:")
        print(f"R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
        print(f"\n测试集指标:")
        print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")

        train_cov_metrics = self.calculate_cov_and_mean(X_train, y_train)
        test_cov_metrics = self.calculate_cov_and_mean(X_test, y_test)
        print(f"\n训练集COV和MEAN统计:")
        for key, value in train_cov_metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"\n测试集COV和MEAN统计:")
        for key, value in test_cov_metrics.items():
            print(f"  {key}: {value:.4f}")

        # 导出训练集和测试集预测结果到同一个 .dat 文件
        train_data = np.column_stack([np.full(len(y_train), 'train'), y_train, y_train_pred])
        test_data  = np.column_stack([np.full(len(y_test), 'test'), y_test, y_test_pred])
        output_data = np.vstack([train_data, test_data])
        header = 'Dataset\tActual\tPredicted'
        np.savetxt('CEGS_Fit.dat', output_data, fmt='%s', delimiter='\t', header=header, comments='')
        print("训练集和测试集预测结果已保存至 CEGS_Fit.dat")

        return y_train_pred, y_test_pred, train_cov_metrics, test_cov_metrics


# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from catboost import CatBoostRegressor
# import optuna
# from optuna.pruners import MedianPruner
# from optuna.samplers import NSGAIISampler
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
# plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', "Microsoft YaHei", "SimHei"]
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 600
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['figure.edgecolor'] = 'white'
#
# GLOBAL_RANDOM_SEED = 1443
#
# class OptunaOptimizedWeightedVotingRegressor:
#     def __init__(self, n_trials=600, manual_weights=None):
#         self.models = {}
#         self.weights = {}
#         self.manual_weights = manual_weights
#         # 使用 QuantileTransformer 替代 StandardScaler 以增强鲁棒性
#         self.scaler_x = QuantileTransformer(output_distribution='normal', random_state=GLOBAL_RANDOM_SEED)
#         # 目标变量变换器 (Box-Cox)
#         self.target_transformer = PowerTransformer(method='box-cox', standardize=False)
#         # 特征多项式扩展
#         self.poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
#         self.is_fitted = False
#         self.best_params_ = {}
#         self.n_trials = n_trials
#         self.X_train_fit_ = None
#         self.y_train_fit_ = None
#         self.eps = 1e-8
#         self.cv_folds = 10
#
#     def _transform_X(self, X, fit=False):
#         """特征预处理：先多项式扩展，再分位数标准化"""
#         if fit:
#             X_poly = self.poly.fit_transform(X)
#             X_scaled = self.scaler_x.fit_transform(X_poly)
#         else:
#             X_poly = self.poly.transform(X)
#             X_scaled = self.scaler_x.transform(X_poly)
#         return X_scaled
#
#     def _transform_y(self, y, fit=False):
#         """目标变量 Box-Cox 变换，要求 y > 0"""
#         y = y.reshape(-1, 1)
#         if fit:
#             y_trans = self.target_transformer.fit_transform(y)
#         else:
#             y_trans = self.target_transformer.transform(y)
#         return y_trans.ravel()
#
#     def _inverse_y(self, y_trans):
#         """逆变换回原始尺度"""
#         return self.target_transformer.inverse_transform(y_trans.reshape(-1, 1)).ravel()
#
#     def _optimize_extra_trees(self, X_train, y_train):
#         print("\n使用Optuna优化ExtraTrees超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#                 'max_depth': trial.suggest_int('max_depth', 2, 15),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
#                 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
#                 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'n_jobs': -1
#             }
#             model = ExtraTreesRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"ExtraTrees最佳参数: {study.best_params}")
#         print(f"ExtraTrees最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         best_params['n_jobs'] = -1
#         return ExtraTreesRegressor(**best_params)
#
#     def _optimize_gbdt(self, X_train, y_train):
#         print("\n使用Optuna优化GBDT超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
#                 'max_depth': trial.suggest_int('max_depth', 2, 10),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
#                 'subsample': trial.suggest_float('subsample', 0.4, 1.0),
#                 'alpha': trial.suggest_float('alpha', 0.01, 0.5),
#                 'random_state': GLOBAL_RANDOM_SEED
#             }
#             model = GradientBoostingRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"GBDT最佳参数: {study.best_params}")
#         print(f"GBDT最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         return GradientBoostingRegressor(**best_params)
#
#     def _optimize_svr(self, X_train, y_train):
#         print("\n使用Optuna优化SVR超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         # X_train 已经是预处理后的特征，无需在内部再次缩放
#         def objective(trial):
#             kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
#             base_params = {
#                 "C": trial.suggest_float("C", 0.01, 1000, log=True),
#                 "epsilon": trial.suggest_float("epsilon", 0.001, 0.5),
#                 "cache_size": 1000
#             }
#             if kernel == "rbf":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_rbf", ["scale", "auto", 0.001, 0.01, 0.1, 1, 10, 100])
#             elif kernel == "poly":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_poly", ["scale", "auto", 0.001, 0.01, 0.1, 1])
#                 base_params["degree"] = trial.suggest_int("degree", 2, 6)
#             elif kernel == "sigmoid":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_sigmoid", ["scale", "auto", 0.001, 0.01, 0.1, 1])
#                 base_params["coef0"] = trial.suggest_float("coef0", -1.0, 2.0)
#             model = SVR(kernel=kernel,** base_params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring="r2", n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         best_raw_params = study.best_params
#         print(f"SVR最佳参数: {best_raw_params}")
#         print(f"SVR最佳交叉验证R²分数: {study.best_value:.4f}")
#         best_params = {}
#         kernel = best_raw_params["kernel"]
#         best_params["kernel"] = kernel
#         best_params["C"] = best_raw_params["C"]
#         best_params["epsilon"] = best_raw_params["epsilon"]
#         best_params["cache_size"] = 1000
#         if kernel == "rbf":
#             best_params["gamma"] = best_raw_params["gamma_rbf"]
#         elif kernel == "poly":
#             best_params["gamma"] = best_raw_params["gamma_poly"]
#             best_params["degree"] = best_raw_params["degree"]
#         elif kernel == "sigmoid":
#             best_params["gamma"] = best_raw_params["gamma_sigmoid"]
#             best_params["coef0"] = best_raw_params["coef0"]
#         return SVR(**best_params)
#
#     def _optimize_catboost(self, X_train, y_train):
#         print("\n使用Optuna优化CatBoost超参数...")
#         cv_kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         def objective(trial):
#             params = {
#                 'iterations': trial.suggest_int('iterations', 200, 3000),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
#                 'depth': trial.suggest_int('depth', 2, 10),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 20),
#                 'random_strength': trial.suggest_float('random_strength', 0.2, 5),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 3.0),
#                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#                 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
#                 'loss_function': 'RMSE',
#                 'eval_metric': 'RMSE',
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'verbose': False,
#                 'thread_count': -1,
#                 'early_stopping_rounds': 150
#             }
#             model = CatBoostRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=10),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"CatBoost最佳参数: {study.best_params}")
#         print(f"CatBoost最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params.update({
#             'loss_function': 'RMSE',
#             'eval_metric': 'RMSE',
#             'random_state': GLOBAL_RANDOM_SEED,
#             'verbose': False,
#             'thread_count': -1,
#             'early_stopping_rounds': 150
#         })
#         return CatBoostRegressor(**best_params)
#
#     def _initialize_optimized_models(self, X_train_main, y_train_main):
#         print("开始Optuna超参数优化...")
#         optimized_et = self._optimize_extra_trees(X_train_main, y_train_main)
#         optimized_gbdt = self._optimize_gbdt(X_train_main, y_train_main)
#         optimized_svr = self._optimize_svr(X_train_main, y_train_main)
#         optimized_catboost = self._optimize_catboost(X_train_main, y_train_main)
#         self.models = {
#             'ExtraTrees': optimized_et,
#             'GBDT': optimized_gbdt,
#             'SVR': optimized_svr,
#             'CatBoost': optimized_catboost
#         }
#         self.best_params_ = {
#             'ExtraTrees': optimized_et.get_params(),
#             'GBDT': optimized_gbdt.get_params(),
#             'SVR': optimized_svr.get_params(),
#             'CatBoost': optimized_catboost.get_params()
#         }
#
#     def _get_cv_metrics(self, model, X, y, is_svr=False):
#         """交叉验证并返回原始尺度上的平均 R²、RMSE、MAE、MAPE"""
#         cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#         r2_list, rmse_list, mae_list, mape_list = [], [], [], []
#         for train_idx, val_idx in cv.split(X):
#             X_cv_train, X_cv_val = X[train_idx], X[val_idx]
#             y_cv_train, y_cv_val = y[train_idx], y[val_idx]
#             # 在训练子集上拟合目标变换器，保证真实模拟
#             trans_y_train = PowerTransformer(method='box-cox', standardize=False)
#             y_train_trans = trans_y_train.fit_transform(y_cv_train.reshape(-1,1)).ravel()
#             model.fit(X_cv_train, y_train_trans)
#             pred_trans = model.predict(X_cv_val)
#             pred = trans_y_train.inverse_transform(pred_trans.reshape(-1,1)).ravel()
#             r2_list.append(r2_score(y_cv_val, pred))
#             rmse_list.append(np.sqrt(mean_squared_error(y_cv_val, pred)))
#             mae_list.append(mean_absolute_error(y_cv_val, pred))
#             mape_list.append(mean_absolute_percentage_error(np.maximum(np.abs(y_cv_val), self.eps), pred))
#         return np.mean(r2_list), np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list)
#
#     def _calculate_model_weights(self, X_train_main, y_train_main):
#         required_models = ['ExtraTrees', 'GBDT', 'SVR', 'CatBoost']
#         if self.manual_weights is not None:
#             if not all(model in self.manual_weights for model in required_models):
#                 raise ValueError(f"手动权重字典必须包含以下所有模型: {required_models}")
#             weight_sum = sum(self.manual_weights.values())
#             norm_weights = {k: v / (weight_sum + self.eps) for k, v in self.manual_weights.items()}
#             self.weights = norm_weights
#             print("\n使用手动固定并归一化后的模型权重:")
#             for name, weight in self.weights.items():
#                 print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#             return
#         print("\n未检测到手动权重，基于10折交叉验证平均R²复合指标自动计算权重...")
#         performance_scores = {}
#         individual_metrics = {}
#         for name, model in self.models.items():
#             avg_r2, avg_rmse, avg_mae, avg_mape = self._get_cv_metrics(model, X_train_main, y_train_main)
#             individual_metrics[name] = {
#                 'R2': avg_r2,
#                 'RMSE': avg_rmse,
#                 'MAE': avg_mae,
#                 'MAPE': avg_mape
#             }
#             # 复合指标：R²越高越好，RMSE和MAPE越低越好
#             composite_score = max(avg_r2, self.eps) / (avg_rmse * (1 + avg_mape) + self.eps)
#             performance_scores[name] = composite_score
#         print("\n各个模型在训练集交叉验证表现:")
#         for name, metrics in individual_metrics.items():
#             print(f"  {name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
#         total_score = sum(performance_scores.values())
#         self.weights = {name: score / total_score for name, score in performance_scores.items()}
#         print("\n基于交叉验证复合性能自动计算的最终模型权重分配:")
#         for name, weight in self.weights.items():
#             print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#
#     def fit(self, X, y, val_ratio=0.3):
#         print("开始训练Optuna优化加权的投票集成模型...")
#         self.X_train_fit_ = X
#         self.y_train_fit_ = y
#         # 先进行特征工程拟合
#         X_processed = self._transform_X(X, fit=True)
#         # 目标变换拟合
#         y_transformed = self._transform_y(y, fit=True)
#         print("\n数据预处理完成（多项式特征扩展 + QuantileTransformer + Box-Cox变换）")
#         # 超参数优化和模型初始化使用变换后的数据
#         self._initialize_optimized_models(X_processed, y_transformed)
#         # 全量数据重训
#         for name, model in self.models.items():
#             print(f"全量数据集重训 {name}...")
#             model.fit(X_processed, y_transformed)
#         # 权重计算需要在原始目标尺度上进行，传回 X_processed 和原始 y
#         self._calculate_model_weights(X_processed, y)
#         self.is_fitted = True
#         print("\nOptuna优化集成模型训练完成！")
#
#     def predict(self, X):
#         if not self.is_fitted:
#             raise ValueError("模型尚未训练，请先调用fit方法")
#         X_processed = self._transform_X(X, fit=False)
#         predictions_transformed = np.zeros(len(X))
#         for name, model in self.models.items():
#             pred_trans = model.predict(X_processed)
#             predictions_transformed += self.weights[name] * pred_trans
#         # 逆变换回原始承载力
#         predictions = self._inverse_y(predictions_transformed)
#         return predictions
#
#     def predict_individual(self, X):
#         individual_predictions = {}
#         X_processed = self._transform_X(X, fit=False)
#         for name, model in self.models.items():
#             pred_trans = model.predict(X_processed)
#             individual_predictions[name] = self._inverse_y(pred_trans)
#         return individual_predictions
#
#     def evaluate(self, X, y):
#         predictions = self.predict(X)
#         y_safe = np.where(np.abs(y) < self.eps, self.eps, y)
#         mape = mean_absolute_percentage_error(y_safe, predictions)
#         metrics = {
#             'MSE': mean_squared_error(y, predictions),
#             'RMSE': np.sqrt(mean_squared_error(y, predictions)),
#             'MAE': mean_absolute_error(y, predictions),
#             'R2': r2_score(y, predictions),
#             'MAPE': mape
#         }
#         return metrics, predictions
#
#     def calculate_cov_and_mean(self, X, y):
#         y_pred = self.predict(X)
#         mean_actual = np.mean(y)
#         mean_pred = np.mean(y_pred)
#         std_actual = np.std(y, ddof=1)
#         std_pred = np.std(y_pred, ddof=1)
#         absolute_errors = np.abs(y - y_pred)
#         mean_absolute_error_val = np.mean(absolute_errors)
#         std_absolute_error = np.std(absolute_errors, ddof=1)
#         ratio = y_pred / (y + self.eps)
#         mean_ratio = np.mean(ratio)
#         std_ratio = np.std(ratio, ddof=1)
#         cov_actual = (std_actual / (mean_actual + self.eps)) * 100
#         cov_pred = (std_pred / (mean_pred + self.eps)) * 100
#         cov_absolute_error = (std_absolute_error / (mean_absolute_error_val + self.eps)) * 100
#         cov_ratio = (std_ratio / (mean_ratio + self.eps)) * 100
#         return {
#             'MEAN_Actual': mean_actual,
#             'MEAN_Predicted': mean_pred,
#             'MEAN_Absolute_Error': mean_absolute_error_val,
#             'MEAN_Ratio': mean_ratio,
#             'STD_Actual': std_actual,
#             'STD_Predicted': std_pred,
#             'STD_Ratio': std_ratio,
#             'COV_Actual(%)': cov_actual,
#             'COV_Predicted(%)': cov_pred,
#             'COV_Absolute_Error(%)': cov_absolute_error,
#             'COV_Ratio(%)': cov_ratio
#         }
#
#     def plot_comparison(self, X_test, y_test):
#         individual_preds = self.predict_individual(X_test)
#         ensemble_pred = self.predict(X_test)
#         model_r2 = {name: r2_score(y_test, pred) for name, pred in individual_preds.items()}
#         ensemble_r2 = r2_score(y_test, ensemble_pred)
#         plt.figure(figsize=(12, 7))
#         models_list = list(model_r2.keys()) + ['Optuna Ensemble']
#         r2_scores = list(model_r2.values()) + [ensemble_r2]
#         colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#FF6B6B']
#         bars = plt.bar(models_list, r2_scores, color=colors, alpha=1.0)
#         plt.ylabel('R² Score', fontsize=16, fontweight='bold')
#         plt.title('Performance Comparison Between Base Models and Ensemble Model', fontsize=18, fontweight='bold')
#         plt.xticks(rotation=45, fontsize=14, fontweight='bold')
#         plt.yticks(fontsize=14, fontweight='bold')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=6)
#         ax.tick_params(axis='both', which='minor', width=1.5, length=3)
#         plt.grid(True, alpha=0.3, axis='y')
#         for bar, score in zip(bars, r2_scores):
#             plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{score:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
#         plt.axhline(y=ensemble_r2, color='r', linestyle='--', alpha=0.7,
#                     label=f'Ensemble R² ({ensemble_r2:.4f})')
#         plt.legend(fontsize=14, prop={'weight': 'bold', 'family': 'Times New Roman'})
#         plt.tight_layout()
#         plt.savefig('CEGS模型性能对比.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         return model_r2, ensemble_r2
#
#     def plot_predictions_vs_actual(self, X_train, X_test, y_train, y_test):
#         y_train_pred = self.predict(X_train)
#         y_test_pred = self.predict(X_test)
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_mape = mean_absolute_percentage_error(np.where(np.abs(y_train)<self.eps,self.eps,y_train), y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_mape = mean_absolute_percentage_error(np.where(np.abs(y_test)<self.eps,self.eps,y_test), y_test_pred)
#         plt.figure(figsize=(10, 8))
#         plt.scatter(y_train, y_train_pred, color='#64B5CD', s=100, marker='o', alpha=1,
#                     label=f'Train List')
#         plt.scatter(y_test, y_test_pred, color='#D4BE83', s=100, marker='s', alpha=1,
#                     label=f'Test List')
#         plt.xlabel('True ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         plt.ylabel('Predicted ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=8)
#         ax.tick_params(axis='both', which='minor', width=2.0, length=4)
#         text_box = f"R²: {test_r2:.4f}\nMAE: {test_mae:.4f} kN\nRMSE: {test_rmse:.4f} kN\nMAPE: {test_mape:.2%}"
#         plt.text(70, 2400, text_box, fontsize=20, fontfamily='Times New Roman',
#                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#         plt.plot([0, 3500], [0, 3500], "--k")
#         plt.plot([0, 3500], [0 * 1.15, 3500 * 1.15], "--b", alpha=0.7, label='±15% Error Bound')
#         plt.plot([0, 3500], [0 * 0.85, 3500 * 0.85], "--b", alpha=0.7)
#         plt.xlim(0, 3500)
#         plt.ylim(0, 3500)
#         plt.legend(fontsize=20, loc='upper left', prop={'family': 'Times New Roman'})
#         plt.xticks(fontsize=20, fontfamily='Times New Roman')
#         plt.yticks(fontsize=20, fontfamily='Times New Roman')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('CEGS Regression.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
#         plt.close()
#         print(f"\n训练集指标:")
#         print(f"R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
#         print(f"\n测试集指标:")
#         print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")
#         train_cov_metrics = self.calculate_cov_and_mean(X_train, y_train)
#         test_cov_metrics = self.calculate_cov_and_mean(X_test, y_test)
#         print(f"\n训练集COV和MEAN统计:")
#         for key, value in train_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         print(f"\n测试集COV和MEAN统计:")
#         for key, value in test_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         return y_train_pred, y_test_pred, train_cov_metrics, test_cov_metrics




# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from catboost import CatBoostRegressor
# import optuna
# from optuna.pruners import MedianPruner
# import matplotlib.pyplot as plt
# import warnings
#
# warnings.filterwarnings('ignore')
# plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', "Microsoft YaHei", "SimHei"]
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 600
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['figure.edgecolor'] = 'white'
#
# GLOBAL_RANDOM_SEED = 1443
#
# class OptunaOptimizedWeightedVotingRegressor:
#     def __init__(self, n_trials=10, manual_weights=None):
#         self.models = {}
#         self.weights = {}
#         self.manual_weights = manual_weights
#         self.scaler_x = StandardScaler()
#         self.scaler_y = StandardScaler()
#         self.is_fitted = False
#         self.best_params_ = {}
#         self.n_trials = n_trials
#         self.X_train_fit_ = None
#         self.y_train_fit_ = None
#         self.eps = 1e-8
#
#     def _optimize_extra_trees(self, X_train, y_train):
#         print("\n使用Optuna优化ExtraTrees超参数...")
#         cv_kfold = KFold(n_splits=5, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 30, 60, step=15),
#                 'max_depth': trial.suggest_int('max_depth', 2, 5, step=2),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 3),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 4),
#                 'max_features': trial.suggest_categorical('max_features', ['sqrt']),
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'n_jobs': -1
#             }
#             model = ExtraTreesRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=2),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"ExtraTrees最佳参数: {study.best_params}")
#         print(f"ExtraTrees最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         best_params['n_jobs'] = -1
#         return ExtraTreesRegressor(**best_params)
#
#     def _optimize_gbdt(self, X_train, y_train):
#         print("\n使用Optuna优化GBDT超参数...")
#         cv_kfold = KFold(n_splits=5, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 15, 50, step=15),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.18, step=0.04),
#                 'max_depth': trial.suggest_int('max_depth', 1, 3),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 4),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 4),
#                 'subsample': trial.suggest_float('subsample', 0.6, 0.85, step=0.15),
#                 'random_state': GLOBAL_RANDOM_SEED
#             }
#             model = GradientBoostingRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=2),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"GBDT最佳参数: {study.best_params}")
#         print(f"GBDT最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         return GradientBoostingRegressor(**best_params)
#
#     def _optimize_svr(self, X_train, y_train):
#         print("\n使用Optuna优化SVR超参数...")
#         cv_kfold = KFold(n_splits=3, shuffle=True, random_state=1443)
#         X_train_scaled = self.scaler_x.fit_transform(X_train)
#         y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
#
#         def objective(trial):
#             kernel = trial.suggest_categorical("kernel", ["rbf"])
#             base_params = {
#                 "C": trial.suggest_float("C", 1, 30, step=5),
#                 "epsilon": trial.suggest_float("epsilon", 0.05, 0.18, step=0.05),
#                 "cache_size": 1000
#             }
#             if kernel == "rbf":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_rbf", ["scale", 0.1])
#             model = SVR(kernel=kernel, **base_params)
#             scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=cv_kfold,
#                                      scoring="neg_mean_squared_error", n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=2),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         best_raw_params = study.best_params
#         print(f"SVR最佳参数: {best_raw_params}")
#         print(f"SVR最佳交叉验证负MSE分数: {study.best_value:.4f}")
#         print(f"SVR最佳交叉验证RMSE: {np.sqrt(-study.best_value):.4f}")
#         best_params = {}
#         kernel = best_raw_params["kernel"]
#         best_params["kernel"] = kernel
#         best_params["C"] = best_raw_params["C"]
#         best_params["epsilon"] = best_raw_params["epsilon"]
#         best_params["cache_size"] = 1000
#         if kernel == "rbf":
#             best_params["gamma"] = best_raw_params["gamma_rbf"]
#         return SVR(**best_params)
#
#     def _optimize_catboost(self, X_train, y_train):
#         print("\n使用Optuna优化CatBoost超参数...")
#         cv_kfold = KFold(n_splits=5, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#
#         def objective(trial):
#             params = {
#                 'iterations': trial.suggest_int('iterations', 100, 400, step=150),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.22, step=0.06),
#                 'depth': trial.suggest_int('depth', 2, 3),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2, 4, step=1),
#                 'random_strength': trial.suggest_float('random_strength', 1.5, 2.5, step=0.5),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 0.4, 1.0, step=0.3),
#                 'subsample': trial.suggest_float('subsample', 0.7, 0.9, step=0.1),
#                 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.9, step=0.1),
#                 'loss_function': 'RMSE',
#                 'eval_metric': 'RMSE',
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'verbose': False,
#                 'thread_count': -1,
#                 'early_stopping_rounds': 30
#             }
#             model = CatBoostRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=2),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"CatBoost最佳参数: {study.best_params}")
#         print(f"CatBoost最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params.update({
#             'loss_function': 'RMSE',
#             'eval_metric': 'RMSE',
#             'random_state': GLOBAL_RANDOM_SEED,
#             'verbose': False,
#             'thread_count': -1,
#             'early_stopping_rounds': 30
#         })
#         return CatBoostRegressor(**best_params)
#
#     def _initialize_optimized_models(self, X_train, y_train):
#         print("开始Optuna超参数优化...")
#         optimized_et = self._optimize_extra_trees(X_train, y_train)
#         optimized_gbdt = self._optimize_gbdt(X_train, y_train)
#         optimized_svr = self._optimize_svr(X_train, y_train)
#         optimized_catboost = self._optimize_catboost(X_train, y_train)
#         self.models = {
#             'ExtraTrees': optimized_et,
#             'GBDT': optimized_gbdt,
#             'SVR': optimized_svr,
#             'CatBoost': optimized_catboost
#         }
#         self.best_params_ = {
#             'ExtraTrees': optimized_et.get_params(),
#             'GBDT': optimized_gbdt.get_params(),
#             'SVR': optimized_svr.get_params(),
#             'CatBoost': optimized_catboost.get_params()
#         }
#
#     def _calculate_model_weights(self, X_val, y_val):
#         required_models = ['ExtraTrees', 'GBDT', 'SVR', 'CatBoost']
#         if self.manual_weights is not None:
#             if not all(model in self.manual_weights for model in required_models):
#                 raise ValueError(f"手动权重字典必须包含以下所有模型: {required_models}")
#             weight_sum = sum(self.manual_weights.values())
#             norm_weights = {k: v / (weight_sum + self.eps) for k, v in self.manual_weights.items()}
#             self.weights = norm_weights
#             print("\n使用手动固定并归一化后的模型权重:")
#             for name, weight in self.weights.items():
#                 print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#             return
#         print("\n未检测到手动权重，将基于验证集R²自动计算权重...")
#         performance_scores = {}
#         individual_metrics = {}
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_val_scaled = self.scaler_x.transform(X_val)
#                 y_pred_scaled = model.predict(X_val_scaled)
#                 y_pred_original = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
#             elif name == 'CatBoost':
#                 y_pred_original = model.predict(X_val)
#             else:
#                 y_pred_original = model.predict(X_val)
#             r2 = r2_score(y_val, y_pred_original)
#             rmse = np.sqrt(mean_squared_error(y_val, y_pred_original))
#             mae = mean_absolute_error(y_val, y_pred_original)
#             mape = mean_absolute_percentage_error(y_val, y_pred_original)
#             individual_metrics[name] = {
#                 'R2': r2,
#                 'RMSE': rmse,
#                 'MAE': mae,
#                 'MAPE': mape
#             }
#             if r2 < 0:
#                 print(f"警告: {name} 的R²为负值 ({r2:.4f})，将使用保守权重")
#                 performance_scores[name] = self.eps
#             else:
#                 performance_scores[name] = max(r2, self.eps)
#         print("\n各个模型在验证集上的表现:")
#         for name, metrics in individual_metrics.items():
#             print(f"  {name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
#         total_score = sum(performance_scores.values())
#         self.weights = {name: score / total_score for name, score in performance_scores.items()}
#         print("\n基于验证集性能自动计算的最终模型权重分配:")
#         for name, weight in self.weights.items():
#             print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#
#     def fit(self, X, y, val_ratio=0.3):
#         print("开始训练Optuna优化加权的投票集成模型...")
#         self.X_train_fit_ = X
#         self.y_train_fit_ = y
#         X_train_main, X_val, y_train_main, y_val = train_test_split(
#             X, y, test_size=val_ratio, random_state=GLOBAL_RANDOM_SEED, shuffle=True
#         )
#         self._initialize_optimized_models(X_train_main, y_train_main)
#         print("\n在完整训练集上重新训练优化后的模型...")
#         self.scaler_x.fit(X)
#         self.scaler_y.fit(y.reshape(-1, 1))
#         X_scaled_full = self.scaler_x.transform(X)
#         y_scaled_full = self.scaler_y.transform(y.reshape(-1, 1)).ravel()
#         for name, model in self.models.items():
#             print(f"重新训练 {name}...")
#             if name == 'SVR':
#                 model.fit(X_scaled_full, y_scaled_full)
#             elif name == 'CatBoost':
#                 model.fit(X, y)
#             else:
#                 model.fit(X, y)
#         self._calculate_model_weights(X_val, y_val)
#         self.is_fitted = True
#         print("\nOptuna优化集成模型训练完成！")
#
#     def predict(self, X):
#         if not self.is_fitted:
#             raise ValueError("模型尚未训练，请先调用fit方法")
#         predictions = np.zeros(len(X))
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.scaler_x.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             elif name == 'CatBoost':
#                 pred = model.predict(X)
#             else:
#                 pred = model.predict(X)
#             predictions += self.weights[name] * pred
#         return predictions
#
#     def predict_individual(self, X):
#         individual_predictions = {}
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.scaler_x.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 individual_predictions[name] = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             elif name == 'CatBoost':
#                 individual_predictions[name] = model.predict(X)
#             else:
#                 individual_predictions[name] = model.predict(X)
#         return individual_predictions
#
#     def evaluate(self, X, y):
#         predictions = self.predict(X)
#         y_safe = np.where(np.abs(y) < self.eps, self.eps, y)
#         mape = mean_absolute_percentage_error(y, predictions)
#         metrics = {
#             'MSE': mean_squared_error(y, predictions),
#             'RMSE': np.sqrt(mean_squared_error(y, predictions)),
#             'MAE': mean_absolute_error(y, predictions),
#             'R2': r2_score(y, predictions),
#             'MAPE': mape
#         }
#         return metrics, predictions
#
#     def calculate_cov_and_mean(self, X, y):
#         y_pred = self.predict(X)
#         mean_actual = np.mean(y)
#         mean_pred = np.mean(y_pred)
#         std_actual = np.std(y, ddof=1)
#         std_pred = np.std(y_pred, ddof=1)
#         absolute_errors = np.abs(y - y_pred)
#         mean_absolute_error_val = np.mean(absolute_errors)
#         std_absolute_error = np.std(absolute_errors, ddof=1)
#         ratio = y_pred / (y + self.eps)
#         mean_ratio = np.mean(ratio)
#         std_ratio = np.std(ratio, ddof=1)
#         cov_actual = (std_actual / (mean_actual + self.eps)) * 100
#         cov_pred = (std_pred / (mean_pred + self.eps)) * 100
#         cov_absolute_error = (std_absolute_error / (mean_absolute_error_val + self.eps)) * 100
#         cov_ratio = (std_ratio / (mean_ratio + self.eps)) * 100
#         return {
#             'MEAN_Actual': mean_actual,
#             'MEAN_Predicted': mean_pred,
#             'MEAN_Absolute_Error': mean_absolute_error_val,
#             'MEAN_Ratio': mean_ratio,
#             'STD_Actual': std_actual,
#             'STD_Predicted': std_pred,
#             'STD_Ratio': std_ratio,
#             'COV_Actual(%)': cov_actual,
#             'COV_Predicted(%)': cov_pred,
#             'COV_Absolute_Error(%)': cov_absolute_error,
#             'COV_Ratio(%)': cov_ratio
#         }
#
#     def plot_comparison(self, X_test, y_test):
#         individual_preds = self.predict_individual(X_test)
#         ensemble_pred = self.predict(X_test)
#         model_r2 = {name: r2_score(y_test, pred) for name, pred in individual_preds.items()}
#         ensemble_r2 = r2_score(y_test, ensemble_pred)
#         plt.figure(figsize=(12, 7))
#         models_list = list(model_r2.keys()) + ['Optuna Ensemble']
#         r2_scores = list(model_r2.values()) + [ensemble_r2]
#         colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#FF6B6B']
#         bars = plt.bar(models_list, r2_scores, color=colors, alpha=1.0)
#         plt.ylabel('R² Score', fontsize=16, fontweight='bold')
#         plt.title('Performance Comparison Between Base Models and Ensemble Model', fontsize=18, fontweight='bold')
#         plt.xticks(rotation=45, fontsize=14, fontweight='bold')
#         plt.yticks(fontsize=14, fontweight='bold')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=6)
#         ax.tick_params(axis='both', which='minor', width=1.5, length=3)
#         plt.grid(True, alpha=0.3, axis='y')
#         for bar, score in zip(bars, r2_scores):
#             plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{score:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
#         plt.axhline(y=ensemble_r2, color='r', linestyle='--', alpha=0.7,
#                     label=f'Ensemble R² ({ensemble_r2:.4f})')
#         plt.legend(fontsize=14, prop={'weight': 'bold', 'family': 'Times New Roman'})
#         plt.tight_layout()
#         plt.savefig('CEGS模型性能对比.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         return model_r2, ensemble_r2
#
#     def plot_predictions_vs_actual(self, X_train, X_test, y_train, y_test):
#         y_train_pred = self.predict(X_train)
#         y_test_pred = self.predict(X_test)
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
#         plt.figure(figsize=(10, 8))
#         plt.scatter(y_train, y_train_pred, color='#64B5CD', s=100, marker='o', alpha=1,
#                     label=f'Train List')
#         plt.scatter(y_test, y_test_pred, color='#D4BE83', s=100, marker='s', alpha=1,
#                     label=f'Test List')
#         plt.xlabel('True ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         plt.ylabel('Predicted ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=8)
#         ax.tick_params(axis='both', which='minor', width=2.0, length=4)
#         text_box = f"R²: {test_r2:.4f}\nMAE: {test_mae:.4f} kN\nRMSE: {test_rmse:.4f} kN\nMAPE: {test_mape:.2%}"
#         plt.text(70, 2400, text_box, fontsize=20, fontfamily='Times New Roman',
#                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#         plt.plot([0, 3500], [0, 3500], "--k")
#         plt.plot([0, 3500], [0 * 1.15, 3500 * 1.15], "--b", alpha=0.7, label='±15% Error Bound')
#         plt.plot([0, 3500], [0 * 0.85, 3500 * 0.85], "--b", alpha=0.7)
#         plt.xlim(0, 3500)
#         plt.ylim(0, 3500)
#         plt.legend(fontsize=20, loc='upper left', prop={'family': 'Times New Roman'})
#         plt.xticks(fontsize=20, fontfamily='Times New Roman')
#         plt.yticks(fontsize=20, fontfamily='Times New Roman')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('CEGS Regression.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
#         plt.close()
#         print(f"\n训练集指标:")
#         print(f"R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
#         print(f"\n测试集指标:")
#         print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")
#         train_cov_metrics = self.calculate_cov_and_mean(X_train, y_train)
#         test_cov_metrics = self.calculate_cov_and_mean(X_test, y_test)
#         print(f"\n训练集COV和MEAN统计:")
#         for key, value in train_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         print(f"\n测试集COV和MEAN统计:")
#         for key, value in test_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         return y_train_pred, y_test_pred, train_cov_metrics, test_cov_metrics



# True Best
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from catboost import CatBoostRegressor
# import optuna
# from optuna.pruners import MedianPruner
# import matplotlib.pyplot as plt
# import warnings
#
# warnings.filterwarnings('ignore')
# plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', "Microsoft YaHei", "SimHei"]
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 600
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['figure.edgecolor'] = 'white'
#
# GLOBAL_RANDOM_SEED = 1443
#
# class OptunaOptimizedWeightedVotingRegressor:
#     # 进一步降低寻优次数 n_trials=1
#     def __init__(self, n_trials=1, manual_weights=None):
#         self.models = {}
#         self.weights = {}
#         self.manual_weights = manual_weights
#         self.scaler_x = StandardScaler()
#         self.scaler_y = StandardScaler()
#         self.is_fitted = False
#         self.best_params_ = {}
#         self.n_trials = n_trials
#         self.X_train_fit_ = None
#         self.y_train_fit_ = None
#         self.eps = 1e-8
#
#     def _optimize_extra_trees(self, X_train, y_train):
#         print("\n使用Optuna优化ExtraTrees超参数...")
#         # CV折数最低2折
#         cv_kfold = KFold(n_splits=2, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#
#         def objective(trial):
#             # 大幅降低树规模、限制深度、加大叶子样本，强制欠拟合
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 30, 40, step=10),
#                 'max_depth': trial.suggest_int('max_depth', 2, 3),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 3, 4),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 4, 6),
#                 'max_features': trial.suggest_categorical('max_features', ['sqrt']),
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'n_jobs': -1
#             }
#             model = ExtraTreesRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=0),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"ExtraTrees最佳参数: {study.best_params}")
#         print(f"ExtraTrees最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         best_params['n_jobs'] = -1
#         return ExtraTreesRegressor(**best_params)
#
#     def _optimize_gbdt(self, X_train, y_train):
#         print("\n使用Optuna优化GBDT超参数...")
#         cv_kfold = KFold(n_splits=2, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#
#         def objective(trial):
#             # 少树、浅深度、大叶子、低采样，抑制拟合
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 15, 25, step=10),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.15, 0.18, step=0.03),
#                 'max_depth': trial.suggest_int('max_depth', 1, 2),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 3, 4),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 4, 6),
#                 'subsample': trial.suggest_float('subsample', 0.6, 0.7, step=0.1),
#                 'random_state': GLOBAL_RANDOM_SEED
#             }
#             model = GradientBoostingRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=0),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"GBDT最佳参数: {study.best_params}")
#         print(f"GBDT最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = GLOBAL_RANDOM_SEED
#         return GradientBoostingRegressor(**best_params)
#
#     def _optimize_svr(self, X_train, y_train):
#         print("\n使用Optuna优化SVR超参数...")
#         cv_kfold = KFold(n_splits=2, shuffle=True, random_state=1443)
#         X_train_scaled = self.scaler_x.fit_transform(X_train)
#         y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
#
#         def objective(trial):
#             kernel = trial.suggest_categorical("kernel", ["rbf"])
#             # 加大epsilon容忍误差、降低C减小拟合力度
#             base_params = {
#                 "C": trial.suggest_float("C", 1, 10, step=9),
#                 "epsilon": trial.suggest_float("epsilon", 0.12, 0.18, step=0.06),
#                 "cache_size": 1000
#             }
#             if kernel == "rbf":
#                 base_params["gamma"] = trial.suggest_categorical("gamma_rbf", ["scale"])
#             model = SVR(kernel=kernel,** base_params)
#             scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=cv_kfold,
#                                      scoring="neg_mean_squared_error", n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=0),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         best_raw_params = study.best_params
#         print(f"SVR最佳参数: {best_raw_params}")
#         print(f"SVR最佳交叉验证负MSE分数: {study.best_value:.4f}")
#         print(f"SVR最佳交叉验证RMSE: {np.sqrt(-study.best_value):.4f}")
#         best_params = {}
#         kernel = best_raw_params["kernel"]
#         best_params["kernel"] = kernel
#         best_params["C"] = best_raw_params["C"]
#         best_params["epsilon"] = best_raw_params["epsilon"]
#         best_params["cache_size"] = 1000
#         if kernel == "rbf":
#             best_params["gamma"] = best_raw_params["gamma_rbf"]
#         return SVR(**best_params)
#
#     def _optimize_catboost(self, X_train, y_train):
#         print("\n使用Optuna优化CatBoost超参数...")
#         cv_kfold = KFold(n_splits=2, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
#
#         def objective(trial):
#             # 极少迭代、浅深度、强正则、极早停止，大幅削弱拟合
#             params = {
#                 'iterations': trial.suggest_int('iterations', 100, 150, step=50),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.18, 0.22, step=0.04),
#                 'depth': trial.suggest_int('depth', 2, 2),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 4, step=1),
#                 'random_strength': trial.suggest_float('random_strength', 2.0, 2.5, step=0.5),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 0.8, 1.0, step=0.2),
#                 'subsample': trial.suggest_float('subsample', 0.7, 0.7, step=0.1),
#                 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.7, step=0.1),
#                 'loss_function': 'RMSE',
#                 'eval_metric': 'RMSE',
#                 'random_state': GLOBAL_RANDOM_SEED,
#                 'verbose': False,
#                 'thread_count': -1,
#                 'early_stopping_rounds': 5
#             }
#             model = CatBoostRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=cv_kfold, scoring='r2', n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(
#             direction='maximize',
#             pruner=MedianPruner(n_warmup_steps=0),
#             sampler=optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_SEED)
#         )
#         study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
#         print(f"CatBoost最佳参数: {study.best_params}")
#         print(f"CatBoost最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params.update({
#             'loss_function': 'RMSE',
#             'eval_metric': 'RMSE',
#             'random_state': GLOBAL_RANDOM_SEED,
#             'verbose': False,
#             'thread_count': -1,
#             'early_stopping_rounds': 5
#         })
#         return CatBoostRegressor(**best_params)
#
#     def _initialize_optimized_models(self, X_train, y_train):
#         print("开始Optuna超参数优化...")
#         optimized_et = self._optimize_extra_trees(X_train, y_train)
#         optimized_gbdt = self._optimize_gbdt(X_train, y_train)
#         optimized_svr = self._optimize_svr(X_train, y_train)
#         optimized_catboost = self._optimize_catboost(X_train, y_train)
#         self.models = {
#             'ExtraTrees': optimized_et,
#             'GBDT': optimized_gbdt,
#             'SVR': optimized_svr,
#             'CatBoost': optimized_catboost
#         }
#         self.best_params_ = {
#             'ExtraTrees': optimized_et.get_params(),
#             'GBDT': optimized_gbdt.get_params(),
#             'SVR': optimized_svr.get_params(),
#             'CatBoost': optimized_catboost.get_params()
#         }
#
#     def _calculate_model_weights(self, X_val, y_val):
#         required_models = ['ExtraTrees', 'GBDT', 'SVR', 'CatBoost']
#         if self.manual_weights is not None:
#             if not all(model in self.manual_weights for model in required_models):
#                 raise ValueError(f"手动权重字典必须包含以下所有模型: {required_models}")
#             weight_sum = sum(self.manual_weights.values())
#             norm_weights = {k: v / (weight_sum + self.eps) for k, v in self.manual_weights.items()}
#             self.weights = norm_weights
#             print("\n使用手动固定并归一化后的模型权重:")
#             for name, weight in self.weights.items():
#                 print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#             return
#         print("\n未检测到手动权重，将基于验证集R²自动计算权重...")
#         performance_scores = {}
#         individual_metrics = {}
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_val_scaled = self.scaler_x.transform(X_val)
#                 y_pred_scaled = model.predict(X_val_scaled)
#                 y_pred_original = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
#             elif name == 'CatBoost':
#                 y_pred_original = model.predict(X_val)
#             else:
#                 y_pred_original = model.predict(X_val)
#             r2 = r2_score(y_val, y_pred_original)
#             rmse = np.sqrt(mean_squared_error(y_val, y_pred_original))
#             mae = mean_absolute_error(y_val, y_pred_original)
#             mape = mean_absolute_percentage_error(y_val, y_pred_original)
#             individual_metrics[name] = {
#                 'R2': r2,
#                 'RMSE': rmse,
#                 'MAE': mae,
#                 'MAPE': mape
#             }
#             if r2 < 0:
#                 print(f"警告: {name} 的R²为负值 ({r2:.4f})，将使用保守权重")
#                 performance_scores[name] = self.eps
#             else:
#                 performance_scores[name] = max(r2, self.eps)
#         print("\n各个模型在验证集上的表现:")
#         for name, metrics in individual_metrics.items():
#             print(f"  {name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
#         total_score = sum(performance_scores.values())
#         self.weights = {name: score / total_score for name, score in performance_scores.items()}
#         print("\n基于验证集性能自动计算的最终模型权重分配:")
#         for name, weight in self.weights.items():
#             print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#
#     def fit(self, X, y, val_ratio=0.3):
#         print("开始训练Optuna优化加权的投票集成模型...")
#         self.X_train_fit_ = X
#         self.y_train_fit_ = y
#         X_train_main, X_val, y_train_main, y_val = train_test_split(
#             X, y, test_size=val_ratio, random_state=GLOBAL_RANDOM_SEED, shuffle=True
#         )
#         self._initialize_optimized_models(X_train_main, y_train_main)
#         print("\n在完整训练集上重新训练优化后的模型...")
#         self.scaler_x.fit(X)
#         self.scaler_y.fit(y.reshape(-1, 1))
#         X_scaled_full = self.scaler_x.transform(X)
#         y_scaled_full = self.scaler_y.transform(y.reshape(-1, 1)).ravel()
#         for name, model in self.models.items():
#             print(f"重新训练 {name}...")
#             if name == 'SVR':
#                 model.fit(X_scaled_full, y_scaled_full)
#             elif name == 'CatBoost':
#                 model.fit(X, y)
#             else:
#                 model.fit(X, y)
#         self._calculate_model_weights(X_val, y_val)
#         self.is_fitted = True
#         print("\nOptuna优化集成模型训练完成！")
#
#     def predict(self, X):
#         if not self.is_fitted:
#             raise ValueError("模型尚未训练，请先调用fit方法")
#         predictions = np.zeros(len(X))
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.scaler_x.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             elif name == 'CatBoost':
#                 pred = model.predict(X)
#             else:
#                 pred = model.predict(X)
#             predictions += self.weights[name] * pred
#         return predictions
#
#     def predict_individual(self, X):
#         individual_predictions = {}
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.scaler_x.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 individual_predictions[name] = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             elif name == 'CatBoost':
#                 individual_predictions[name] = model.predict(X)
#             else:
#                 individual_predictions[name] = model.predict(X)
#         return individual_predictions
#
#     def evaluate(self, X, y):
#         predictions = self.predict(X)
#         y_safe = np.where(np.abs(y) < self.eps, self.eps, y)
#         mape = mean_absolute_percentage_error(y, predictions)
#         metrics = {
#             'MSE': mean_squared_error(y, predictions),
#             'RMSE': np.sqrt(mean_squared_error(y, predictions)),
#             'MAE': mean_absolute_error(y, predictions),
#             'R2': r2_score(y, predictions),
#             'MAPE': mape
#         }
#         return metrics, predictions
#
#     def calculate_cov_and_mean(self, X, y):
#         y_pred = self.predict(X)
#         mean_actual = np.mean(y)
#         mean_pred = np.mean(y_pred)
#         std_actual = np.std(y, ddof=1)
#         std_pred = np.std(y_pred, ddof=1)
#         absolute_errors = np.abs(y - y_pred)
#         mean_absolute_error_val = np.mean(absolute_errors)
#         std_absolute_error = np.std(absolute_errors, ddof=1)
#         ratio = y_pred / (y + self.eps)
#         mean_ratio = np.mean(ratio)
#         std_ratio = np.std(ratio, ddof=1)
#         cov_actual = (std_actual / (mean_actual + self.eps)) * 100
#         cov_pred = (std_pred / (mean_pred + self.eps)) * 100
#         cov_absolute_error = (std_absolute_error / (mean_absolute_error_val + self.eps)) * 100
#         cov_ratio = (std_ratio / (mean_ratio + self.eps)) * 100
#         return {
#             'MEAN_Actual': mean_actual,
#             'MEAN_Predicted': mean_pred,
#             'MEAN_Absolute_Error': mean_absolute_error_val,
#             'MEAN_Ratio': mean_ratio,
#             'STD_Actual': std_actual,
#             'STD_Predicted': std_pred,
#             'STD_Ratio': std_ratio,
#             'COV_Actual(%)': cov_actual,
#             'COV_Predicted(%)': cov_pred,
#             'COV_Absolute_Error(%)': cov_absolute_error,
#             'COV_Ratio(%)': cov_ratio
#         }
#
#     def plot_comparison(self, X_test, y_test):
#         individual_preds = self.predict_individual(X_test)
#         ensemble_pred = self.predict(X_test)
#         model_r2 = {name: r2_score(y_test, pred) for name, pred in individual_preds.items()}
#         ensemble_r2 = r2_score(y_test, ensemble_pred)
#         plt.figure(figsize=(12, 7))
#         models_list = list(model_r2.keys()) + ['Optuna Ensemble']
#         r2_scores = list(model_r2.values()) + [ensemble_r2]
#         colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#FF6B6B']
#         bars = plt.bar(models_list, r2_scores, color=colors, alpha=1.0)
#         plt.ylabel('R² Score', fontsize=16, fontweight='bold')
#         plt.title('Performance Comparison Between Base Models and Ensemble Model', fontsize=18, fontweight='bold')
#         plt.xticks(rotation=45, fontsize=14, fontweight='bold')
#         plt.yticks(fontsize=14, fontweight='bold')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=6)
#         ax.tick_params(axis='both', which='minor', width=1.5, length=3)
#         plt.grid(True, alpha=0.3, axis='y')
#         for bar, score in zip(bars, r2_scores):
#             plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{score:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
#         plt.axhline(y=ensemble_r2, color='r', linestyle='--', alpha=0.7,
#                     label=f'Ensemble R² ({ensemble_r2:.4f})')
#         plt.legend(fontsize=14, prop={'weight': 'bold', 'family': 'Times New Roman'})
#         plt.tight_layout()
#         plt.savefig('CEGS模型性能对比.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         return model_r2, ensemble_r2
#
#     def plot_predictions_vs_actual(self, X_train, X_test, y_train, y_test):
#         y_train_pred = self.predict(X_train)
#         y_test_pred = self.predict(X_test)
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
#         plt.figure(figsize=(10, 8))
#         plt.scatter(y_train, y_train_pred, color='#64B5CD', s=100, marker='o', alpha=1,
#                     label=f'Train List')
#         plt.scatter(y_test, y_test_pred, color='#D4BE83', s=100, marker='s', alpha=1,
#                     label=f'Test List')
#         plt.xlabel('True ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         plt.ylabel('Predicted ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         ax = plt.gca()
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         ax.tick_params(axis='both', which='major', width=2.0, length=8)
#         ax.tick_params(axis='both', which='minor', width=2.0, length=4)
#         text_box = f"R²: {test_r2:.4f}\nMAE: {test_mae:.4f} kN\nRMSE: {test_rmse:.4f} kN\nMAPE: {test_mape:.2%}"
#         plt.text(70, 2400, text_box, fontsize=20, fontfamily='Times New Roman',
#                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#         plt.plot([0, 3500], [0, 3500], "--k")
#         plt.plot([0, 3500], [0 * 1.15, 3500 * 1.15], "--b", alpha=0.7, label='±15% Error Bound')
#         plt.plot([0, 3500], [0 * 0.85, 3500 * 0.85], "--b", alpha=0.7)
#         plt.xlim(0, 3500)
#         plt.ylim(0, 3500)
#         plt.legend(fontsize=20, loc='upper left', prop={'family': 'Times New Roman'})
#         plt.xticks(fontsize=20, fontfamily='Times New Roman')
#         plt.yticks(fontsize=20, fontfamily='Times New Roman')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('CEGS Regression.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
#         plt.close()
#         print(f"\n训练集指标:")
#         print(f"R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
#         print(f"\n测试集指标:")
#         print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")
#         train_cov_metrics = self.calculate_cov_and_mean(X_train, y_train)
#         test_cov_metrics = self.calculate_cov_and_mean(X_test, y_test)
#         print(f"\n训练集COV和MEAN统计:")
#         for key, value in train_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         print(f"\n测试集COV和MEAN统计:")
#         for key, value in test_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         return y_train_pred, y_test_pred, train_cov_metrics, test_cov_metrics


# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
# from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from catboost import CatBoostRegressor
# import optuna
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
# plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', "Microsoft YaHei", "SimHei"]
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 600
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams['figure.edgecolor'] = 'white'
#
# class OptunaOptimizedWeightedVotingRegressor:
#     ### 修改点 1: __init__ 方法增加 manual_weights 参数 ###
#     def __init__(self, n_trials=50, manual_weights=None):
#         self.models = {}
#         self.weights = {}
#         # 如果提供了手动权重，则直接设置
#         self.manual_weights = manual_weights
#         self.svr_scaler = StandardScaler()
#         self.target_scaler = StandardScaler()
#         self.is_fitted = False
#         self.best_params_ = {}
#         self.n_trials = n_trials
#         self.X_train_fit_ = None
#         self.y_train_fit_ = None
#
#     def _optimize_extra_trees(self, X_train, y_train):
#         print("\n使用Optuna优化ExtraTrees超参数...")
#
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#                 'max_depth': trial.suggest_int('max_depth', 10, 50),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
#                 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
#                 'random_state': 1443,
#                 'n_jobs': -1
#             }
#             model = ExtraTreesRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2', n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=self.n_trials)
#         print(f"ExtraTrees最佳参数: {study.best_params}")
#         print(f"ExtraTrees最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = 1443
#         best_params['n_jobs'] = -1
#         return ExtraTreesRegressor(**best_params)
#
#     def _optimize_gbdt(self, X_train, y_train):
#         print("\n使用Optuna优化GBDT超参数...")
#
#         def objective(trial):
#             params = {
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 800),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
#                 'max_depth': trial.suggest_int('max_depth', 3, 10),
#                 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
#                 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
#                 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#                 'random_state': 1443
#             }
#             model = GradientBoostingRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2', n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=self.n_trials)
#         print(f"GBDT最佳参数: {study.best_params}")
#         print(f"GBDT最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['random_state'] = 1443
#         return GradientBoostingRegressor(**best_params)
#
#     def _optimize_svr(self, X_train, y_train):
#         print("\n使用Optuna优化SVR超参数...")
#         y_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
#         X_scaled = self.svr_scaler.fit_transform(X_train)
#
#         def objective(trial):
#             params = {
#                 'C': trial.suggest_float('C', 0.1, 1000, log=True),
#                 'gamma': trial.suggest_float('gamma', 0.001, 1, log=True),
#                 'epsilon': trial.suggest_float('epsilon', 0.001, 0.5),
#                 'kernel': 'rbf',
#                 'cache_size': 1000
#             }
#             model = SVR(**params)
#             scores = cross_val_score(model, X_scaled, y_scaled, cv=10, scoring='r2', n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=self.n_trials)
#         print(f"SVR最佳参数: {study.best_params}")
#         print(f"SVR最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params['kernel'] = 'rbf'
#         best_params['cache_size'] = 1000
#         return SVR(**best_params)
#
#     def _optimize_catboost(self, X_train, y_train):
#         print("\n使用Optuna优化CatBoost超参数...")
#
#         def objective(trial):
#             params = {
#                 'iterations': trial.suggest_int('iterations', 500, 2000),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'depth': trial.suggest_int('depth', 4, 10),
#                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
#                 'random_strength': trial.suggest_float('random_strength', 0.1, 2),
#                 'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
#                 'loss_function': 'RMSE',
#                 'eval_metric': 'RMSE',
#                 'random_state': 1443,
#                 'verbose': False,
#                 'thread_count': -1
#             }
#             model = CatBoostRegressor(**params)
#             scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2', n_jobs=-1)
#             return scores.mean()
#
#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=self.n_trials)
#         print(f"CatBoost最佳参数: {study.best_params}")
#         print(f"CatBoost最佳分数: {study.best_value:.4f}")
#         best_params = study.best_params.copy()
#         best_params.update({
#             'loss_function': 'RMSE',
#             'eval_metric': 'RMSE',
#             'random_state': 1443,
#             'verbose': False,
#             'thread_count': -1
#         })
#         return CatBoostRegressor(**best_params)
#
#     def _initialize_optimized_models(self, X_train, y_train):
#         print("开始Optuna超参数优化...")
#         optimized_et = self._optimize_extra_trees(X_train, y_train)
#         optimized_gbdt = self._optimize_gbdt(X_train, y_train)
#         optimized_svr = self._optimize_svr(X_train, y_train)
#         optimized_catboost = self._optimize_catboost(X_train, y_train)
#         self.models = {
#             'ExtraTrees': optimized_et,
#             'GBDT': optimized_gbdt,
#             'SVR': optimized_svr,
#             'CatBoost': optimized_catboost
#         }
#         self.best_params_ = {
#             'ExtraTrees': optimized_et.get_params(),
#             'GBDT': optimized_gbdt.get_params(),
#             'SVR': optimized_svr.get_params(),
#             'CatBoost': optimized_catboost.get_params()
#         }
#
#     def _calculate_model_weights(self, X_val, y_val):
#         # 如果手动权重已设置，则直接使用
#         if self.manual_weights is not None:
#             # 简单验证权重字典的有效性
#             required_models = ['ExtraTrees', 'GBDT', 'SVR', 'CatBoost']
#             if not all(model in self.manual_weights for model in required_models):
#                 raise ValueError(f"手动权重字典必须包含以下所有模型: {required_models}")
#             if not np.isclose(sum(self.manual_weights.values()), 1.0, atol=1e-3):
#                 raise ValueError(f"手动权重的总和必须为 1.0")
#
#             self.weights = self.manual_weights
#             print("\n使用手动固定的模型权重:")
#             for name, weight in self.weights.items():
#                 print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#             return
#         # 否则，执行原有的自动计算权重逻辑
#         print("\n未检测到手动权重，将基于验证集性能自动计算权重...")
#         performance_scores = {}
#         individual_metrics = {}
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_val_scaled = self.svr_scaler.transform(X_val)
#                 y_pred_scaled = model.predict(X_val_scaled)
#                 y_pred_original = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 y_pred_original = model.predict(X_val)
#             r2 = r2_score(y_val, y_pred_original)
#             rmse = np.sqrt(mean_squared_error(y_val, y_pred_original))
#             mae = mean_absolute_error(y_val, y_pred_original)
#             mape = mean_absolute_percentage_error(y_val, y_pred_original)
#             individual_metrics[name] = {
#                 'R2': r2,
#                 'RMSE': rmse,
#                 'MAE': mae,
#                 'MAPE': mape
#             }
#             if r2 < 0:
#                 print(f"警告: {name} 的R²为负值 ({r2:.4f})，将使用保守权重")
#                 performance_scores[name] = 0.01
#             else:
#                 performance_scores[name] = max(r2, 0.05)
#
#         print("\n各个模型在验证集上的表现:")
#         for name, metrics in individual_metrics.items():
#             print(f"  {name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
#
#         total_score = sum(performance_scores.values())
#         self.weights = {name: score / total_score for name, score in performance_scores.items()}
#
#         print("\n基于验证集性能自动计算的最终模型权重分配:")
#         for name, weight in self.weights.items():
#             print(f"  {name}: {weight:.4f} ({weight * 100:.1f}%)")
#
#     def fit(self, X, y, val_ratio=0.3):
#         print("开始训练Optuna优化加权的投票集成模型...")
#         self.X_train_fit_ = X
#         self.y_train_fit_ = y
#
#         X_train_main, X_val, y_train_main, y_val = train_test_split(
#             X, y, test_size=val_ratio, random_state=42
#         )
#
#         self._initialize_optimized_models(X_train_main, y_train_main)
#
#         print("\n在完整训练集上重新训练优化后的模型...")
#         for name, model in self.models.items():
#             print(f"重新训练 {name}...")
#             if name == 'SVR':
#                 y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
#                 X_scaled = self.svr_scaler.fit_transform(X)
#                 model.fit(X_scaled, y_scaled)
#             else:
#                 model.fit(X, y)
#         # 调用权重计算方法（该方法内部会判断是否使用手动权重）
#         self._calculate_model_weights(X_val, y_val)
#
#         self.is_fitted = True
#         print("\nOptuna优化集成模型训练完成！")
#
#     def predict(self, X):
#         if not self.is_fitted:
#             raise ValueError("模型尚未训练，请先调用fit方法")
#         predictions = np.zeros(len(X))
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.svr_scaler.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 pred = model.predict(X)
#             predictions += self.weights[name] * pred
#         return predictions
#
#     def predict_individual(self, X):
#         individual_predictions = {}
#         for name, model in self.models.items():
#             if name == 'SVR':
#                 X_scaled = self.svr_scaler.transform(X)
#                 pred_scaled = model.predict(X_scaled)
#                 individual_predictions[name] = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
#             else:
#                 individual_predictions[name] = model.predict(X)
#         return individual_predictions
#
#     def evaluate(self, X, y):
#         predictions = self.predict(X)
#         metrics = {
#             'MSE': mean_squared_error(y, predictions),
#             'RMSE': np.sqrt(mean_squared_error(y, predictions)),
#             'MAE': mean_absolute_error(y, predictions),
#             'R2': r2_score(y, predictions),
#             'MAPE': mean_absolute_percentage_error(y, predictions)
#         }
#         return metrics, predictions
#
#     def calculate_cov_and_mean(self, X, y):
#         y_pred = self.predict(X)
#         mean_actual = np.mean(y)
#         mean_pred = np.mean(y_pred)
#         std_actual = np.std(y, ddof=1)
#         std_pred = np.std(y_pred, ddof=1)
#         absolute_errors = np.abs(y - y_pred)
#         mean_absolute_error_val = np.mean(absolute_errors)
#         std_absolute_error = np.std(absolute_errors, ddof=1)
#         ratio = y_pred / y
#         mean_ratio = np.mean(ratio)
#         std_ratio = np.std(ratio, ddof=1)
#         cov_actual = (std_actual / mean_actual) * 100 if mean_actual != 0 else 0
#         cov_pred = (std_pred / mean_pred) * 100 if mean_pred != 0 else 0
#         cov_absolute_error = (std_absolute_error / mean_absolute_error_val) * 100 if mean_absolute_error_val != 0 else 0
#         cov_ratio = (std_ratio / mean_ratio) * 100 if mean_ratio != 0 else 0
#         return {
#             'MEAN_Actual': mean_actual,
#             'MEAN_Predicted': mean_pred,
#             'MEAN_Absolute_Error': mean_absolute_error_val,
#             'MEAN_Ratio': mean_ratio,
#             'STD_Actual': std_actual,
#             'STD_Predicted': std_pred,
#             'STD_Absolute_Error': std_absolute_error,
#             'STD_Ratio': std_ratio,
#             'COV_Actual(%)': cov_actual,
#             'COV_Predicted(%)': cov_pred,
#             'COV_Absolute_Error(%)': cov_absolute_error,
#             'COV_Ratio(%)': cov_ratio
#         }
#
#     def plot_comparison(self, X_test, y_test):
#         individual_preds = self.predict_individual(X_test)
#         ensemble_pred = self.predict(X_test)
#         model_r2 = {name: r2_score(y_test, pred) for name, pred in individual_preds.items()}
#         ensemble_r2 = r2_score(y_test, ensemble_pred)
#         plt.figure(figsize=(12, 7))
#         models_list = list(model_r2.keys()) + ['Optuna Ensemble']
#         r2_scores = list(model_r2.values()) + [ensemble_r2]
#         colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFB366', '#FF6B6B']
#         bars = plt.bar(models_list, r2_scores, color=colors, alpha=1.0)
#         # 放大加粗y轴标签
#         plt.ylabel('R² Score', fontsize=16, fontweight='bold')
#         # 放大加粗标题
#         plt.title('Performance Comparison Between Base Models and Ensemble Model', fontsize=18, fontweight='bold')
#         # 放大加粗x轴刻度标签
#         plt.xticks(rotation=45, fontsize=14, fontweight='bold')
#         # 放大加粗y轴刻度标签
#         plt.yticks(fontsize=14, fontweight='bold')
#         # ========== 新增：加粗图表边线、坐标轴轴线和刻度线 ==========
#         ax = plt.gca()
#         # 加粗坐标轴轴线（x轴和y轴）
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         # 加粗刻度线
#         ax.tick_params(axis='both', which='major', width=2.0, length=6)
#         ax.tick_params(axis='both', which='minor', width=1.5, length=3)
#         # ==========================================================
#
#         plt.grid(True, alpha=0.3, axis='y')
#
#         # 放大加粗柱状图上的数值标签
#         for bar, score in zip(bars, r2_scores):
#             plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                      f'{score:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
#
#         # 修正：使用prop参数设置图例字体大小和粗细
#         plt.axhline(y=ensemble_r2, color='r', linestyle='--', alpha=0.7,
#                     label=f'Ensemble R² ({ensemble_r2:.4f})')
#         plt.legend(fontsize=14, prop={'weight': 'bold', 'family': 'Times New Roman'})
#
#         plt.tight_layout()
#         plt.savefig('CEGS模型性能对比.png', dpi=300, bbox_inches='tight')
#         plt.close()
#         return model_r2, ensemble_r2
#
#     def plot_predictions_vs_actual(self, X_train, X_test, y_train, y_test):
#         y_train_pred = self.predict(X_train)
#         y_test_pred = self.predict(X_test)
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
#
#         plt.figure(figsize=(10, 8))
#         # 图例文字放大加粗（通过label文字本身设置，实际效果由legend的prop控制）
#         plt.scatter(y_train, y_train_pred, color='#64B5CD', s=100, marker='o', alpha=1,
#                     label=f'Train List')
#         plt.scatter(y_test, y_test_pred, color='#D4BE83', s=100, marker='s', alpha=1,
#                     label=f'Test List')
#         # 放大加粗x轴标签
#         plt.xlabel('True ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         # 放大加粗y轴标签
#         plt.ylabel('Predicted ultimate bearing capacity (kN)', fontsize=28, fontfamily='Times New Roman')
#         # 放大加粗标题
#         # plt.title("CEGS Regression", fontsize=28, fontfamily='Times New Roman', pad=20)
#         # ========== 新增：加粗图表边线、坐标轴轴线和刻度线 ==========
#         ax = plt.gca()
#         # 加粗坐标轴轴线（x轴和y轴）
#         ax.spines['left'].set_linewidth(3.0)
#         ax.spines['bottom'].set_linewidth(3.0)
#         ax.spines['right'].set_linewidth(3.0)
#         ax.spines['top'].set_linewidth(3.0)
#         # 加粗刻度线
#         ax.tick_params(axis='both', which='major', width=2.0, length=8)
#         ax.tick_params(axis='both', which='minor', width=2.0, length=4)
#         # ==========================================================
#         # 放大加粗文本框文字
#         text_box = f"R²: {test_r2:.4f}\nMAE: {test_mae:.4f} kN\nRMSE: {test_rmse:.4f} kN\nMAPE: {test_mape:.2%}"
#         plt.text(70, 2400, text_box, fontsize=20, fontfamily='Times New Roman',
#                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#         plt.plot([0, 3500], [0, 3500], "--k")
#         plt.plot([0, 3500], [0 * 1.15, 3500 * 1.15], "--b", alpha=0.7, label='±15% Error Bound')
#         plt.plot([0, 3500], [0 * 0.85, 3500 * 0.85], "--b", alpha=0.7)
#         plt.xlim(0, 3500)
#         plt.ylim(0, 3500)
#         # 修正：使用prop参数设置图例字体大小和粗细
#         plt.legend(fontsize=20, loc='upper left', prop={'family': 'Times New Roman'})
#         # 放大加粗坐标轴刻度标签
#         plt.xticks(fontsize=20, fontfamily='Times New Roman')
#         plt.yticks(fontsize=20, fontfamily='Times New Roman')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('CEGS Regression.png', dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
#         plt.close()
#
#         print(f"\n训练集指标:")
#         print(f"R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
#         print(f"\n测试集指标:")
#         print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")
#
#         train_cov_metrics = self.calculate_cov_and_mean(X_train, y_train)
#         test_cov_metrics = self.calculate_cov_and_mean(X_test, y_test)
#
#         print(f"\n训练集COV和MEAN统计:")
#         for key, value in train_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#         print(f"\n测试集COV和MEAN统计:")
#         for key, value in test_cov_metrics.items():
#             print(f"  {key}: {value:.4f}")
#
#         return y_train_pred, y_test_pred, train_cov_metrics, test_cov_metrics