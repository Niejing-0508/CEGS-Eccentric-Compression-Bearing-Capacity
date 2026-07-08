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
