import numpy as np
import pandas as pd
import time
import json
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.naive_bayes import GaussianNB


def seed_torch(seed=20):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def train_model_repeat_cv_eval(repeats=5, k=5):
    # 加载数据
    dt_kmer_count_0 = pd.read_csv("../case_scripts_by_zhangzhiyuan/cases_res_files/xy_4kmer.csv", sep="\t", header=0, index_col=0) ## 用来映射ID和Virulence的
    dict_label = {str(dt_kmer_count_0["id"][i]):dt_kmer_count_0["Virulence"][i] for i in range(dt_kmer_count_0.shape[0])}
    dt_kmer_count = pd.read_csv("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase3/proteome_kmer_v214/df_3kmer.csv", sep=",", header=0)
    X = np.array(dt_kmer_count.iloc[:, 1:])  # Features
    y = np.array([dict_label[str(dt_kmer_count["Unnamed: 0"][i])] for i in range(dt_kmer_count.shape[0])])  # Target labels

    # 缺失值填充
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X = imputer.fit_transform(X)

    all_metrics = []

    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.2, random_state=76)
    for repeat_idx, (train_val_idx, test_idx) in enumerate(sss.split(X, y)):
        print(f"\n=== Repeat {repeat_idx + 1}/{repeats} ===", flush=True)
        
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=76)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
            print(f"Fold {fold_idx + 1}/{k}", flush=True)

            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            # 使用 RandomForest 训练
            rf = GaussianNB()
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_val)
            y_score = rf.predict_proba(X_val)[:, 1]

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auroc = roc_auc_score(y_val, y_score)
            precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_score)
            auprc = auc(recall_curve, precision_curve)

            metrics = {
                "repeat": repeat_idx + 1,
                "fold": fold_idx + 1,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "auroc": auroc,
                "auprc": auprc,
            }
            all_metrics.append(metrics)

    # 保存所有重复和折的结果
    df_all = pd.DataFrame(all_metrics)
    #df_all.to_csv("./tmp/cv_res_rf_vprot_3kmer.csv", index=False)

    # 平均结果
    mean_metrics = df_all.mean(numeric_only=True).to_dict()
    print("\n=== Average Test Metrics Across 5 Repeats ===")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f}")




def test_model_on_independent_set(repeats=5):
    # 加载数据
    dt_kmer_count_0 = pd.read_csv("../case_scripts_by_zhangzhiyuan/cases_res_files/xy_4kmer.csv", sep="\t", header=0, index_col=0) ## 用来映射ID和Virulence的
    dict_label = {str(dt_kmer_count_0["id"][i]):dt_kmer_count_0["Virulence"][i] for i in range(dt_kmer_count_0.shape[0])}
    dt_kmer_count = pd.read_csv("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase3/proteome_kmer_v214/df_3kmer.csv", sep=",", header=0)
    X = np.array(dt_kmer_count.iloc[:, 1:])  # Features
    y = np.array([dict_label[str(dt_kmer_count["Unnamed: 0"][i])] for i in range(dt_kmer_count.shape[0])])  # Target labels
    
    # 缺失值填充
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X = imputer.fit_transform(X)

    # 记录结果
    test_results = []

    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.2, random_state=76)
    for repeat_idx, (train_idx, test_idx) in enumerate(sss.split(X, y)):
        print(f"\n--- Testing Repeat {repeat_idx + 1}/{repeats} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 初始化并训练随机森林
        rf = GaussianNB()
        rf.fit(X_train, y_train)

        y_pred_label = rf.predict(X_test)
        y_pred_prob = rf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred_label)
        precision = precision_score(y_test, y_pred_label, zero_division=0)
        recall = recall_score(y_test, y_pred_label, zero_division=0)
        f1 = f1_score(y_test, y_pred_label, zero_division=0)
        auroc = roc_auc_score(y_test, y_pred_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob)
        auprc = auc(recall_curve, precision_curve)

        test_results.append({
            "repeat": repeat_idx + 1,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auroc": auroc,
            "auprc": auprc
        })
        print(f"[Repeat {repeat_idx + 1}] acc={acc:.4f}, prec={precision:.4f}, rec={recall:.4f}, f1={f1:.4f}, auroc={auroc:.4f}, auprc={auprc:.4f}")

    # 保存结果
    df_test = pd.DataFrame(test_results)
    #df_test.to_csv("./tmp/test_res_rf_vprot_3kmer.csv", index=False)

    # 平均结果
    mean_metrics = df_test.mean(numeric_only=True).to_dict()
    print("\n=== Average Test Metrics Across 5 Repeats ===")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f}")

    
    
if __name__ == "__main__":
    print("START: ", time.ctime(), flush=True)
    seed_torch()
    #train_model_repeat_cv_eval()
    test_model_on_independent_set()
    print("END: ", time.ctime(), flush=True)
