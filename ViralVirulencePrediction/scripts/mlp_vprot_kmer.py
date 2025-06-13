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


def seed_torch(seed=20):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def train_model_repeat_cv(repeats = 5, k = 5):
    # 加载数据
    dt_kmer_count_0 = pd.read_csv("../case_scripts_by_zhangzhiyuan/cases_res_files/xy_4kmer.csv", sep="\t", header=0, index_col=0) ## 用来映射ID和Virulence的
    dict_label = {str(dt_kmer_count_0["id"][i]):dt_kmer_count_0["Virulence"][i] for i in range(dt_kmer_count_0.shape[0])}
    dt_kmer_count = pd.read_csv("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase3/proteome_kmer_v214/df_2kmer.csv", sep=",", header=0)
    X = np.array(dt_kmer_count.iloc[:, 1:])  # Features
    y = np.array([dict_label[str(dt_kmer_count["Unnamed: 0"][i])] for i in range(dt_kmer_count.shape[0])])  # Target labels
        
    # 缺失值填充
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X = imputer.fit_transform(X)
    
    # 模型参数
    num_epochs = 100
    learning_rate = 0.01
    input_dim = X.shape[1]
    hidden_dim = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型定义
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.model(x)
    
    # 日志存储
    all_logs = []
    
    # 外层分层采样重复划分训练/测试
    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.2, random_state=76)
    for repeat_idx, (train_val_idx, test_idx) in enumerate(sss.split(X, y)):
        print(f"\n=== Repeat {repeat_idx + 1}/{repeats} ===", flush=True)
        
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
    
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=76)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
            print(f"Fold {fold + 1}/{k}", flush=True)
            print(f"Train:Val={len(train_idx)}/{len(val_idx)}", flush=True)
    
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
    
            # 转换为tensor
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    
            model = MLP().to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
    
                with torch.no_grad():
                    model.eval()
                    train_pred = model(X_train_tensor)
                    val_pred = model(X_val_tensor)
    
                    train_loss = criterion(train_pred, y_train_tensor).item()
                    val_loss = criterion(val_pred, y_val_tensor).item()
    
                    train_acc = accuracy_score(y_train, (train_pred.cpu().numpy() >= 0.5).astype(int))
                    val_acc = accuracy_score(y_val, (val_pred.cpu().numpy() >= 0.5).astype(int))
    
                    all_logs.append({
                        "repeat": repeat_idx + 1,
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc
                    })
    
    # 保存所有日志
    df_logs = pd.DataFrame(all_logs)
    df_logs.to_csv("./tmp/loss_mlp_vprot_2kmer_new.csv", index=False)


def train_model_repeat_cv_eval(repeats=5, k=5, num_epochs = 40, learning_rate = 0.01):
    # 加载数据
    dt_kmer_count_0 = pd.read_csv("../case_scripts_by_zhangzhiyuan/cases_res_files/xy_4kmer.csv", sep="\t", header=0, index_col=0) ## 用来映射ID和Virulence的
    dict_label = {str(dt_kmer_count_0["id"][i]):dt_kmer_count_0["Virulence"][i] for i in range(dt_kmer_count_0.shape[0])}
    dt_kmer_count = pd.read_csv("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase3/proteome_kmer_v214/df_2kmer.csv", sep=",", header=0)
    X = np.array(dt_kmer_count.iloc[:, 1:])  # Features
    y = np.array([dict_label[str(dt_kmer_count["Unnamed: 0"][i])] for i in range(dt_kmer_count.shape[0])])  # Target labels
    
    # 缺失值填充
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X = imputer.fit_transform(X)

    # 训练参数
    input_dim = X.shape[1]
    hidden_dim = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型定义
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.model(x)

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

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

            model = MLP().to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

            # 最后 epoch 完成后评估验证集
            model.eval()
            with torch.no_grad():            
                probs_val = model(X_val_tensor).cpu().numpy().flatten()
                y_pred = (probs_val >= 0.5).astype(int)
                y_val = y_val_tensor.cpu().numpy().flatten()

                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred, zero_division=0)
                rec = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                auroc = roc_auc_score(y_val, probs_val)
                precision_curve, recall_curve, _ = precision_recall_curve(y_val, probs_val)
                auprc = auc(recall_curve, precision_curve)
                # 直接在此处计算各项指标
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
    #df_all.to_csv("./tmp/cv_res_mlp_vprot_3kmer.csv", index=False)

    # 平均结果
    mean_metrics = df_all.mean(numeric_only=True).to_dict()
    print("\n=== Average Test Metrics Across 5 Repeats ===")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f}")




def test_model_on_independent_set(repeats = 5, num_epochs = 70, learning_rate = 0.001):
    # 加载数据
    dt_kmer_count_0 = pd.read_csv("../case_scripts_by_zhangzhiyuan/cases_res_files/xy_4kmer.csv", sep="\t", header=0, index_col=0) ## 用来映射ID和Virulence的
    dict_label = {str(dt_kmer_count_0["id"][i]):dt_kmer_count_0["Virulence"][i] for i in range(dt_kmer_count_0.shape[0])}
    dt_kmer_count = pd.read_csv("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase3/proteome_kmer_v214/df_3kmer.csv", sep=",", header=0)
    X = np.array(dt_kmer_count.iloc[:, 1:])  # Features
    y = np.array([dict_label[str(dt_kmer_count["Unnamed: 0"][i])] for i in range(dt_kmer_count.shape[0])])  # Target labels
    
    # 缺失值填充
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X = imputer.fit_transform(X)
    
    # 参数设置
    input_dim = X.shape[1]
    hidden_dim = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型定义
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.model(x)
    
    # 记录结果
    test_results = []
    
    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.2, random_state=76)
    for repeat_idx, (train_idx, test_idx) in enumerate(sss.split(X, y)):
        print(f"\n--- Testing Repeat {repeat_idx + 1}/{repeats} ---")
    
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
        # Tensor 转换
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    
        model = MLP().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
        # 训练
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
    
        # 测试
        with torch.no_grad():
            model.eval()
            test_pred = model(X_test_tensor).cpu().numpy().flatten()
            test_pred_label = (test_pred >= 0.5).astype(int)
    
            acc = accuracy_score(y_test, test_pred_label)
            precision = precision_score(y_test, test_pred_label, zero_division=0)
            recall = recall_score(y_test, test_pred_label, zero_division=0)
            f1 = f1_score(y_test, test_pred_label, zero_division=0)
            auroc = roc_auc_score(y_test, test_pred)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_pred)
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
    #df_test.to_csv("./tmp/test_res_mlp_vprot_3kmer.csv", index=False)

    # 平均结果
    mean_metrics = df_test.mean(numeric_only=True).to_dict()
    print("\n=== Average Test Metrics Across 5 Repeats ===")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    
if __name__ == "__main__":
    print("START: ", time.ctime(), flush=True)
    seed_torch()
    #train_model_repeat_cv()
    #train_model_repeat_cv_eval()
    test_model_on_independent_set()
    print("END: ", time.ctime(), flush=True)
