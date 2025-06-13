import os
import random
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import time


def seed_torch(seed=20):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def load_kmer_embedding(file_path):
    """ 读取单个 kmer 文件并转换为 NumPy 数组 """
    file_name = os.path.basename(file_path)
    sample_id = file_name[:-12]  # 移除 ".fasta.nkmer"
    embedding = pd.read_csv(file_path).values.tolist()[0]
    return sample_id, np.array(embedding)

def get_Kmerfeatures_fict():
    dict_kmer_embedding = {}

    vpath = "/data/150T/databases/help_zhangzhiyuan/vhPPIpredCasesDT/case3_pred_virulence_v214/vprot_kmer/"
    hpath = "/data/150T/databases/help_zhangzhiyuan/vhPPIpredCasesDT/case3_pred_virulence_v214/hprot_kmer/"

    # 获取所有病毒和人类的 kmer 文件路径
    vfiles = [os.path.join(vpath, vf) for vf in os.listdir(vpath) if vf.endswith(".fasta.1kmer")]
    hfiles = [os.path.join(hpath, hf) for hf in os.listdir(hpath) if hf.endswith(".fasta.1kmer")]

    # 使用多进程并行处理 kmer 文件
    with ProcessPoolExecutor(5) as executor:
        results = executor.map(load_kmer_embedding, vfiles + hfiles, chunksize=10)

    # 合并结果
    for sample_id, embedding in results:
        dict_kmer_embedding[sample_id] = embedding

    return dict_kmer_embedding

def get_PLMfeatures_fict():
    dict_plm_embedding = {}
    vpath, hpath = "/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedFengYang/virusProtPLM/",\
                    "/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedFengYang/humanProtPLM/"
    for vf in os.listdir(vpath):
        vid = vf[0:-3]  # remove ".pt"
        vembed = torch.load(vpath + vf, weights_only=True).tolist()
        dict_plm_embedding[vid] = vembed
    for hf in os.listdir(hpath):
        hid = hf[0:-3]
        hembed = torch.load(hpath + hf, weights_only=True).tolist()
        dict_plm_embedding[hid] = hembed
    return dict_plm_embedding

"""Model"""


def aggregate_features_by_virus(node_features, virus_to_index):
    aggregated_features = []
    # 对于每个病毒，聚合其所有蛋白特征
    for virus in virus_list:
        virus_features = node_features[virus_to_index[virus]]
        aggregated_features.append(virus_features.mean(dim=0))

    return torch.stack(aggregated_features)

class GCN_Model(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim1=512, hidden_dim2=256, output_dim=1):
        super(GCN_Model, self).__init__()

        # 图卷积层
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

        # 多层感知机（MLP）
        self.fc1 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x, edge_index):
        # GCN层
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout 防止过拟合
        x = self.conv2(x, edge_index)
        aggregated_features = aggregate_features_by_virus(x, dict_taxid4viruspro)
        # MLP层
        x = F.relu(self.fc1(aggregated_features))
        x = self.fc2(x)

        return x


# 交叉验证
def train_model_repeat_cv(repeats=5, k=5, device='cuda'):
    epochs = 100
    lr = 0.0005
    criterion = BCEWithLogitsLoss()

    all_logs = []

    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.2, random_state=76)
    for repeat_idx, (train_val_idx, test_idx) in enumerate(sss.split(virus_list, tables_list)):
        print(f"\n=== Repeat {repeat_idx + 1}/{repeats} ===")
        
        # 获取训练+验证集和测试集
        virus_train_val = [virus_list[i] for i in train_val_idx]
        virus_test = [virus_list[i] for i in test_idx]
        labels_train_val = [dict_taxid4table[v] for v in virus_train_val]
        labels_test = [dict_taxid4table[v] for v in virus_test]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=76)
        fold_logs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(virus_train_val, labels_train_val)):
            print(f"Training Fold {fold + 1}/{k}...", flush=True)
            print(f"Train:Val={len(train_idx)}/{len(val_idx)}", flush=True)

            # 获取训练和验证病毒 taxid
            train_virus = [virus_train_val[i] for i in train_idx]
            val_virus = [virus_train_val[i] for i in val_idx]

            # 标签
            train_labels = torch.tensor([dict_taxid4table[v] for v in train_virus], dtype=torch.float).view(-1, 1).to(device)
            val_labels = torch.tensor([dict_taxid4table[v] for v in val_virus], dtype=torch.float).view(-1, 1).to(device)

            # 模型和优化器
            model = GCN_Model(input_dim=node_features.shape[1], hidden_dim1=512, hidden_dim2=256, output_dim=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                out = model(node_features.to(device), edge_index.to(device))

                out_train = out[[virus_list.index(v) for v in train_virus]]
                loss_train = criterion(out_train, train_labels)
                loss_train.backward()
                optimizer.step()

                pred_train = torch.sigmoid(out_train).cpu().detach().numpy()
                train_acc = accuracy_score(train_labels.cpu(), pred_train.round())

                model.eval()
                with torch.no_grad():
                    out = model(node_features.to(device), edge_index.to(device))
                    out_val = out[[virus_list.index(v) for v in val_virus]]
                    loss_val = criterion(out_val, val_labels)
                    pred_val = torch.sigmoid(out_val).cpu().detach().numpy()
                    val_acc = accuracy_score(val_labels.cpu(), pred_val.round())

                    fold_logs.append({
                        "repeat": repeat_idx + 1,
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "train_loss": loss_train.item(),
                        "val_loss": loss_val.item(),
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                    })

        all_logs.extend(fold_logs)

    df_logs = pd.DataFrame(all_logs)
    df_logs.to_csv("./tmp/loss_mlp_ppi_gnn.csv", index=False)



def train_model_repeat_cv_eval(repeats=5, k=5, epochs=45, lr=0.0005, device='cuda'):
    criterion = BCEWithLogitsLoss()

    all_metrics = []

    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.2, random_state=76)
    for repeat_idx, (train_val_idx, test_idx) in enumerate(sss.split(virus_list, tables_list)):
        print(f"\n=== Repeat {repeat_idx + 1}/{repeats} ===")
        
        virus_train_val = [virus_list[i] for i in train_val_idx]
        labels_train_val = [dict_taxid4table[v] for v in virus_train_val]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=76)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(virus_train_val, labels_train_val)):
            print(f"Fold {fold_idx + 1}/{k}", flush=True)
            train_virus = [virus_train_val[i] for i in train_idx]
            val_virus = [virus_train_val[i] for i in val_idx]

            train_labels = torch.tensor([dict_taxid4table[v] for v in train_virus], dtype=torch.float).view(-1, 1).to(device)
            val_labels = torch.tensor([dict_taxid4table[v] for v in val_virus], dtype=torch.float).view(-1, 1).to(device)

            model = GCN_Model(input_dim=node_features.shape[1], hidden_dim1=512, hidden_dim2=256, output_dim=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                out = model(node_features.to(device), edge_index.to(device))
                out_train = out[[virus_list.index(v) for v in train_virus]]
                loss_train = criterion(out_train, train_labels)
                loss_train.backward()
                optimizer.step()

            # 评估阶段
            model.eval()
            with torch.no_grad():
                out = model(node_features.to(device), edge_index.to(device))
                out_val = out[[virus_list.index(v) for v in val_virus]]
                probs_val = torch.sigmoid(out_val).cpu().numpy().flatten()
                y_val = val_labels.cpu().numpy().flatten()
                y_pred = (probs_val >= 0.5).astype(int)

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

    # 保存详细交叉验证结果
    df_all = pd.DataFrame(all_metrics)
    df_all.to_csv("./tmp/cv_res_mlp_ppi_gnn.csv", index=False)
    
    # 平均结果
    mean_metrics = df_all.mean(numeric_only=True).to_dict()
    print("\n=== Average Test Metrics Across 5 Repeats ===")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f}")
    

def test_model_on_independent_set(repeats=5, epoch_num=45, lr=0.0005, device='cuda'):
    criterion = BCEWithLogitsLoss()
    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.2, random_state=76)
    test_metrics = []

    for repeat_idx, (train_val_idx, test_idx) in enumerate(sss.split(virus_list, tables_list)):
        print(f"\n=== Test Repeat {repeat_idx + 1}/{repeats} ===")

        virus_train_val = [virus_list[i] for i in train_val_idx]
        virus_test = [virus_list[i] for i in test_idx]
        labels_train_val = [dict_taxid4table[v] for v in virus_train_val]
        labels_test = [dict_taxid4table[v] for v in virus_test]

        # 训练集数据
        train_labels_tensor = torch.tensor(labels_train_val, dtype=torch.float).view(-1, 1).to(device)

        # 训练模型
        model = GCN_Model(input_dim=node_features.shape[1], hidden_dim1=512, hidden_dim2=256, output_dim=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epoch_num):
            model.train()
            optimizer.zero_grad()
            out = model(node_features.to(device), edge_index.to(device))
            out_train = out[[virus_list.index(v) for v in virus_train_val]]
            loss_train = criterion(out_train, train_labels_tensor)
            loss_train.backward()
            optimizer.step()

        # 测试
        model.eval()
        with torch.no_grad():
            out = model(node_features.to(device), edge_index.to(device))
            out_test = out[[virus_list.index(v) for v in virus_test]]
            test_labels_tensor = torch.tensor(labels_test, dtype=torch.float).view(-1, 1).to(device)

            pred_probs = torch.sigmoid(out_test).cpu().numpy().flatten()
            pred_labels = (pred_probs >= 0.5).astype(int)
            true_labels = np.array(labels_test)

            acc = accuracy_score(true_labels, pred_labels)
            prec = precision_score(true_labels, pred_labels, zero_division=0)
            rec = recall_score(true_labels, pred_labels, zero_division=0)
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            auroc = roc_auc_score(true_labels, pred_probs)
            precision_curve, recall_curve, _ = precision_recall_curve(true_labels, pred_probs)
            auprc = auc(recall_curve, precision_curve)
            #auprc = average_precision_score(true_labels, pred_probs)

            test_metrics.append({
                "repeat": repeat_idx + 1,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "auroc": auroc,
                "auprc": auprc
            })

            print(f"[Repeat {repeat_idx + 1}] acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, auroc={auroc:.4f}, auprc={auprc:.4f}")

    # 保存结果
    df_test = pd.DataFrame(test_metrics)
    df_test.to_csv("./tmp/test_res_mlp_ppi_gnn.csv", index=False)

    # 平均结果
    mean_metrics = df_test.mean(numeric_only=True).to_dict()
    print("\n=== Average Test Metrics Across 5 Repeats ===")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f}")
    

    
if __name__ == "__main__":
    print("START: ", time.ctime(), flush=True)
    
    seed_torch()

    df = pd.read_csv("/data/150T/databases/help_zhangzhiyuan/vhPPIpredCasesDT/case3_pred_virulence_v214/pred_ppi_norm_score_v214.txt", sep="\t")
    df = df.sort_values(by="vtaxid")
    
    # 节点特征 边 标签 # data = Data(x=node_features, edge_index=edge_index, y=label)
    # 得到 nodes 列表
    node_viruspro = list(set(df["virus_unid"]))  # 927
    node_humanpro = list(set(df["human_unid"]))  # 6110
    nodes = node_viruspro + node_humanpro
    nodes.sort()  # 规定了节点的顺序

    dict_plm_embedding = get_PLMfeatures_fict()
    node_features = torch.tensor([dict_plm_embedding[v] for v in nodes], dtype=torch.float)

    # 得到 edge_index 列表
    vhid_to_index = {v: i for i, v in enumerate(nodes)}
    edge_index = torch.tensor([
        [vhid_to_index[df["virus_unid"][i]] for i in range(df.shape[0])],
        [vhid_to_index[df["human_unid"][i]] for i in range(df.shape[0])]
    ], dtype=torch.long)

    # 得到 label 列表
    prelabel = "label"
    df_grouped = df.groupby("vtaxid")
    dict_taxid4viruspro = {}  # 用于得到病毒的嵌入
    dict_taxid4table = {}
    for taxid, gp in df_grouped:
        gp.reset_index(drop=True, inplace=True)
        dict_taxid4table[taxid] = gp[prelabel][0]
        dict_taxid4viruspro[taxid] = [nodes.index(i) for i in list(set(gp["virus_unid"]))]
    virus_list = []  # 后期计算损失时， 按照此顺序取出相应的嵌入进行概率的计算。
    tables_list = []
    for taxid, table in dict_taxid4table.items():
        virus_list.append(taxid)
    virus_list = sorted(virus_list)
    for virus in virus_list:
        tables_list.append(dict_taxid4table[virus])
    

    # 运行交叉验证
    #train_model_repeat_cv()
    train_model_repeat_cv_eval()
    #test_model_on_independent_set()

    print("END: ", time.ctime(), flush=True)