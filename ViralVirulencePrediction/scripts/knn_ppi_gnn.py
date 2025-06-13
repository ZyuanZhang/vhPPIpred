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
from sklearn.neighbors import KNeighborsClassifier


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
    virus_feature_dict = {}
    for virus in virus_list:
        indices = virus_to_index[virus]  # 是一个 list of protein indices
        virus_feature = node_features[indices].mean(dim=0)
        virus_feature_dict[virus] = virus_feature
    return virus_feature_dict

# GCN模型
class GCN_Model(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim1=512, hidden_dim2=256):
        super(GCN_Model, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # 输出每个蛋白的256维特征

# 主训练函数
def train_model_repeat_cv_eval(repeats=5, k=5, epochs=45, lr=0.0005, device='cuda'):
    all_metrics = []

    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.2, random_state=76)
    for repeat_idx, (train_val_idx, test_idx) in enumerate(sss.split(virus_list, tables_list)):
        print(f"\n=== Repeat {repeat_idx + 1}/{repeats} ===")

        virus_train_val = [virus_list[i] for i in train_val_idx]
        labels_train_val = [dict_taxid4table[v] for v in virus_train_val]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=76)
        for fold, (train_idx, val_idx) in enumerate(skf.split(virus_train_val, labels_train_val)):
            print(f"Fold {fold + 1}/{k}")

            train_virus = [virus_train_val[i] for i in train_idx]
            val_virus = [virus_train_val[i] for i in val_idx]
            train_labels = [dict_taxid4table[v] for v in train_virus]
            val_labels = [dict_taxid4table[v] for v in val_virus]

            model = GCN_Model(input_dim=node_features.shape[1], hidden_dim1=512, hidden_dim2=256).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                out = model(node_features.to(device), edge_index.to(device))
                virus_feat_dict = aggregate_features_by_virus(out, dict_taxid4viruspro)
                out_train = torch.stack([virus_feat_dict[v] for v in train_virus])
                labels_tensor = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1).to(device)
                loss = criterion(torch.mean(out_train, dim=1).unsqueeze(1), labels_tensor)
                loss.backward()
                optimizer.step()

            # 提取所有病毒特征（使用训练好的模型）
            model.eval()
            with torch.no_grad():
                all_node_embeddings = model(node_features.to(device), edge_index.to(device))
                virus_embeddings = aggregate_features_by_virus(all_node_embeddings, dict_taxid4viruspro)

            # 构造RF训练集和验证集
            X_train = np.array([virus_embeddings[v].cpu().numpy() for v in train_virus])
            y_train = np.array(train_labels)
            X_val = np.array([virus_embeddings[v].cpu().numpy() for v in val_virus])
            y_val = np.array(val_labels)

            rf = KNeighborsClassifier(n_neighbors=7)
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
            # 直接在此处计算各项指标
            metrics = {
                "repeat": repeat_idx + 1,
                "fold": fold + 1,
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
    #df_all.to_csv("./tmp/cv_res_rf_ppi_gnn.csv", index=False)
    
    # 平均结果
    mean_metrics = df_all.mean(numeric_only=True).to_dict()
    print("\n=== Average Test Metrics Across 5 Repeats ===")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f}")



def test_model_on_independent_set(repeats=5, epochs=45, lr=0.0005, device='cuda'):
    all_metrics = []

    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.2, random_state=76)
    for repeat_idx, (train_val_idx, test_idx) in enumerate(sss.split(virus_list, tables_list)):
        print(f"\n=== Repeat {repeat_idx + 1}/{repeats} ===")

        # 获取训练+验证集 和 测试集
        virus_train_val = [virus_list[i] for i in train_val_idx]
        virus_test = [virus_list[i] for i in test_idx]
        labels_train_val = [dict_taxid4table[v] for v in virus_train_val]
        labels_test = [dict_taxid4table[v] for v in virus_test]

        # 初始化GCN模型
        model = GCN_Model(input_dim=node_features.shape[1], hidden_dim1=512, hidden_dim2=256).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # 训练GCN
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(node_features.to(device), edge_index.to(device))
            virus_feat_dict = aggregate_features_by_virus(out, dict_taxid4viruspro)
            out_train = torch.stack([virus_feat_dict[v] for v in virus_train_val])
            labels_tensor = torch.tensor(labels_train_val, dtype=torch.float32).view(-1, 1).to(device)
            loss = criterion(torch.mean(out_train, dim=1).unsqueeze(1), labels_tensor)
            loss.backward()
            optimizer.step()

        # 提取特征
        model.eval()
        with torch.no_grad():
            all_node_embeddings = model(node_features.to(device), edge_index.to(device))
            virus_embeddings = aggregate_features_by_virus(all_node_embeddings, dict_taxid4viruspro)

        # 构建RF训练和测试集
        X_train = np.array([virus_embeddings[v].cpu().numpy() for v in virus_train_val])
        y_train = np.array(labels_train_val)
        X_test = np.array([virus_embeddings[v].cpu().numpy() for v in virus_test])
        y_test = np.array(labels_test)
        
        rf = KNeighborsClassifier(n_neighbors=7)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_score = rf.predict_proba(X_test)[:, 1]

        # 评估指标
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auroc = roc_auc_score(y_test, y_score)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
        auprc = auc(recall_curve, precision_curve)

        metrics = {
            "repeat": repeat_idx + 1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auroc": auroc,
            "auprc": auprc,
        }

        all_metrics.append(metrics)

    # 保存每次重复的测试结果
    df_all = pd.DataFrame(all_metrics)
    #df_all.to_csv("./tmp/test_res_rf_ppi_gnn.csv", index=False)

    # 平均结果
    mean_metrics = df_all.mean(numeric_only=True).to_dict()
    print("\n=== Average Independent Test Metrics Over 5 Repeats ===")
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
    #train_model_repeat_cv_eval()
    test_model_on_independent_set()

    print("END: ", time.ctime(), flush=True)