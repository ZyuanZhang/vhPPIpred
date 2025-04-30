import torch
import os
import pandas as pd
import numpy as np

def loadSeqAndPssmembed(dataframe_all, FeatPath):
    dict_embedding = {}
    vf_list = [FeatPath+"virusProtPLM/"+f for f in os.listdir(FeatPath+"virusProtPLM/") if f.endswith(".pt")]
    for vf in vf_list:
        tmp_emb = torch.load(vf)
        tmp_emb = tmp_emb.unsqueeze(0)
        dict_embedding[vf.split("/")[-1][0:-3]] = tmp_emb

    hf_list = [FeatPath+"humanProtPLM/"+f for f in os.listdir(FeatPath+"humanProtPLM/") if f.endswith(".pt")]
    for hf in hf_list:
        tmp_emb = torch.load(hf)
        tmp_emb = tmp_emb.unsqueeze(0)
        dict_embedding[hf.split("/")[-1][0:-3]] = tmp_emb

    dict_embedding_pssm = {}
    vf_list_pssm = [FeatPath+"virusProtPSSMembed/"+f for f in os.listdir(FeatPath+"virusProtPSSMembed/") if f.endswith(".npy")]
    for vf in vf_list_pssm:
        tmp_emb = np.load(vf)
        dict_embedding_pssm[vf.split("/")[-1][0:-4]] = tmp_emb

    hf_list_pssm = [FeatPath+"humanProtPSSMembed/"+f for f in os.listdir(FeatPath+"humanProtPSSMembed/") if f.endswith(".npy")]
    for hf in hf_list_pssm:
        tmp_emb = np.load(hf)
        dict_embedding_pssm[hf.split("/")[-1][0:-4]] = tmp_emb

    feat_seqembed_all, feat_pssmembed_all = [], []
    for j in range(dataframe_all.shape[0]):
        feat_seqembed_temp, feat_pssmembed_temp = [], []
        
        tmp_feat_seq_v = dict_embedding[dataframe_all["virus_unid"][j]]
        tmp_feat_seq_h = dict_embedding[dataframe_all["human_unid"][j]]
        temp_feat_seq = torch.cat((tmp_feat_seq_v, tmp_feat_seq_h), dim=1)
        temp_feat_seq = temp_feat_seq.squeeze(0)
        temp_feat_seq = temp_feat_seq.tolist()
        feat_seqembed_temp += temp_feat_seq

        tmp_feat_pssm_v = dict_embedding_pssm[dataframe_all["virus_unid"][j]]
        tmp_feat_pssm_h = dict_embedding_pssm[dataframe_all["human_unid"][j]]
        temp_feat_pssm = np.concatenate((tmp_feat_pssm_v, tmp_feat_pssm_h), axis=1)
        temp_feat_pssm = temp_feat_pssm.squeeze(0)
        temp_feat_pssm = temp_feat_pssm.tolist()
        feat_pssmembed_temp += temp_feat_pssm
        
        feat_seqembed_all.append(feat_seqembed_temp)
        feat_pssmembed_all.append(feat_pssmembed_temp)

    feat_seqembed_all = np.array(feat_seqembed_all)
    feat_pssmembed_all = np.array(feat_pssmembed_all) 

    return feat_seqembed_all, feat_pssmembed_all


def loadfeatures(dataframe_train, dataframe_test, FeatPath):
    dict_embedding = {}
    vf_list = [FeatPath+"virusProtPLM/"+f for f in os.listdir(FeatPath+"virusProtPLM/") if f.endswith(".pt")]
    for vf in vf_list:
        tmp_emb = torch.load(vf)
        tmp_emb = tmp_emb.unsqueeze(0)
        dict_embedding[vf.split("/")[-1][0:-3]] = tmp_emb

    hf_list = [FeatPath+"humanProtPLM/"+f for f in os.listdir(FeatPath+"humanProtPLM/") if f.endswith(".pt")]
    for hf in hf_list:
        tmp_emb = torch.load(hf)
        tmp_emb = tmp_emb.unsqueeze(0)
        dict_embedding[hf.split("/")[-1][0:-3]] = tmp_emb

    dt_hp_degree = pd.read_csv(FeatPath+"VirusHumanSimiAndDegree/node_feat_human.tsv", sep="\t", header=0)
    dict_hp_degree = {dt_hp_degree["#"][i]:dt_hp_degree["degree"][i] for i in range(dt_hp_degree.shape[0])} 

    dt_simi = pd.read_csv(FeatPath+"VirusHumanSimiAndDegree/script/virus_human_seqsimi.txt", sep="\t", header=0)
    dict_simi = {dt_simi["virus_unid"][i]+"-"+dt_simi["human_unid"][i]:[dt_simi["vp_np_seqsimi"][i], dt_simi["np_hp_score"][i]] for i in range(dt_simi.shape[0])}

    dict_embedding_pssm = {}
    vf_list_pssm = [FeatPath+"virusProtPSSMembed/"+f for f in os.listdir(FeatPath+"virusProtPSSMembed/") if f.endswith(".npy")]
    for vf in vf_list_pssm:
        tmp_emb = np.load(vf)
        dict_embedding_pssm[vf.split("/")[-1][0:-4]] = tmp_emb

    hf_list_pssm = [FeatPath+"humanProtPSSMembed/"+f for f in os.listdir(FeatPath+"humanProtPSSMembed/") if f.endswith(".npy")]
    for hf in hf_list_pssm:
        tmp_emb = np.load(hf)
        dict_embedding_pssm[hf.split("/")[-1][0:-4]] = tmp_emb

    feat_train, feat_test = [], []
    for j in range(dataframe_train.shape[0]):
        feat_train_temp = []
        tmp_feat_v = dict_embedding[dataframe_train["virus_unid"][j]]
        tmp_feat_h = dict_embedding[dataframe_train["human_unid"][j]]
        temp_feat = torch.cat((tmp_feat_v, tmp_feat_h), dim=1)
        temp_feat = temp_feat.squeeze(0)
        temp_feat = temp_feat.tolist()
        feat_train_temp += temp_feat

        try:
            feat_train_temp.append(dict_hp_degree[dataframe_train["human_unid"][j]])
        except:
            feat_train_temp.append(0.0)
        try:
            feat_train_temp += dict_simi[dataframe_train["virus_unid"][j]+"-"+dataframe_train["human_unid"][j]]
        except:
            feat_train_temp += [0.0, 0.0]

        tmp_feat_pssm_v = dict_embedding_pssm[dataframe_train["virus_unid"][j]]
        tmp_feat_pssm_h = dict_embedding_pssm[dataframe_train["human_unid"][j]]
        temp_feat_pssm = np.concatenate((tmp_feat_pssm_v, tmp_feat_pssm_h), axis=1)
        temp_feat_pssm = temp_feat_pssm.squeeze(0)
        temp_feat_pssm = temp_feat_pssm.tolist()
        feat_train_temp += temp_feat_pssm

        feat_train_temp.append(dataframe_train["label"][j])
        feat_train.append(feat_train_temp)

    for k in range(dataframe_test.shape[0]):
        feat_test_temp = []

        tmp_feat_v = dict_embedding[dataframe_test["virus_unid"][k]]
        tmp_feat_h = dict_embedding[dataframe_test["human_unid"][k]]
        temp_feat = torch.cat((tmp_feat_v, tmp_feat_h), dim=1)
        temp_feat = temp_feat.squeeze(0)
        temp_feat = temp_feat.tolist()
        feat_test_temp += temp_feat

        try:
            feat_test_temp.append(dict_hp_degree[dataframe_test["human_unid"][k]])
        except:
            feat_test_temp.append(0.0)
        try:
            feat_test_temp += dict_simi[dataframe_test["virus_unid"][k]+"-"+dataframe_test["human_unid"][k]]
        except:
            feat_test_temp += [0.0, 0.0]

        tmp_feat_pssm_v = dict_embedding_pssm[dataframe_test["virus_unid"][k]]
        tmp_feat_pssm_h = dict_embedding_pssm[dataframe_test["human_unid"][k]]
        temp_feat_pssm = np.concatenate((tmp_feat_pssm_v, tmp_feat_pssm_h), axis=1)
        temp_feat_pssm = temp_feat_pssm.squeeze(0)
        temp_feat_pssm = temp_feat_pssm.tolist()
        feat_test_temp += temp_feat_pssm

        feat_test_temp.append(dataframe_test["label"][k])
        feat_test.append(feat_test_temp)

    feat_train = np.array(feat_train) ## seqembed (1x2048), hpdegree (1x1), simi (1x2), pssmembed(1x40)
    feat_test = np.array(feat_test) 

    return feat_train, feat_test




def loadfeaturesForBanchmarkDataset(dataframe_train, dataframe_test, FeatPath_train, FeatPath_test, testset_name):
    dict_embedding = {}
    vf_list = [FeatPath_train+"virusProtPLM/"+f for f in os.listdir(FeatPath_train+"virusProtPLM/") if f.endswith(".pt")] + [FeatPath_test+"virusProtPLM/"+f for f in os.listdir(FeatPath_test+"virusProtPLM/") if f.endswith(".pt")]
    for vf in vf_list:
        tmp_emb = torch.load(vf)
        tmp_emb = tmp_emb.unsqueeze(0)
        dict_embedding[vf.split("/")[-1][0:-3]] = tmp_emb

    hf_list = [FeatPath_train+"humanProtPLM/"+f for f in os.listdir(FeatPath_train+"humanProtPLM/") if f.endswith(".pt")] + [FeatPath_test+"humanProtPLM/"+f for f in os.listdir(FeatPath_test+"humanProtPLM/") if f.endswith(".pt")]
    for hf in hf_list:
        tmp_emb = torch.load(hf)
        tmp_emb = tmp_emb.unsqueeze(0)
        dict_embedding[hf.split("/")[-1][0:-3]] = tmp_emb

    dt_hp_degree = pd.read_csv(FeatPath_train+"VirusHumanSimiAndDegree/node_feat_human.tsv", sep="\t", header=0)
    dict_hp_degree = {dt_hp_degree["#"][i]:dt_hp_degree["degree"][i] for i in range(dt_hp_degree.shape[0])} 

    dt_simi_train = pd.read_csv(FeatPath_train+"VirusHumanSimiAndDegree/script/virus_human_seqsimi.txt", sep="\t", header=0)
    if testset_name == "barman":
        dt_simi_test = pd.read_csv(FeatPath_test+"VirusHumanSimiAndDegree/virus_human_seqsimi_barman_dataset.txt", sep="\t", header=0)
    elif testset_name == "yangdeepviral":
        dt_simi_test = pd.read_csv(FeatPath_test+"VirusHumanSimiAndDegree/virus_human_seqsimi_yangdeepviral_dataset.txt", sep="\t", header=0)
    elif testset_name == "denovo":
        dt_simi_test = pd.read_csv(FeatPath_test+"VirusHumanSimiAndDegree/virus_human_seqsimi_denovo_dataset.txt", sep="\t", header=0)
    elif testset_name == "zhou":
        dt_simi_test = pd.read_csv(FeatPath_test+"VirusHumanSimiAndDegree/virus_human_seqsimi_zhou_dataset.txt", sep="\t", header=0)
    elif testset_name == "rbprecep":
        dt_simi_test = pd.read_csv(FeatPath_test+"VirusHumanSimiAndDegree/virus_human_seqsimi_rbprecep_dataset.txt", sep="\t", header=0)
    elif testset_name == "rbphcm":
        dt_simi_test = pd.read_csv(FeatPath_test+"VirusHumanSimiAndDegree/virus_human_seqsimi_rbphcm_dataset.txt", sep="\t", header=0)
    
    dict_simi = {}
    for i in range(dt_simi_train.shape[0]):
        dict_simi[dt_simi_train["virus_unid"][i]+"-"+dt_simi_train["human_unid"][i]] = [dt_simi_train["vp_np_seqsimi"][i], dt_simi_train["np_hp_score"][i]]
    for j in range(dt_simi_test.shape[0]):
        dict_simi[dt_simi_test["virus_unid"][j]+"-"+dt_simi_test["human_unid"][j]] = [dt_simi_test["vp_np_seqsimi"][j], dt_simi_test["np_hp_score"][j]]

    dict_embedding_pssm = {}
    vf_list_pssm = [FeatPath_train+"virusProtPSSMembed/"+f for f in os.listdir(FeatPath_train+"virusProtPSSMembed/") if f.endswith(".npy")] + [FeatPath_test+"virusProtPSSMembed/"+f for f in os.listdir(FeatPath_test+"virusProtPSSMembed/") if f.endswith(".npy")]
    for vf in vf_list_pssm:
        tmp_emb = np.load(vf)
        dict_embedding_pssm[vf.split("/")[-1][0:-4]] = tmp_emb

    hf_list_pssm = [FeatPath_train+"humanProtPSSMembed/"+f for f in os.listdir(FeatPath_train+"humanProtPSSMembed/") if f.endswith(".npy")] + [FeatPath_test+"humanProtPSSMembed/"+f for f in os.listdir(FeatPath_test+"humanProtPSSMembed/") if f.endswith(".npy")]
    for hf in hf_list_pssm:
        tmp_emb = np.load(hf)
        dict_embedding_pssm[hf.split("/")[-1][0:-4]] = tmp_emb

    feat_train, feat_test = [], []
    for j in range(dataframe_train.shape[0]):
        feat_train_temp = []
        tmp_feat_v = dict_embedding[dataframe_train["virus_unid"][j]]
        tmp_feat_h = dict_embedding[dataframe_train["human_unid"][j]]
        temp_feat = torch.cat((tmp_feat_v, tmp_feat_h), dim=1)
        temp_feat = temp_feat.squeeze(0)
        temp_feat = temp_feat.tolist()
        feat_train_temp += temp_feat

        try:
            feat_train_temp.append(dict_hp_degree[dataframe_train["human_unid"][j]])
        except:
            feat_train_temp.append(0.0)
        try:
            feat_train_temp += dict_simi[dataframe_train["virus_unid"][j]+"-"+dataframe_train["human_unid"][j]]
        except:
            feat_train_temp += [0.0, 0.0]

        tmp_feat_pssm_v = dict_embedding_pssm[dataframe_train["virus_unid"][j]]
        tmp_feat_pssm_h = dict_embedding_pssm[dataframe_train["human_unid"][j]]
        temp_feat_pssm = np.concatenate((tmp_feat_pssm_v, tmp_feat_pssm_h), axis=1)
        temp_feat_pssm = temp_feat_pssm.squeeze(0)
        temp_feat_pssm = temp_feat_pssm.tolist()
        feat_train_temp += temp_feat_pssm

        feat_train_temp.append(dataframe_train["label"][j])
        feat_train.append(feat_train_temp)

    for k in range(dataframe_test.shape[0]):
        feat_test_temp = []

        tmp_feat_v = dict_embedding[dataframe_test["virus_unid"][k]]
        tmp_feat_h = dict_embedding[dataframe_test["human_unid"][k]]
        temp_feat = torch.cat((tmp_feat_v, tmp_feat_h), dim=1)
        temp_feat = temp_feat.squeeze(0)
        temp_feat = temp_feat.tolist()
        feat_test_temp += temp_feat

        try:
            feat_test_temp.append(dict_hp_degree[dataframe_test["human_unid"][k]])
        except:
            feat_test_temp.append(0.0)
        try:
            feat_test_temp += dict_simi[dataframe_test["virus_unid"][k]+"-"+dataframe_test["human_unid"][k]]
        except:
            feat_test_temp += [0.0, 0.0]

        tmp_feat_pssm_v = dict_embedding_pssm[dataframe_test["virus_unid"][k]]
        tmp_feat_pssm_h = dict_embedding_pssm[dataframe_test["human_unid"][k]]
        temp_feat_pssm = np.concatenate((tmp_feat_pssm_v, tmp_feat_pssm_h), axis=1)
        temp_feat_pssm = temp_feat_pssm.squeeze(0)
        temp_feat_pssm = temp_feat_pssm.tolist()
        feat_test_temp += temp_feat_pssm

        feat_test_temp.append(dataframe_test["label"][k])
        feat_test.append(feat_test_temp)

    feat_train = np.array(feat_train) ## seqembed (1x2048), hpdegree (1x1), simi (1x2), pssmembed(1x40)
    feat_test = np.array(feat_test) 

    return feat_train, feat_test



