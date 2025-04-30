## apply vhPPIpred to predicting virus-human PPIs

## Take demo_dataset as an example

import numpy as np
from xgboost import XGBClassifier
import pandas as pd
import time
import torch
import pickle as pk
import warnings
warnings.filterwarnings("ignore")


def feat_split_pred(np_feat):
    np_feat_seq = np_feat[:, :2048]
    np_feat_hpdegree = np_feat[:, 2048]
    np_feat_hpdegree = np_feat_hpdegree[:, np.newaxis]
    np_feat_simi = np_feat[:, 2049:2051]
    np_feat_pssm = np_feat[:, 2051:2091] 
    return np_feat_seq, np_feat_hpdegree, np_feat_simi, np_feat_pssm



def run_predictor(infile, outfile, feature_path):
    ## ------------- load pre-trained model (including PCA and xgbclassifier) ---------------- ##
    PCA_SEQ = pk.load(open("./model/PCA_SEQ.pkl", "rb"))
    PCA_PSSM = pk.load(open("./model/PCA_PSSM.pkl", "rb"))
    xgbclf = XGBClassifier()
    xgbclf.load_model("./model/xgbclf.json")

    ## ------------- infile and outfile of ppi predicted --------------- ##
    dt_infile = pd.read_csv(infile, sep="\t", header=0)

    dict_seq_embedding_pred_virus, dict_seq_embedding_pred_human, dict_pssm_embedding_pred_virus, dict_pssm_embedding_pred_human = {}, {}, {}, {}
    for i in range(dt_infile.shape[0]):
        vid, hid = dt_infile["virus_unid"][i], dt_infile["human_unid"][i]  
        ## ProtT5 embedding         
        tmp_emb = torch.load(feature_path+"virusProtPLM/"+vid+".pt")
        dict_seq_embedding_pred_virus[vid] = tmp_emb.unsqueeze(0)

        tmp_emb = torch.load(feature_path+"humanProtPLM/"+hid+".pt")
        dict_seq_embedding_pred_human[hid] = tmp_emb.unsqueeze(0)

        ## PSSM embedding
        tmp_emb = np.load(feature_path+"virusProtPSSMembed/"+vid+".npy")
        dict_pssm_embedding_pred_virus[vid] = tmp_emb

        tmp_emb = np.load(feature_path+"humanProtPSSMembed/"+hid+".npy")
        dict_pssm_embedding_pred_human[hid] = tmp_emb
    
    ## human protein degree
    dt_hp_degree = pd.read_csv(feature_path+"VirusHumanSimiAndDegree/node_feat_human.tsv", sep="\t", header=0)
    dict_hp_degree = {dt_hp_degree["#"][i]:dt_hp_degree["degree"][i] for i in range(dt_hp_degree.shape[0])}
    
    ## viral mimicry of human protein interactions
    dt_simi_pred = pd.read_csv(feature_path+"VirusHumanSimiAndDegree/dt_simi.txt", sep="\t", header=0)
    dict_simi_pred = {}
    for i in range(dt_simi_pred.shape[0]):
        dict_simi_pred[dt_simi_pred["virus_unid"][i]+"-"+dt_simi_pred["human_unid"][i]] = [dt_simi_pred["vp_np_seqsimi"][i], dt_simi_pred["np_hp_score"][i]]    

        
    dataframe_pred = dt_infile
    feat_pred = []
    for k in range(dataframe_pred.shape[0]):
        feat_pred_temp = []
        tmp_feat_v = dict_seq_embedding_pred_virus[dataframe_pred["virus_unid"][k]]
        tmp_feat_h = dict_seq_embedding_pred_human[dataframe_pred["human_unid"][k]]
        temp_feat = torch.cat((tmp_feat_v, tmp_feat_h), dim=1)
        temp_feat = temp_feat.squeeze(0)
        temp_feat = temp_feat.tolist()
        feat_pred_temp += temp_feat

        try:
            feat_pred_temp.append(dict_hp_degree[dataframe_pred["human_unid"][k]])
        except:
            feat_pred_temp.append(0.0)
        try:
            feat_pred_temp += dict_simi_pred[dataframe_pred["virus_unid"][k]+"-"+dataframe_pred["human_unid"][k]]
        except:
            feat_pred_temp += [0.0, 0.0]

        tmp_feat_pssm_v = dict_pssm_embedding_pred_virus[dataframe_pred["virus_unid"][k]]
        tmp_feat_pssm_h = dict_pssm_embedding_pred_human[dataframe_pred["human_unid"][k]]
        temp_feat_pssm = np.concatenate((tmp_feat_pssm_v, tmp_feat_pssm_h), axis=1)
        temp_feat_pssm = temp_feat_pssm.squeeze(0)
        temp_feat_pssm = temp_feat_pssm.tolist()
        feat_pred_temp += temp_feat_pssm

        feat_pred.append(feat_pred_temp)
        
    feat_pred = np.array(feat_pred)

    feat_pred_seq, feat_pred_hpdegree, feat_pred_simi, feat_pred_pssm = feat_split_pred(np_feat=feat_pred)
    feat_pred_pca_seq = PCA_SEQ.transform(feat_pred_seq)
    feat_pred_pca_pssm = PCA_PSSM.transform(feat_pred_pssm)
    feat_pred_concat = np.concatenate([feat_pred_pca_seq, feat_pred_hpdegree, feat_pred_simi, feat_pred_pca_pssm], axis=1)

    pred_score_list = np.array(xgbclf.predict_proba(feat_pred_concat))[:, 1].tolist()
    pred_label_list = [1.0 if v>0.999856 else 0.0 for v in pred_score_list] ## specifity=0.99
    #dataframe_pred["pred_score"] = pred_score_list
    dataframe_pred["pred_label"] = pred_label_list
    dataframe_pred.to_csv(outfile, sep="\t", index=False)
    

def main():
    print("START: ",time.ctime(), flush=True)

    run_predictor(infile="./demo_dataset/demo_ppi.csv",
                  outfile="./demo_dataset/demo_result.csv",
                  feature_path="./demo_dataset/"
                  )
    
    print("END: ",time.ctime(), flush=True)


if __name__ == "__main__":
    main()
    
