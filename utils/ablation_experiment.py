## xgboost的函数需要修改...

import numpy as np
from utils import load_features, dimreduce, models, oversampling
import pandas as pd
import warnings


warnings.filterwarnings("ignore")

## 读取数据集

def feat_split(np_feat):
    """
        np_feat: [feat_seq(nx2048), feat_hpdegree(nx1), feat_simi(nx2), feat_pssm(nx40)]
    """
    np_feat_seq = np_feat[:, :2048]
    np_feat_hpdegree = np_feat[:, 2048]
    np_feat_hpdegree = np_feat_hpdegree[:, np.newaxis]
    np_feat_simi = np_feat[:, 2049:2051]
    np_feat_pssm = np_feat[:, 2051:2091]
    np_feat_label = np_feat[:, 2091]
    np_feat_label = np_feat_label[:, np.newaxis]

    return np_feat_seq, np_feat_hpdegree, np_feat_simi, np_feat_pssm, np_feat_label


class ClsTrainAndTest(object):
    def __init__(self, dt_dataset):
        self.dt_dataset = dt_dataset

    def __data_constructor(self):
        for testI in range(1,7):
            dt_train = self.dt_dataset[self.dt_dataset["group"]!="gp_"+str(testI)].reset_index(drop=True) 
            dt_test = self.dt_dataset[self.dt_dataset["group"]=="gp_"+str(testI)].reset_index(drop=True) 
            yield (testI, dt_train, dt_test)

    def xgboost_predictor(self, feat_del, n_jobs):
        outF = open("../tmp_res/ablationexp_"+feat_del+".txt", "w")
        for (testI, dt_train, dt_test) in self.__data_constructor():
            ## load features
            feat_train, feat_test = load_features.loadfeatures(dataframe_train=dt_train, dataframe_test=dt_test, FeatPath="../../AllVirusAndHumanProteinEmbed/")
            ## oversampling
            feat_train = oversampling.oversampling(np_feat=feat_train, n_jobs=n_jobs)
            feat_train_seq, feat_train_hpdegree, feat_train_simi, feat_train_pssm, feat_train_label = feat_split(np_feat=feat_train)
            feat_test_seq, feat_test_hpdegree, feat_test_simi, feat_test_pssm, feat_test_label = feat_split(np_feat=feat_test)
            # pca
            feat_train_pca_seq, feat_test_pca_seq = dimreduce.dimreducePCA(feat_train=feat_train_seq, feat_test=feat_test_seq)
            feat_train_pca_pssm, feat_test_pca_pssm = dimreduce.dimreducePCA(feat_train=feat_train_pssm, feat_test=feat_test_pssm)
            # features concat
            if feat_del == "no_del":
                feat_train_concat = np.concatenate([feat_train_pca_seq, feat_train_hpdegree, feat_train_simi, feat_train_pca_pssm, feat_train_label], axis=1) 
                feat_test_concat = np.concatenate([feat_test_pca_seq, feat_test_hpdegree, feat_test_simi, feat_test_pca_pssm, feat_test_label], axis=1)
            elif feat_del == "seqEmbed_pca":
                feat_train_concat = np.concatenate([feat_train_hpdegree, feat_train_simi, feat_train_pca_pssm, feat_train_label], axis=1) 
                feat_test_concat = np.concatenate([feat_test_hpdegree, feat_test_simi, feat_test_pca_pssm, feat_test_label], axis=1)
            if feat_del == "hp_degree":
                feat_train_concat = np.concatenate([feat_train_pca_seq, feat_train_simi, feat_train_pca_pssm, feat_train_label], axis=1) 
                feat_test_concat = np.concatenate([feat_test_pca_seq, feat_test_simi, feat_test_pca_pssm, feat_test_label], axis=1)
            if feat_del == "vh_simi":
                feat_train_concat = np.concatenate([feat_train_pca_seq, feat_train_hpdegree, feat_train_pca_pssm, feat_train_label], axis=1) 
                feat_test_concat = np.concatenate([feat_test_pca_seq, feat_test_hpdegree, feat_test_pca_pssm, feat_test_label], axis=1)
            if feat_del == "pssmEmbed_pca":
                feat_train_concat = np.concatenate([feat_train_pca_seq, feat_train_hpdegree, feat_train_simi, feat_train_label], axis=1) 
                feat_test_concat = np.concatenate([feat_test_pca_seq, feat_test_hpdegree, feat_test_simi, feat_test_label], axis=1)
            # model
            model_name, accuracy, precision, recall, f1_s, auroc, auprc, *_ = models.xgboostForAblationExperiment(np_train=feat_train_concat, np_test=feat_test_concat, n_jobs=n_jobs)
            outF.write("TestI:%s\tModelName:%s\tFeatDelete:%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (testI, model_name, feat_del, accuracy, precision, recall, f1_s, auroc, auprc))
        outF.close()


def xgboost_ablation_experiment(dt_dataset, n_jobs):
    TAT = ClsTrainAndTest(dt_dataset=dt_dataset)
    TAT.xgboost_predictor(feat_del="no_del", n_jobs=n_jobs)
    TAT.xgboost_predictor(feat_del="seqEmbed_pca", n_jobs=n_jobs)
    TAT.xgboost_predictor(feat_del="hp_degree", n_jobs=n_jobs)
    TAT.xgboost_predictor(feat_del="vh_simi", n_jobs=n_jobs)
    TAT.xgboost_predictor(feat_del="pssmEmbed_pca", n_jobs=n_jobs)

