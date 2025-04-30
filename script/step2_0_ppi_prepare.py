import os
import time
import pandas as pd
from itertools import product


def getPPI():
    virus_list = [f[0:-4] for f in os.listdir("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase2/virusProtPSSMembed/") if f.endswith('.npy')]
    human_list = [f[0:-4] for f in os.listdir("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase2/humanProtPSSMembed/") if f.endswith('.npy')] 
    print('virus:{}\thuman:{}'.format(len(virus_list), len(human_list)), flush=True)

    for vid in virus_list:
        dt_tmp = list(product([vid,], human_list))
        dt_tmp = pd.DataFrame(dt_tmp, columns=["virus_unid","human_unid"])
        dt_tmp.to_csv("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase2/dataset_interactions/"+vid+"_pairs.txt", sep="\t", index=False)

def getPPIparallel():
    virus_list = [f[0:-4] for f in os.listdir("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase2/virusProtPSSMembed/") if f.endswith('.npy')]
    human_list = [f[0:-4] for f in os.listdir("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase2/humanProtPSSMembed/") if f.endswith('.npy')] 
    print('virus:{}\thuman:{}'.format(len(virus_list), len(human_list)), flush=True)

    for i in range(len(virus_list)):
        vid=virus_list[i]
        dt_tmp = list(product([vid,], human_list))
        dt_tmp = pd.DataFrame(dt_tmp, columns=["virus_unid","human_unid"])
        if i%1000==0:
            os.mkdir("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase2/dataset_interactions/subset_"+str(i)+"/")
            jj = i
        dt_tmp.to_csv("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase2/dataset_interactions/subset_"+str(jj)+"/"+vid+"_pairs.txt", sep="\t", index=False)
        print("subset_{} = {}/{}".format(jj, i+1, len(virus_list)), flush=True, end="\r")

if __name__ == "__main__":
    print("START: ", time.ctime(), flush=True)
    getPPIparallel()
    print("END: ", time.ctime(), flush=True)
