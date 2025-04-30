import pandas as pd
import time

def main():
    print(time.ctime(), flush=True)
    dt_hppi = pd.read_csv("../datasource/Known_Human_PPI.txt", sep="\t", header=0)
    dt = pd.read_csv("../../../ours_data_augment/banchmark_dataset_test/dataset_denovo/data_denovo.csv", sep="\t", header=0)
    print(dt.shape, flush=True)
    dt_diamond = pd.read_csv("/home/zhangzhiyuan/BioSoftwares/Diamond/matches.tsv", sep="\t", names=["qseqid","sseqid","pident","length","mismatch","gapopen","qstart","qend","sstart","send","evalue","bitscore"])
    dict_diamond = {dt_diamond["qseqid"][i]+"-"+dt_diamond["sseqid"][i]:dt_diamond["bitscore"][i] for i in range(dt_diamond.shape[0])}
    
    for i in range(dt.shape[0]):
        virus_unid = dt["virus_unid"][i]
        human_unid = dt["human_unid"][i]
        neighbor_unid = dt_hppi[(dt_hppi["unid1"]==human_unid) | (dt_hppi["unid2"]==human_unid)]
        if neighbor_unid.shape[0] != 0:
            np_hp_score = sum(neighbor_unid["combined_score"])/neighbor_unid.shape[0]
            neighbor_unid_list = set(neighbor_unid["unid1"]) | set(neighbor_unid["unid2"]) - set(human_unid)
            vp_np_score = []
            for np in neighbor_unid_list:
                try:
                    vp_np_score.append(dict_diamond[virus_unid+"-"+np])
                except:
                    vp_np_score.append(0.0)
            vp_np_score = sum(vp_np_score)/len(vp_np_score)
            print("%s\t%s\t%s\t%s" % (virus_unid, human_unid, vp_np_score, np_hp_score), flush=True)
        else:
            print("%s\t%s\t%s\t%s" % (virus_unid, human_unid, 0.0, 0.0), flush=True)
    print(time.ctime(), flush=True)


if __name__ == "__main__":
    main()
