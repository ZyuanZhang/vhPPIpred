import pandas as pd
import os
import time
import multiprocessing
from joblib import Parallel, delayed

def calcuSimi(args):
    ppifile, outfile = args[0], args[1]
    outF = open(outfile, 'w')
    outF.write("%s\t%s\t%s\t%s\n" % ('virus_unid','human_unid','vp_np_seqsimi','np_hp_score'))

    # 预先加载和优化数据
    dt_hppi = pd.read_csv("./Known_Human_PPI.txt", sep="\t", header=0)
    hppi_dict = dt_hppi.groupby('unid1').apply(lambda x: x[['unid2', 'combined_score']].values.tolist()).to_dict()
    hppi_dict_ = dt_hppi.groupby('unid2').apply(lambda x: x[['unid1', 'combined_score']].values.tolist()).to_dict()
    dt_diamond = pd.read_csv("/home/zhangzhiyuan/BioSoftwares/Diamond/matches.tsv", sep="\t", names=["qseqid","sseqid","pident","length","mismatch","gapopen","qstart","qend","sstart","send","evalue","bitscore"])
    dict_diamond = {row['qseqid'] + "-" + row['sseqid']: row['bitscore'] for _, row in dt_diamond.iterrows()}

    dt = pd.read_csv(ppifile, sep="\t", header=0)
    result_lines = []

    for i, row in dt.iterrows():
        virus_unid = row["virus_unid"]
        human_unid = row["human_unid"]

        # 优化：查找 human_unid 的邻居
        if human_unid in hppi_dict or human_unid in hppi_dict_:
            neighbor_unid_data = []
            if human_unid in hppi_dict:
                neighbor_unid_data += hppi_dict[human_unid]
            if human_unid in hppi_dict_ and hppi_dict_[human_unid] not in neighbor_unid_data:
                neighbor_unid_data += hppi_dict_[human_unid]
            np_hp_score = sum([x[1] for x in neighbor_unid_data]) / len(neighbor_unid_data)
    
            # 提取邻居列表
            neighbor_unid_list = set(x[0] for x in neighbor_unid_data)
            vp_np_score = [dict_diamond.get(virus_unid + "-" + np, 0.0) for np in neighbor_unid_list]
            vp_np_score = sum(vp_np_score) / len(vp_np_score)
    
            result_lines.append(f"{virus_unid}\t{human_unid}\t{vp_np_score}\t{np_hp_score}\n")
            
        else:
            result_lines.append(f"{virus_unid}\t{human_unid}\t0.0\t0.0\n")

    # 批量写入文件，减少I/O操作
    outF.writelines(result_lines)
    outF.close()


def mainDT():
    ppifile = "/home/zhangzhiyuan/Desktop/vhPPI_Other_SOTA_Methods_multiTest/ZhouDatasetEmbed_new/data_zhou_ppi_new.csv"
    outfile = "/home/zhangzhiyuan/Desktop/vhPPI_Other_SOTA_Methods_multiTest/ZhouDatasetEmbed_new/VirusHumanSimiAndDegree/dt_zhou_new_simi.txt"
    calcuSimi([ppifile, outfile])

    
def main(cpu):
    path1 = '/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase2/dataset_interactions/'
    path2 = '/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase2/VirusHumanSimiAndDegree/'

    infile_list = [f[:-10] for f in os.listdir(path1) if f.endswith('_pairs.txt')]
    already_list = [f[:-9] for f in os.listdir(path2) if f.endswith('_simi.txt')]
    need_list = list(set(infile_list) - set(already_list))

    print('全部:{}\t已完成:{}\t还需要:{}'.format(len(infile_list), len(already_list), len(need_list)), flush=True)
    need_list.sort()
    need_list = need_list

    infile_list_more = [path1 + f + '_pairs.txt' for f in need_list]
    outfile_list_more = [path2 + f + '_simi.txt' for f in need_list]

    # 使用 joblib 提高并行效率
    Parallel(n_jobs=int(cpu))(delayed(calcuSimi)(args) for args in zip(infile_list_more, outfile_list_more))


def mainParallel(cpu):
    for ii in range(0, 9000, 1000):
        path1 = '/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase2/dataset_interactions/subset_'+str(ii)+"/"
        path2 = '/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase2/VirusHumanSimiAndDegree/subset_'+str(ii)+"/"
        os.mkdir(path2)

        infile_list = [f[:-10] for f in os.listdir(path1) if f.endswith('_pairs.txt')]
        already_list = [f[:-9] for f in os.listdir(path2) if f.endswith('_simi.txt')]
        need_list = list(set(infile_list) - set(already_list))

        print('Subset_{}\t全部:{}\t已完成:{}\t还需要:{}'.format(ii,len(infile_list), len(already_list), len(need_list)), flush=True)
        need_list.sort()
        need_list = need_list

        infile_list_more = [path1 + f + '_pairs.txt' for f in need_list]
        outfile_list_more = [path2 + f + '_simi.txt' for f in need_list]

        # 使用 joblib 提高并行效率
        Parallel(n_jobs=int(cpu))(delayed(calcuSimi)(args) for args in zip(infile_list_more, outfile_list_more))



if __name__ == "__main__":
    print('START: ', time.ctime(), flush=True)
    #mainParallel(cpu=15)  # 调整 CPU 核心数
    mainDT()
    print('END: ', time.ctime(), flush=True)
