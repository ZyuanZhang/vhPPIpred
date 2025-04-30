# pipeline for benchmark dataset construction

import pandas as pd
from collections import Counter, defaultdict
import random
from itertools import product
import numpy as np
from Bio import SeqIO


class PosAndNegSamplesConstruction(object):
    def __init__(self, vf_pos_cluster, hf_pos_cluster, ppif_pos, vf_neg_cluster, hf_neg_cluster, ppif_neg):
        ## positive data
        self.vdf_pos_cluster = pd.read_csv(vf_pos_cluster, sep="\t", names=["cluster","representative"])
        self.hdf_pos_cluster = pd.read_csv(hf_pos_cluster, sep="\t", names=["cluster","representative"])
        self.ppidf_pos = pd.read_csv(ppif_pos, sep="\t", header=0) ## colnames = ["virus_unid","human_unid"]
        ## negative data
        self.vdf_neg_cluster = pd.read_csv(vf_neg_cluster, sep="\t", names=["cluster","representative"])
        self.hdf_neg_cluster = pd.read_csv(hf_neg_cluster, sep="\t", names=["cluster","representative"])
        self.ppidf_neg = pd.read_csv(ppif_neg, sep="\t", header=0) ## colnames = ["virus_unid","human_unid"]


    def pos_sample_division(self):
        dict_virus_cluster = {self.vdf_pos_cluster["representative"][i]:self.vdf_pos_cluster["cluster"][i] for i in range(self.vdf_pos_cluster.shape[0])}
        self.ppidf_pos["virus_cluster"] = [dict_virus_cluster[x] for x in self.ppidf_pos["virus_unid"]]
        virus_pos_cluster_list = list(set(self.ppidf_pos["virus_cluster"]))
    
        random.seed(3) ## divide pos ppi to six groups based on viral protein clusters
        virus_pos_cluster_list = random.sample(virus_pos_cluster_list, len(virus_pos_cluster_list))
        vpgp1 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[0:84])].reset_index(drop=True)
        vpgp2 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[84:84*2])].reset_index(drop=True)
        vpgp3 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[84*2:84*3])].reset_index(drop=True)
        vpgp4 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[84*3:84*4])].reset_index(drop=True)
        vpgp5 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[84*4:84*5])].reset_index(drop=True)
        vpgp6 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[84*5:])].reset_index(drop=True)
        minPosCount = min([vpgp1.shape[0],vpgp2.shape[0],vpgp3.shape[0],vpgp4.shape[0],vpgp5.shape[0],vpgp6.shape[0]]) ## set minimum as threshold
        
        return vpgp1, vpgp2, vpgp3, vpgp4, vpgp5, vpgp6, minPosCount
    

    def neg_sample_division(self):
        dict_virus_cluster = {self.vdf_neg_cluster["representative"][i]:self.vdf_neg_cluster["cluster"][i] for i in range(self.vdf_neg_cluster.shape[0])}
        self.ppidf_neg["virus_cluster"] = [dict_virus_cluster[x] for x in self.ppidf_neg["virus_unid"]]
        virus_neg_cluster_list = list(set(self.ppidf_neg["virus_cluster"]))
    
        random.seed(3) ## divide pos ppi to six groups based on viral protein clusters
        virus_pos_cluster_list = random.sample(virus_pos_cluster_list, len(virus_pos_cluster_list))
        vpgp1 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[0:84])].reset_index(drop=True)
        vpgp2 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[84:84*2])].reset_index(drop=True)
        vpgp3 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[84*2:84*3])].reset_index(drop=True)
        vpgp4 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[84*3:84*4])].reset_index(drop=True)
        vpgp5 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[84*4:84*5])].reset_index(drop=True)
        vpgp6 = self.ppidf_pos[self.ppidf_pos["virus_cluster"].isin(virus_pos_cluster_list[84*5:])].reset_index(drop=True)

        minPosCount = min([vpgp1.shape[0],vpgp2.shape[0],vpgp3.shape[0],vpgp4.shape[0],vpgp5.shape[0],vpgp6.shape[0]]) ## set minimum as threshold
        
        return vpgp1, vpgp2, vpgp3, vpgp4, vpgp5, vpgp6, minPosCount
        
        

        dt_allhumanvirus_fengyang = pd.read_excel("../original_dataset/all_humanvirus_fengyang.xlsx", sheet_name="Sheet1", header=0)
        virusspeciesname_fengyang = set(dt_allhumanvirus_fengyang["Species"])

        ## 所有宿主为非人哺乳动物的病毒蛋白对应的病毒物种
        dict_nohumanvirus_species_prot = defaultdict(list)
        for record in SeqIO.parse("../original_dataset/virushostdb.formatted.cds.faa", "fasta"):
            virusspecies, virusprot = " ".join(str(record.description).split("|")[0].split()[1:]), str(record.id)
            dict_nohumanvirus_species_prot[virusspecies].append(virusprot)
        virusspeciesname_nohuman = set(dict_nohumanvirus_species_prot.keys())

        except_virus_id = []
        for vspecies, vprotlist in dict_nohumanvirus_species_prot.items():
            if vspecies in virusspeciesname_fengyang:
                except_virus_id += vprotlist

        dt_virus_nohvirus_cluster = pd.read_csv("../original_dataset/virus_prots_nohuman_virus_protsRes_cluster.tsv", sep="\t", names=["cluster","representative"])
        dt_nohvirus_cluster = pd.DataFrame(data=None, columns=["cluster", "representative"])
        for _,gp in dt_virus_nohvirus_cluster.groupby(by=["cluster"]):
            gp.reset_index(drop=True, inplace=True)
            if (len(set(gp["representative"]) & set(dt_virus_cluster["representative"]))==0) and (len(set(gp["representative"]) & set(except_virus_id))==0):
                dt_nohvirus_cluster = pd.concat([dt_nohvirus_cluster, gp], ignore_index=True)
        virus_neg_cluster_list = list(set(dt_nohvirus_cluster["cluster"]))
        print(len(virus_neg_cluster_list)) ## 负样本病毒蛋白簇


        dt_human_pos_cluster = pd.read_csv("../original_dataset/human_prots_posRes_cluster.tsv", sep="\t", names=["cluster","representative"])
        dt_human_pos_neg_cluster = pd.read_csv("../original_dataset/human_prots_neg_posRes_cluster.tsv", sep="\t", names=["cluster","representative"])
        dt_human_neg_cluster = pd.DataFrame(data=None, columns=["cluster","representative"])
        for _,gp in dt_human_pos_neg_cluster.groupby(by=["cluster"]):
            gp.reset_index(drop=True, inplace=True)
            if len(set(gp["representative"]) & set(dt_human_pos_cluster["representative"]))==0:
                dt_human_neg_cluster = pd.concat([dt_human_neg_cluster, gp], ignore_index=True)
        human_neg_cluster_list = list(set(dt_human_neg_cluster["cluster"]))


    def vpgp_under_sampling(self, vpgpN, vpgpNRest, minPosCount, group):
        dict_human_cluster = {self.hdf_pos_cluster["representative"][i]:self.hdf_pos_cluster["cluster"][i] for i in range(self.hdf_pos_cluster.shape[0])}
        if vpgpN.shape[0] == minPosCount:
            vpgpN_new = vpgpN.loc[:, ['virus_unid', 'human_unid']]
            vpgpN_new["group"] = [group,]*len(vpgpN_new)
            vpgpN_new["label"] = [1.0,]*len(vpgpN_new)
        else:
            vpgpN["human_cluster"] = [dict_human_cluster[vpgpN["human_unid"][i]] for i in range(vpgpN.shape[0])]
            vpgpNRest["human_cluster"] = [dict_human_cluster[vpgpNRest["human_unid"][i]] for i in range(vpgpNRest.shape[0])]
            vpgpN_new = []
            vpgpN_vhppi_list = [vpgpN["virus_unid"][i]+"-"+vpgpN["human_unid"][i]+"-"+vpgpN["human_cluster"][i] for i in range(vpgpN.shape[0])]
            vpgpN_virus_list = list([x.split("-")[0] for x in vpgpN_vhppi_list])
            vpgpN_vhppi_cluster, vpgpNRest_vhppi_cluster = list(set(vpgpN["human_cluster"])), list(set(vpgpNRest["human_cluster"]))

            ## Top1
            vpgpN_vhppi_target_cluster_top1 = list(set(vpgpN_vhppi_cluster)-set(vpgpNRest_vhppi_cluster)) ## firstly select vpgpN human clusters that were not in vpgpNRest
            
            temp_intersection = list(set(vpgpN_vhppi_cluster) & set(vpgpNRest_vhppi_cluster))
            dt_temp_cluster_count, temp_element = [], [x for x in temp_intersection if x in vpgpNRest_vhppi_cluster]
            for cluster, count in Counter(temp_element).items():
                dt_temp_cluster_count.append([cluster, count])
            dt_temp_cluster_count = pd.DataFrame(dt_temp_cluster_count, columns=["cluster","count"])
            dt_temp_cluster_count.sort_values(by=["count"], ascending=True, ignore_index=True, inplace=True)
            ## Top2
            vpgpN_vhppi_target_cluster_top2 = list(dt_temp_cluster_count["cluster"])
            
            while len(vpgpN_new) < minPosCount:
                vpgpN_virus_set = list(set(vpgpN_virus_list))
                for vid in vpgpN_virus_set:
                    for vhppi in vpgpN_vhppi_list:
                        if vid == vhppi.split("-")[0]:
                            if vhppi.split("-")[2] in vpgpN_vhppi_target_cluster_top1:
                                vpgpN_new.append("-".join(vhppi.split("-")[0:2]))
                                break
                            elif vhppi.split("-")[2] in vpgpN_vhppi_target_cluster_top2:
                                vpgpN_new.append("-".join(vhppi.split("-")[0:2]))
                                break
                            #else:
                                #print(vhppi)
                            #    break
                    if len(vpgpN_new)==minPosCount:
                        break
                vpgpN_vhppi_list = list(set(vpgpN_vhppi_list) - set(vpgpN_new))
                vpgpN_virus_list = list([x.split("-")[0] for x in vpgpN_vhppi_list])
            vpgpN_new = [[x.split("-")[0],x.split("-")[1]] for x in vpgpN_new]
            vpgpN_new = pd.DataFrame(vpgpN_new, columns=["virus_unid","human_unid"])
            vpgpN_new["group"] = [group,]*len(vpgpN_new)
            vpgpN_new["label"] = [1.0,]*len(vpgpN_new)

        return vpgpN_new


def main_pos_samples_construct():
    PSC = PositiveSamplesConstruction(vf_pos_cluster="../",
                               hf_pos_cluster="../",
                               ppif_pos="../")
    vpgp1, vpgp2, vpgp3, vpgp4, vpgp5, vpgp6, minPosCount = PSC.pos_sample_division()
    vpgpN_1, vpgpNRest_1 = vpgp1, pd.concat([vpgp2, vpgp3, vpgp4, vpgp5, vpgp6], ignore_index=True)
    vpgpN_2, vpgpNRest_2 = vpgp2, pd.concat([vpgp1, vpgp3, vpgp4, vpgp5, vpgp6], ignore_index=True)
    vpgpN_3, vpgpNRest_3 = vpgp3, pd.concat([vpgp1, vpgp2, vpgp4, vpgp5, vpgp6], ignore_index=True)
    vpgpN_4, vpgpNRest_4 = vpgp4, pd.concat([vpgp1, vpgp2, vpgp3, vpgp5, vpgp6], ignore_index=True)
    vpgpN_5, vpgpNRest_5 = vpgp5, pd.concat([vpgp1, vpgp2, vpgp3, vpgp4, vpgp6], ignore_index=True)
    vpgpN_6, vpgpNRest_6 = vpgp6, pd.concat([vpgp1, vpgp2, vpgp3, vpgp4, vpgp5], ignore_index=True)

    vpgp1_new = PSC.vpgp_under_sampling(vpgpN=vpgpN_1, vpgpNRest=vpgpNRest_1, minPosCount=minPosCount, group="gp_1")
    vpgp2_new = PSC.vpgp_under_sampling(vpgpN=vpgpN_2, vpgpNRest=vpgpNRest_2, minPosCount=minPosCount, group="gp_2")
    vpgp3_new = PSC.vpgp_under_sampling(vpgpN=vpgpN_3, vpgpNRest=vpgpNRest_3, minPosCount=minPosCount, group="gp_3")
    vpgp4_new = PSC.vpgp_under_sampling(vpgpN=vpgpN_4, vpgpNRest=vpgpNRest_4, minPosCount=minPosCount, group="gp_4")
    vpgp5_new = PSC.vpgp_under_sampling(vpgpN=vpgpN_5, vpgpNRest=vpgpNRest_5, minPosCount=minPosCount, group="gp_5")
    vpgp6_new = PSC.vpgp_under_sampling(vpgpN=vpgpN_6, vpgpNRest=vpgpNRest_6, minPosCount=minPosCount, group="gp_6")
        
    print("dt_group_N.shape = {}".format(vpgp1_new.shape))
    dt_positive_samples = pd.concat([vpgp1_new, vpgp2_new, vpgp3_new, vpgp4_new, vpgp5_new, vpgp6_new], ignore_index=True)
    print("dt_positive_samples = {}".format(dt_positive_samples.shape))


class NegativeSamplesConstruction(object):
    def __init__(self, vf_neg_cluster, hf_neg_cluster, ppif_neg):
        self.vdf_neg_cluster = pd.read_csv(vf_neg_cluster, sep="\t", names=["cluster","representative"])
        self.hdf_neg_cluster = pd.read_csv(hf_neg_cluster, sep="\t", names=["cluster","representative"])
        self.ppidf_neg = pd.read_csv(ppif_neg, sep="\t", header=0) ## colnames = ["virus_unid", "human_unid"]
        


## FengYang提供的所有人类病毒蛋白组数据，根据该数据，将vhppipred的所有宿主为非人哺乳动物的病毒蛋白在病毒种水平上进行去除，剩下的病毒蛋白用于构建vhppipred的负样本。

## FengYang数据集中所有人类病毒
dt_allhumanvirus_fengyang = pd.read_excel("../original_dataset/all_humanvirus_fengyang.xlsx", sheet_name="Sheet1", header=0)
virusspeciesname_fengyang = set(dt_allhumanvirus_fengyang["Species"])

## 所有宿主为非人哺乳动物的病毒蛋白对应的病毒物种
dict_nohumanvirus_species_prot = defaultdict(list)
for record in SeqIO.parse("../original_dataset/virushostdb.formatted.cds.faa", "fasta"):
    virusspecies, virusprot = " ".join(str(record.description).split("|")[0].split()[1:]), str(record.id)
    dict_nohumanvirus_species_prot[virusspecies].append(virusprot)
virusspeciesname_nohuman = set(dict_nohumanvirus_species_prot.keys())

except_virus_id = []
for vspecies, vprotlist in dict_nohumanvirus_species_prot.items():
    if vspecies in virusspeciesname_fengyang:
        except_virus_id += vprotlist

dt_virus_nohvirus_cluster = pd.read_csv("../original_dataset/virus_prots_nohuman_virus_protsRes_cluster.tsv", sep="\t", names=["cluster","representative"])
dt_nohvirus_cluster = pd.DataFrame(data=None, columns=["cluster", "representative"])
for _,gp in dt_virus_nohvirus_cluster.groupby(by=["cluster"]):
    gp.reset_index(drop=True, inplace=True)
    if (len(set(gp["representative"]) & set(dt_virus_cluster["representative"]))==0) and (len(set(gp["representative"]) & set(except_virus_id))==0):
        dt_nohvirus_cluster = pd.concat([dt_nohvirus_cluster, gp], ignore_index=True)
virus_neg_cluster_list = list(set(dt_nohvirus_cluster["cluster"]))
print(len(virus_neg_cluster_list)) ## 负样本病毒蛋白簇


dt_human_pos_cluster = pd.read_csv("../original_dataset/human_prots_posRes_cluster.tsv", sep="\t", names=["cluster","representative"])
dt_human_pos_neg_cluster = pd.read_csv("../original_dataset/human_prots_neg_posRes_cluster.tsv", sep="\t", names=["cluster","representative"])
dt_human_neg_cluster = pd.DataFrame(data=None, columns=["cluster","representative"])
for _,gp in dt_human_pos_neg_cluster.groupby(by=["cluster"]):
    gp.reset_index(drop=True, inplace=True)
    if len(set(gp["representative"]) & set(dt_human_pos_cluster["representative"]))==0:
        dt_human_neg_cluster = pd.concat([dt_human_neg_cluster, gp], ignore_index=True)
human_neg_cluster_list = list(set(dt_human_neg_cluster["cluster"]))


## 负样本构建方法 (-)

random.seed(3)

## 1818/6 = 303
virus_neg_cluster_list = random.sample(virus_neg_cluster_list, len(virus_neg_cluster_list)) ## 负样本人类蛋白簇
vnclu1 = dt_nohvirus_cluster[dt_nohvirus_cluster["cluster"].isin(virus_neg_cluster_list[0:303])].reset_index(drop=True)
vnclu2 = dt_nohvirus_cluster[dt_nohvirus_cluster["cluster"].isin(virus_neg_cluster_list[303:303*2])].reset_index(drop=True)
vnclu3 = dt_nohvirus_cluster[dt_nohvirus_cluster["cluster"].isin(virus_neg_cluster_list[303*2:303*3])].reset_index(drop=True)
vnclu4 = dt_nohvirus_cluster[dt_nohvirus_cluster["cluster"].isin(virus_neg_cluster_list[303*3:303*4])].reset_index(drop=True)
vnclu5 = dt_nohvirus_cluster[dt_nohvirus_cluster["cluster"].isin(virus_neg_cluster_list[303*4:303*5])].reset_index(drop=True)
vnclu6 = dt_nohvirus_cluster[dt_nohvirus_cluster["cluster"].isin(virus_neg_cluster_list[303*5:])].reset_index(drop=True)


human_neg_cluster_list = random.sample(human_neg_cluster_list, len(human_neg_cluster_list))  ## 负样本人类蛋白簇
hnclu1 = dt_human_neg_cluster[dt_human_neg_cluster["cluster"].isin(human_neg_cluster_list[0:681])].reset_index(drop=True)
hnclu2 = dt_human_neg_cluster[dt_human_neg_cluster["cluster"].isin(human_neg_cluster_list[681:681*2])].reset_index(drop=True)
hnclu3 = dt_human_neg_cluster[dt_human_neg_cluster["cluster"].isin(human_neg_cluster_list[681*2:681*3])].reset_index(drop=True)
hnclu4 = dt_human_neg_cluster[dt_human_neg_cluster["cluster"].isin(human_neg_cluster_list[681*3:681*4])].reset_index(drop=True)
hnclu5 = dt_human_neg_cluster[dt_human_neg_cluster["cluster"].isin(human_neg_cluster_list[681*4:681*5])].reset_index(drop=True)
hnclu6 = dt_human_neg_cluster[dt_human_neg_cluster["cluster"].isin(human_neg_cluster_list[681*5:])].reset_index(drop=True)

## 负样本病毒蛋白簇和人类蛋白簇进行随机组合
vngp1 = pd.DataFrame(list(product(list(vnclu1["representative"]), list(hnclu1["representative"]))), columns=["virus_unid", "human_unid"])
vngp2 = pd.DataFrame(list(product(list(vnclu2["representative"]), list(hnclu2["representative"]))), columns=["virus_unid", "human_unid"])
vngp3 = pd.DataFrame(list(product(list(vnclu3["representative"]), list(hnclu3["representative"]))), columns=["virus_unid", "human_unid"])
vngp4 = pd.DataFrame(list(product(list(vnclu4["representative"]), list(hnclu4["representative"]))), columns=["virus_unid", "human_unid"])
vngp5 = pd.DataFrame(list(product(list(vnclu5["representative"]), list(hnclu5["representative"]))), columns=["virus_unid", "human_unid"])
vngp6 = pd.DataFrame(list(product(list(vnclu6["representative"]), list(hnclu6["representative"]))), columns=["virus_unid", "human_unid"])

print(vngp1.shape)
print(vngp2.shape)
print(vngp3.shape)
print(vngp4.shape)
print(vngp5.shape)
print(vngp6.shape)



def vngpSampling(vngpN, vngpNRest, minPosCount, group, fold=10): ## fold: pos:neg=1:10
    if vngpN.shape[0] == minPosCount*fold:
        vngpN_new = vngpN.loc[:, ['virus_unid', 'human_unid']]
        vngpN_new["group"] = [group,]*len(vngpN_new)
        vngpN_new["label"] = [0.0,]*len(vngpN_new)
    else:
        vngpN["human_cluster"] = [dict_human_cluster[vngpN["human_unid"][i]] for i in range(vngpN.shape[0])]
        vngpNRest["human_cluster"] = [dict_human_cluster[vngpNRest["human_unid"][i]] for i in range(vngpNRest.shape[0])]
        vngpN_new = []
        vngpN_vhppi_list = [vngpN["virus_unid"][i]+"-"+vngpN["human_unid"][i]+"-"+vngpN["human_cluster"][i] for i in range(vngpN.shape[0])]
        vngpN_virus_list = list([x.split("-")[0] for x in vngpN_vhppi_list])
        vngpN_vhppi_cluster, vngpNRest_vhppi_cluster = list(set(vngpN["human_cluster"])), list(set(vngpNRest["human_cluster"]))

        ## Top1
        vngpN_vhppi_target_cluster_top1 = list(set(vngpN_vhppi_cluster)-set(vngpNRest_vhppi_cluster)) ## vngpN的human clusters 不在 vngpNRest的human clusters中 (采样优先选这种)
        #print(len(vngpN_vhppi_target_cluster_top1))

        temp_intersection = list(set(vngpN_vhppi_cluster) & set(vngpNRest_vhppi_cluster))
        dt_temp_cluster_count, temp_element = [], [x for x in temp_intersection if x in vngpNRest_vhppi_cluster]
        for cluster, count in Counter(temp_element).items():
            dt_temp_cluster_count.append([cluster, count])
        dt_temp_cluster_count = pd.DataFrame(dt_temp_cluster_count, columns=["cluster","count"])
        dt_temp_cluster_count.sort_values(by=["count"], ascending=True, ignore_index=True, inplace=True)
        ## Top2
        vngpN_vhppi_target_cluster_top2 = list(dt_temp_cluster_count["cluster"])
        #print(len(vngpN_vhppi_target_cluster_top2))
        
        while len(vngpN_new) < minPosCount*10:
            vngpN_virus_set = list(set(vngpN_virus_list))
            for vid in vngpN_virus_set:
                for vhppi in vngpN_vhppi_list:
                    if vid == vhppi.split("-")[0]:
                        if vhppi.split("-")[2] in vngpN_vhppi_target_cluster_top1:
                            vngpN_new.append("-".join(vhppi.split("-")[0:2]))
                            break
                        elif vhppi.split("-")[2] in vngpN_vhppi_target_cluster_top2:
                            vngpN_new.append("-".join(vhppi.split("-")[0:2]))
                            break
                        #else:
                            #print(vhppi)
                        #    break
                if len(vngpN_new)==minPosCount*fold:
                    break
            vngpN_vhppi_list = list(set(vngpN_vhppi_list) - set(vngpN_new))
            vngpN_virus_list = list([x.split("-")[0] for x in vngpN_vhppi_list])
        vngpN_new = [[x.split("-")[0],x.split("-")[1]] for x in vngpN_new]
        vngpN_new = pd.DataFrame(vngpN_new, columns=["virus_unid","human_unid"])
        vngpN_new["group"] = [group,]*len(vngpN_new)
        vngpN_new["label"] = [0.0,]*len(vngpN_new)

    return vngpN_new


## 调用负样本构建方法

vngpN_1, vngpNRest_1 = vngp1, pd.concat([vngp2, vngp3, vngp4, vngp5, vngp6], ignore_index=True)
vngpN_2, vngpNRest_2 = vngp2, pd.concat([vngp1, vngp3, vngp4, vngp5, vngp6], ignore_index=True)
vngpN_3, vngpNRest_3 = vngp3, pd.concat([vngp1, vngp2, vngp4, vngp5, vngp6], ignore_index=True)
vngpN_4, vngpNRest_4 = vngp4, pd.concat([vngp1, vngp2, vngp3, vngp5, vngp6], ignore_index=True)
vngpN_5, vngpNRest_5 = vngp5, pd.concat([vngp1, vngp2, vngp3, vngp4, vngp6], ignore_index=True)
vngpN_6, vngpNRest_6 = vngp6, pd.concat([vngp1, vngp2, vngp3, vngp4, vngp5], ignore_index=True)

vngp1_new = vngpSampling(vngpN=vngpN_1, vngpNRest=vngpNRest_1, minPosCount=minPosCount, group="gp_1", fold=10)
vngp2_new = vngpSampling(vngpN=vngpN_2, vngpNRest=vngpNRest_2, minPosCount=minPosCount, group="gp_2", fold=10)
vngp3_new = vngpSampling(vngpN=vngpN_3, vngpNRest=vngpNRest_3, minPosCount=minPosCount, group="gp_3", fold=10)
vngp4_new = vngpSampling(vngpN=vngpN_4, vngpNRest=vngpNRest_4, minPosCount=minPosCount, group="gp_4", fold=10)
vngp5_new = vngpSampling(vngpN=vngpN_5, vngpNRest=vngpNRest_5, minPosCount=minPosCount, group="gp_5", fold=10)
vngp6_new = vngpSampling(vngpN=vngpN_6, vngpNRest=vngpNRest_6, minPosCount=minPosCount, group="gp_6", fold=10)
        
print(vngp1_new.shape)
print(vngp2_new.shape)
print(vngp3_new.shape)
print(vngp4_new.shape)
print(vngp5_new.shape)
print(vngp6_new.shape)


## 正负样本合并
dt_res = pd.concat([vpgp1_new, vngp1_new, 
                    vpgp2_new, vngp2_new,
                    vpgp3_new, vngp3_new,
                    vpgp4_new, vngp4_new,
                    vpgp5_new, vngp5_new,
                    vpgp6_new, vngp6_new], ignore_index=True)


dt_res.to_csv("../dataset/data_fold10.csv", sep="\t", index=False) ## 不同fold之间的病毒蛋白没有相似性，但是人类蛋白可能存在相似性。
dt_human_pos_neg_cluster = pd.read_csv("../original_dataset/human_prots_neg_posRes_cluster.tsv", sep="\t", names=["cluster","representative"])
dict_human_cluster = {dt_human_pos_neg_cluster["representative"][i]:dt_human_pos_neg_cluster["cluster"][i] for i in range(dt_human_pos_neg_cluster.shape[0])}
dt_res["human_cluster"] = [dict_human_cluster[h] for h in dt_res["human_unid"]]
dt_res.to_csv("../dataset/data_fold10_info.csv", sep="\t", index=False)