# 从 PSSM 中获得蛋白的 PSSM Embedding Matrix
# (nx20) --col mean--> (1x20)
# 步骤：先将PSSM按照行进行归一化，之后按列计算均值
import re
import numpy as np
import os
import time

def getEmbedPSSM(pssmfile, outpath):
    pssm_matrix = []
    with open(pssmfile) as inF:
        for line in inF:
            line = line.strip()
            if re.match(r"^\d", line):
                line_ = line.split()[2:22]
                pssm_matrix.append(line_)
            else:
                continue
    pssm_matrix = np.array(pssm_matrix, dtype=int)
    minval = np.min(pssm_matrix, axis=1)
    maxval = np.max(pssm_matrix, axis=1)
    pssm_matrix_norm = (pssm_matrix - minval[:, np.newaxis]) / (maxval[:, np.newaxis] - minval[:, np.newaxis] + 1e-10)
    pssm_matrix_norm_mean = np.mean(pssm_matrix_norm, axis=0)
    pssm_matrix_norm_mean = pssm_matrix_norm_mean[np.newaxis,:]
    np.save(outpath+pssmfile.split("/")[-1][0:-5]+".npy", pssm_matrix_norm_mean)


def main():
    all_vhf = [f[0:-5] for f in os.listdir("/home/zhangzhiyuan/Desktop/vhPPI_Other_SOTA_Methods_multiTest/ZhouDatasetEmbed_new/humanProtPSSM/") if f.endswith(".pssm")]
    already_vhf = [f[0:-4] for f in os.listdir("/home/zhangzhiyuan/Desktop/vhPPI_Other_SOTA_Methods_multiTest/ZhouDatasetEmbed_new/humanProtPSSMembed/") if f.endswith(".npy")]
    need_vhf = list(set(all_vhf)-set(already_vhf))
    print("need: {}".format(len(need_vhf)))
    vhf_list = ["/home/zhangzhiyuan/Desktop/vhPPI_Other_SOTA_Methods_multiTest/ZhouDatasetEmbed_new/humanProtPSSM/"+f+'.pssm' for f in need_vhf]
    for vhf in vhf_list:
        getEmbedPSSM(pssmfile=vhf, outpath="/home/zhangzhiyuan/Desktop/vhPPI_Other_SOTA_Methods_multiTest/ZhouDatasetEmbed_new/humanProtPSSMembed/")


if __name__ == "__main__":
    print("START: ", time.ctime(),flush=True)
    main()
    print("END: ", time.ctime(),flush=True)
