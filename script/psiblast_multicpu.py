## 在 113 (10.0.0.113) 上运行


import subprocess
import multiprocessing
import time
import os

def runPsiBlast(args):
    queryFile, dbFile, outFile = args[0], args[1], args[2]
    #cmd = f"./psiblast -query {queryFile} -db {dbFile} -out_ascii_pssm {outFile} -num_iterations 3 -evalue 0.001"
    cmd = f"./psiblast -query {queryFile} -db {dbFile} -out_ascii_pssm {outFile} -num_iterations 3"
    subprocess.run(cmd, shell=True)


def main(queryPath, outPath, dbFile, cpu):
    """
        queryPath: query fasta文件所在的绝对路径；
        outPath: 生成的 pssm 文件所在的绝对路径；
        dbFile: 库文件（需要绝对路径）；
        cpu: 并行所需的 cpu 数目；
    """
    queryFileList = [x[0:-6] for x in os.listdir(queryPath) if x.endswith(".fasta")] ## 全部的目标fasta文件
    alreadyFileList = [x[0:-5] for x in os.listdir(outPath) if x.endswith(".pssm")] ## 输出目录中已经存在的pssm结果文件（主要是如果程序中止，后续可以接着将没有pssm结果的fasta继续进行psiblast比对，不需要全部重新运行）
    needFileList = list(set(queryFileList) - set(alreadyFileList))
    print("全部的fasta文件的总数: %s\t已经有pssm结果的文件数目: %s\t还需要生成pssm的fasta文件的数目: %s" % (len(queryFileList), len(alreadyFileList), len(needFileList))) ## 全部的，已完成的，未完成的
    
    needFileList.sort()
    queryFastaList = [queryPath+x+".fasta" for x in needFileList]
    outPSSMList = [outPath+x+".pssm" for x in needFileList]
    dbFileList = [dbFile,]*len(queryFastaList)
    
    ## 多进程
    pool = multiprocessing.Pool(processes=int(cpu))
    pool.map(runPsiBlast, list(zip(queryFastaList, dbFileList, outPSSMList)))
    pool.close()
    pool.join()


if __name__ == "__main__":
    print("开始时间: ",time.ctime())
    #queryPath = ""
    #outPath = "/data/12T-1/zhangzhiyuan/DeNovoDatasetEmbed/humanProtPSSM/"
    #

    #dbFile = "/data/12T-1/zhangzhiyuan/TempFiles/SwissProtDB/uniprot_sprot.fasta"    
    ##dbFile = "/data/12T-1/zhangzhiyuan/TempFiles/Uniref50DB/uniref50.fasta"    
    ##dbFile = "/data/12T-1/zhangzhiyuan/TempFiles/Uniref90DB/uniref90.fasta"    

    #main(queryPath=queryPath, outPath=outPath, dbFile=dbFile, cpu=10)
    
    print("结束时间: ",time.ctime())

    
    
