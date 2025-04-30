import os
import time

def getSeqSimi(file1, file2):
    """
        file1: fasta file for creating database of diamond (human proteins in hPPI Network)
        file2: fasta file for searching (virus proteins for calculating seq simi)
    """
    print("Start: ", time.ctime())
    os.chdir("/home/zhangzhiyuan/BioSoftwares/Diamond/")
    
    # create db for diamond
    cmd1 = "./diamond makedb --in "+file1+" -d reference"
    os.system(cmd1)
    print("create db over")
    # running a search in blastp mode
    #cmd2 = "./diamond blastp -d reference -q "+file2+" -o matches.tsv --ultra-sensitive -e 1e-6"
    cmd2 = "./diamond blastp -d reference -q "+file2+" -o matches.tsv --ultra-sensitive"
    os.system(cmd2)
    
    print("End: ", time.ctime())
    
if __name__ == "__main__":
    getSeqSimi(file1="/home/zhangzhiyuan/Desktop/vhPPI_Other_SOTA_Methods_multiTest/ours_vhppipred/scripts/extractFeature/prot_of_human_ppi.fasta",
               file2="/home/zhangzhiyuan/Desktop/vhPPI_Other_SOTA_Methods_multiTest/ZhouDatasetEmbed_new/VirusHumanSimiAndDegree/v_all_seq.fasta")
