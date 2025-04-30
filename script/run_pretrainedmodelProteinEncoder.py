# Run pretrained protein language model encoder to convert protein sequence to feature vectors
# Reference: https://github.com/agemagician/ProtTrans#feature-extraction
# env: conda activate PLMEncode 
# 在蛋白水平上对蛋白质进行编码，得到长度为1024 的特征向量

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import os
from Bio import SeqIO
import numpy as np
import time

class PretrainedProtLanguageModelEncoder(object):
    def __init__(self):
        token_zzy = "xxxx"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained("/data/150T/databases/help_zhangzhiyuan/PretrainModelFromHuggingFace/prot_t5_xl_half_uniref50-enc", do_lower_case=False, torch_dtype=torch.float16, token=token_zzy)
        self.model = T5EncoderModel.from_pretrained("/data/150T/databases/help_zhangzhiyuan/PretrainModelFromHuggingFace/prot_t5_xl_half_uniref50-enc", token=token_zzy).to(self.device)

    def encoder(self, protein_id_list, protein_sequence_list, outpath):
        protein_seqlen_list = [len(s) for s in protein_sequence_list]
        protein_sequence_list = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein_sequence_list]
        ids = self.tokenizer.batch_encode_plus(protein_sequence_list, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids["input_ids"]).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids,attention_mask=attention_mask)

        prot_embed = embedding_repr.last_hidden_state
        for i in range(0,prot_embed.shape[0]):
            prot_embed_i = prot_embed[i,:protein_seqlen_list[i]]
            prot_embed_i_per_protein = prot_embed_i.mean(dim=0)
            ###print(prot_embed_i_per_protein)
            ###prot_embed_np = prot_embed_i_per_protein.numpy()
            ###np.savetxt(prot_embed_np)
            print("%s\t%s\t%s\t%s" % (protein_id_list[i], prot_embed_i.shape[0], prot_embed_i.shape[1], prot_embed_i_per_protein.shape), flush=True)
            torch.save(prot_embed_i_per_protein, outpath+protein_id_list[i]+".pt")



def main(fasta_in_path, out_path, num_thread=20):
    print("Start: ", time.ctime(), flush=True)
    PLM = PretrainedProtLanguageModelEncoder()
    fasta_all = [f[0:-6] for f in os.listdir(fasta_in_path) if f.endswith(".fasta")]
    fasta_already = [f[0:-3] for f in os.listdir(out_path) if f.endswith(".pt")]
    fasta_need = [fasta_in_path+f+".fasta" for f in list(set(fasta_all)-set(fasta_already))]
    print("Need: ", len(fasta_need), flush=True)
    for num in range(0, len(fasta_need), num_thread):
        if num < len(fasta_need):
            part_fasta = fasta_need[num: num+num_thread]
        else:
            part_fasta = fasta_need[num: len(fasta_need)]
        id_list, seq_list = [], []
        for f in part_fasta:
            for record in SeqIO.parse(f, "fasta"):
                id_ = str(record.id)
                seq = str(record.seq)
            id_list.append(id_)
            seq_list.append(seq)
        PLM.encoder(protein_id_list = id_list,
                    protein_sequence_list = seq_list,
                    outpath = out_path)
    print("Finished: ", time.ctime(), flush=True)


if __name__ == "__main__":
    main(fasta_in_path="/home/zhangzhiyuan/Desktop/vhPPI_Other_SOTA_Methods_multiTest/ZhouDatasetEmbed_new/humanProtSeq/",\
         out_path="/home/zhangzhiyuan/Desktop/vhPPI_Other_SOTA_Methods_multiTest/ZhouDatasetEmbed_new/humanProtPLM/",\
         num_thread=1)
