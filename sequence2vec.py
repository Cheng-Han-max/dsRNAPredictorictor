import Bio
import gensim
import numpy as np
import pandas as pd
from Bio import SeqIO


NULL_vec = np.zeros((100))

def get_kmer(dnaSeq, K):
    dnaSeq = dnaSeq.upper()
    l = len(dnaSeq)
    return [dnaSeq[i:i+K] for i in range(0, l-K+1, K)]

seq_file = '../data/train/train_data.fasta'
seq_records = list(Bio.SeqIO.parse(seq_file, 'fasta'))
embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format("../data/Tcas5.2_embedding.w2v")
records_num = len(seq_records)

for K in range(3, 9):
    code_file ='../training_vec/'+str(K)+'mer_train_datavec.csv'
    seqid=1
    for seq_record in seq_records:
        dnaSeq = str(seq_record.seq)
        kmers = get_kmer(dnaSeq, K)
        code = []
        for kmer in kmers:
            if ('n' not in kmer) and ('N' not in kmer):
                code.append(embedding_matrix[kmer])
            else:
                code.append(NULL_vec)
        array = np.array(code)
        ave = array.sum(axis=0)
        ave = pd.DataFrame(ave).T
        id = pd.DataFrame([seqid]).T
        ave = pd.concat([id, ave], axis=1, ignore_index=True)
        ave.to_csv(code_file, index=False, mode='a', header=False)

        print('the %dth seq is done' % seqid)
        seqid += 1


