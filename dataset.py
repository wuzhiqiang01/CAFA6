import time
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import torch.utils.data as data
from transformers import BertModel, BertTokenizer


num_labels = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = r"C:\Users\17539\Desktop\cafa\dataset\cafa-5-protein-function-prediction"
train_sequences_path = root  + r"\Train\train_sequences.fasta"
train_labels_path = root + r"\Train\train_terms.tsv"

tokenizer = BertTokenizer.from_pretrained(r"C:\Users\17539\Desktop\cafa\pretrained\Rostlab\prot_bert", do_lower_case=False )
model = BertModel.from_pretrained(r"C:\Users\17539\Desktop\cafa\pretrained\Rostlab\prot_bert").to(device)


def get_bert_embedding(
    sequence : str,
    len_seq_limit : int
):
    '''
    Function to collect last hidden state embedding vector from pre-trained ProtBERT Model

    INPUTS:
    - sequence (str) : protein sequence (ex : AAABBB) from fasta file
    - len_seq_limit (int) : maximum sequence lenght (i.e nb of letters) for truncation

    OUTPUTS:
    - output_hidden : last hidden state embedding vector for input sequence of length 1024
    '''
    sequence_w_spaces = ' '.join(list(sequence))
    encoded_input = tokenizer(
        sequence_w_spaces,
        truncation=True,
        max_length=len_seq_limit,
        padding='max_length',
        return_tensors='pt').to(device)
    output = model(**encoded_input)
    output_hidden = output['last_hidden_state'][:,0][0].detach().cpu().numpy()
    assert len(output_hidden)==1024
    return output_hidden

### COLLECTING FOR TRAIN SAMPLES :
print("Loading train set ProtBERT Embeddings...")
# fasta_train = SeqIO.parse(train_sequences_path, "fasta")
fasta_train = list(SeqIO.parse(train_sequences_path, "fasta"))[:10]  # 限制为前10条

ids_list = []
embed_vects_list = []
t0 = time.time()
checkpoint = 0
for item in tqdm(fasta_train):
    ids_list.append(item.id)
    # embed_vects_list.append(
    #     get_bert_embedding(sequence = item.seq, len_seq_limit = 1200))
    checkpoint+=1
    if checkpoint>=100:
        # df_res = pd.DataFrame(data={"id" : ids_list, "embed_vect" : embed_vects_list})
        np.save('/kaggle/working/train_ids.npy',np.array(ids_list))
        # np.save('/kaggle/working/train_embeddings.npy',np.array(embed_vects_list))
        checkpoint=0

np.save(r'C:\Users\17539\Desktop\cafa\dataset_emb\cafa-5-protein-function-prediction\train_ids.npy',np.array(ids_list))
np.save(r'C:\Users\17539\Desktop\cafa\dataset_emb\cafa-5-protein-function-prediction\train_embeddings.npy',np.array(embed_vects_list))
print('Total Elapsed Time:',time.time()-t0)

## COLLECTING FOR TEST SAMPLES :
##### SCRIPT FOR LABELS (TARGETS) VECTORS COLLECTING #####

#```python
print("GENERATE TARGETS FOR ENTRY IDS ("+str(num_labels)+" MOST COMMON GO TERMS)")
ids = np.load(r"C:\Users\17539\Desktop\cafa\dataset_emb\cafa-5-protein-function-prediction\train_ids.npy")
labels = pd.read_csv(train_labels_path, sep = "\t")

top_terms = labels.groupby("term")["EntryID"].count().sort_values(ascending=False)
labels_names = top_terms[:num_labels].index.values
train_labels_sub = labels[(labels.term.isin(labels_names)) & (labels.EntryID.isin(ids))]
id_labels = train_labels_sub.groupby('EntryID')['term'].apply(list).to_dict()

go_terms_map = {label: i for i, label in enumerate(labels_names)}
labels_matrix = np.empty((len(ids), len(labels_names)))

for index, id in tqdm(enumerate(ids)):
    id_gos_list = id_labels[id]
    temp = [go_terms_map[go] for go in labels_names if go in id_gos_list]
    labels_matrix[index, temp] = 1

labels_list = []
for l in range(labels_matrix.shape[0]):
    labels_list.append(labels_matrix[l, :])

labels_df = pd.DataFrame(data={"EntryID":ids, "labels_vect":labels_list})
labels_df.to_pickle(r"C:\Users\17539\Desktop\cafa\dataset_emb\cafa-5-protein-function-prediction\train_targets_top"+str(num_labels)+".pkl")
print("GENERATION FINISHED!")
#```


# https://www.kaggle.com/code/henriupton/proteinet-pytorch-ems2-t5-protbert-embeddings/notebook