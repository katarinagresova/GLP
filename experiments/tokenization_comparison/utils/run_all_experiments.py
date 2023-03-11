import papermill as pm
from genomic_benchmarks.data_check import list_datasets

VOCAB_SIZES = [64, 512, 2048]
DATASETS = [
   'demo_coding_vs_intergenomic_seqs',
   'human_nontata_promoters',
   'human_enhancers_cohn',
   'demo_human_or_worm',
   'human_enhancers_ensembl'
] # using only human datasets for now, because subword vocabulary was pretrained on human cDNA

for dataset in DATASETS:

   # execute with character tokenizer
   pm.execute_notebook(
      'torch_cnn.ipynb',
      '../torch_cnn_experiments/' + dataset + '_character.ipynb', 
      parameters = dict(DATASET=dataset, TOKENIZER='character')
   )

   for size in VOCAB_SIZES:
      # execute with subword tokenizer
      pm.execute_notebook(
         'torch_cnn.ipynb',
         '../torch_cnn_experiments/' + dataset + '_subword_' + str(size) + '.ipynb', 
         parameters = dict(DATASET=dataset, TOKENIZER='subword', VOCAB_SIZE=size)
      )

   for kmer in KMERS:
      # execute with kmer tokenizer
      pm.execute_notebook(
         'torch_cnn.ipynb',
         '../torch_cnn_experiments/' + dataset + '_kmer_' + str(kmer) + '.ipynb',
         parameters = dict(DATASET=dataset, TOKENIZER='kmer', KMER=kmer)
      )