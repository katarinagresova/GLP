import papermill as pm
from genomic_benchmarks.data_check import list_datasets

VOCAB_SIZES = [64, 128, 256, 512, 1024, 2048]
KMERS = [2, 3, 4, 5, 6, 7]
DATASETS = list_datasets()

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