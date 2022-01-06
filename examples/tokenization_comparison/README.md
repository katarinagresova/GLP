Comparison of different tokenization techniques on fixed [CNN architecture](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/blob/main/src/genomic_benchmarks/models/torch.py) and multiple datasets from [genomic benchmarks package](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks). Written in pytorch.

Supported tokenizations: 
- character, 
- kmer (k=2,3,4,5,6 and 7) and 
- subword (vocab size = 64, 128, 256, 512, 1024 and 2048).

Future plans:
- try effect of data augmentation