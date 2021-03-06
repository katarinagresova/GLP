from Bio import SeqIO
from pathlib import Path
import random


def rm_tree(path):
    path = Path(path) # allow path to be a string
    assert path.is_dir() # make sure it`s a folder
    for p in reversed(list(path.glob('**/*'))): # iterate contents from leaves to root
        if p.is_file():
           p.unlink()
        elif p.is_dir():
            p.rmdir()


def prepare_folder_structure(root_dir, remove_if_exists=True, labels=['0', '1']):

    root_dir = Path(root_dir)

    if remove_if_exists:
        if root_dir.exists():
            rm_tree(root_dir)

    for label in labels:
        Path(root_dir / 'train' / label).mkdir(parents=True)
        Path(root_dir / 'valid' / label).mkdir(parents=True)


def split_fasta_to_txts(fasta_file, root_dir, label, train_ratio=0.7, kmer=0):

    root_dir = Path(root_dir)

    random.seed(42)
    with open(fasta_file, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            id = record.id
            seq = str(record.seq)
            if kmer > 0:
                seq = ' '.join(seq[i:i+kmer] for i in range(0,len(seq),kmer))

            filename = id + '.txt'
            if random.random() < train_ratio:
                file_path = Path(root_dir / 'train' / label / filename)
            else:
                file_path = Path(root_dir / 'valid' / label / filename)

            file_path.write_text(seq)