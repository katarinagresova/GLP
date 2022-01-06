from collections import Counter
from torchtext.vocab import vocab
import torch
from torch.nn import ConstantPad1d

def build_vocab(dataset, tokenizer, use_padding):
    counter = Counter()
    for i in range(len(dataset)):
        counter.update(tokenizer(dataset[i][0]))
    builded_voc = vocab(counter)
    if use_padding:
        builded_voc.append_token("<pad>")
    builded_voc.insert_token("<unk>", 0)
    builded_voc.set_default_index(0)
    return builded_voc

def coll_factory(vocab, tokenizer, device="cpu", pad_to_length=None):
    def coll(batch):
        xs, ys = [], []

        for text, label in batch:
            ys.append(torch.tensor([label], dtype=torch.float32))
            x = torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)
            if pad_to_length != None:
                PAD_IDX = vocab["<pad>"]
                pad = ConstantPad1d((0, pad_to_length - len(x)), PAD_IDX)
                x = torch.tensor(pad(x), dtype=torch.long)
            xs.append(x)

        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs.to(device), ys.to(device)

    return coll

def check_seq_lengths(dataset, tokenizer):
    # Compute length of the longest sequence
    max_tok_len = max([len(tokenizer(dataset[i][0])) for i in range(len(dataset))])
    print("max_tok_len ", max_tok_len)
    same_length = [len(tokenizer(dataset[i][0])) == max_tok_len for i in range(len(dataset))]
    if not all(same_length):
        print("not all sequences are of the same length")

    return max_tok_len

def check_config(config):
    control_config = {
        "use_padding": bool,
        "run_on_gpu": bool,
        "dataset": str,
        "number_of_classes": int,
        "dataset_version": int,
        "force_download": bool,
        "epochs": int,
        "embedding_dim": int,
        "batch_size": int,
    }

    for key in config.keys():
        assert isinstance(config[key], control_config[key]), '"{}" in config should be of type {} but is {}'.format(
            key, control_config[key], type(config[key])
        )