from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_tokenizer
import pandas as pd

def get_tokenizer(name):
    if name == 'character':
        return CharacterTokenizer()
    elif name == 'subword':
        return SubwordTokenizer()
    elif name == 'kmer':
        return KmerTokenizer()
    else:
        raise ValueError('Tokenizer ' + name + ' not supported')

class CharacterTokenizer():
    def __init__(self, **kwargs):
        pass

    def __call__(self, items):
        if isinstance(items, str):
            return self.__tokenize_str(items)
        else:
            return (self.__tokenize_str(t) for t in items)

    def train(self, train_dset, **kwargs):
        pass

    def __tokenize_str(self, t):
        tokenized = list(t.replace("\n", ""))
        tokenized.append("<eos>")
        tokenized.insert(0, "<bos>")
        return tokenized

class SubwordTokenizer():
    def __init__(self, **kwargs):
        pass

    def __call__(self, items):
      
        if isinstance(items, str):
            return self.__tokenize_str([items])
        else:
            return (self.__tokenize_str([t]) for t in items)

    def train(self, train_dset, vocab_size, prefix = 'sample', **kwargs):
        
        self.vocab_size = vocab_size
        df = pd.DataFrame([x[0] for x in train_dset])
        df.to_csv('sample.csv', index=False, header=False)
        generate_sp_model('sample.csv', vocab_size=vocab_size, model_prefix=prefix)
        vocab_tokenizer = load_sp_model(prefix + ".model")
        self.tokenizer = sentencepiece_tokenizer(sp_model=vocab_tokenizer)

    def __tokenize_str(self, t):
        return list(self.tokenizer(t))[0]

class KmerTokenizer():
    def __init__(self, **kwargs):
        pass

    def __call__(self, items):
        if isinstance(items, str):
            return self.__tokenize_str(items)
        else:
            return (self.__tokenize_str(t) for t in items)

    def __tokenize_str(self, t):
        return self.__kmers(t, self.kmer)

    def __kmers(self, s, k):
        return [s[i:i + k] for i in range(0, len(s), k) if i + k <= len(s)]

    def train(self, train_dset, kmer = 4, **kwargs):
        self.kmer = kmer