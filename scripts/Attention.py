import argparse
import pickle
import torch

class Model(torch.nn.Module):
    def __init__(self, 
                 nchars, 
                 nwords, 
                 max_word_length, 
                 max_sent_length, 
                 char_emb_size=8, 
                 word_emb_size=16, 
                 word_rep_size=128, 
                 sent_rep_size=512):
        super(Model, self).__init__()
        self.char_embeddings = torch.nn.Embedding(nchars + 1, char_emb_size, padding_idx=0)
        self.word_embeddings = torch.nn.Embedding(nwords + 1, word_emb_size, padding_idx=0)
        self.max_word_length = max_word_length
        self.max_sent_length = max_sent_length
        #self.word_encoder = torch.nn.TransformerEncoderLayer(d_model=char_emb_size,
        #                                                     nhead=8,
        #                                                     dim_feedforward=word_rep_size)
        #self.sentence_encoder = torch.nn.TransformerEncoderLayer(d_model=word_emb_size + word_rep_size,
        #                                                         nhead=8,
        #                                                         dim_feedforward=sent_rep_size)

def train_model(train, dev, output, rest):
    parser = argparse.ArgumentParser()
    args, rest = parser.parse_known_args(rest)
    label_counts = {}
    char2id = {}
    word2id = {}
    label2id = {}
    max_word_length = 0
    max_sent_length = 0
    for item in train:
        max_sent_length = max(max_sent_length, len(item["tokens"]))        
        for token in item["tokens"]:
            word = token["form"]
            max_word_length = max(max_word_length, len(word))
            label = token["language"]            
            label2id[label] = label2id.get(label, len(label2id))
            word2id[word] = word2id.get(word, len(word2id) + 1)
            for char in word:
                char2id[char] = char2id.get(char, len(char2id) + 1)
    model = Model(len(char2id), len(word2id), max_word_length, max_sent_length)

    #print(dev)
    # train_instances = train + dev
    # logging.info("Computing alphabet size...")
    # train_alphabet = set()
    # for i in train_instances:
    #     for v in i["sequence"]:
    #         train_alphabet.add(v)
    # logging.info("Training models...")
    # logging.info("Training %d-gram model on train set...", args.ngram_length)
    # model = Classifier(order=args.ngram_length, alphabet_size=len(train_alphabet))
    # for i in train_instances:
    #     model.train(i["label"], i["sequence"])
    return pickle.dumps(model)


def apply_model(model, test, args):
    model = pickle.loads(model)
    #for i in range(len(test)):
    #    test[i]["scores"] = model.probabilities(test[i]["sequence"])
    return test
