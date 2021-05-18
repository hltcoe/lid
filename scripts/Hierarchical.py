import random
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
import einops

# class EncoderDecoder(nn.Module):
#     """
#     A standard Encoder-Decoder architecture. Base for this and many 
#     other models.
#     """
#     def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.trg_embed = trg_embed
#         self.generator = generator
        
#     def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
#         """Take in and process masked src and target sequences."""
#         encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
#         return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
    
#     def encode(self, src, src_mask, src_lengths):
#         return self.encoder(self.src_embed(src), src_mask, src_lengths)
    
#     def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
#                decoder_hidden=None):
#         return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
#                             src_mask, trg_mask, hidden=decoder_hidden)

# class Generator(nn.Module):
#     """Define standard linear + softmax generation step."""
#     def __init__(self, hidden_size, vocab_size):
#         super(Generator, self).__init__()
#         self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

#     def forward(self, x):
#         return F.log_softmax(self.proj(x), dim=-1)


def pack_nested_sequence(unpacked, lengths):
    # (b, w, c) -> (b, c)
    max_length = lengths.max()
    nz = lengths.nonzero()
    word_count = nz.shape[0]
    words = torch.zeros(size=(word_count, max_length, unpacked.shape[-1]), dtype=torch.float32)
    word_lengths = torch.empty(size=(word_count,), dtype=int)
    for i, (s, w) in enumerate(nz):
        word_lengths[i] = lengths[s, w]
        words[i, :lengths[s, w], :] = unpacked[s, w, :lengths[s, w]]
    retval = torch.nn.utils.rnn.pack_padded_sequence(words, word_lengths, batch_first=True, enforce_sorted=False)    
    return (retval, nz)

def unpack_nested_sequence(packed, indices):
    unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
    retval = torch.zeros(size=(indices[:, 0].max() + 1, indices[:, 1].max() + 1, unpacked.shape[1], unpacked.shape[-1]), dtype=torch.float32)
    for i, ix in enumerate(indices):
        retval[ix[0], ix[1], :] = unpacked[i]
    return retval

class CharEncoder(torch.nn.Module):
    def __init__(self,
                 nchars,
                 char_emb_size,
                 hidden_size=8):
        super(CharEncoder, self).__init__()
        self.embeddings = torch.nn.Embedding(nchars + 2, char_emb_size, padding_idx=0)
        self.rnn = torch.nn.LSTM(input_size=char_emb_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size
        
    @property
    def out_size(self):
        return self.hidden_size * 2
                 
    def forward(self, chars, char_counts):
        echars = self.embeddings(chars)
        packed, indices = pack_nested_sequence(echars, char_counts)
        #unpacked = unpack_nested_sequence(packed, indices)
        #print(unpacked == echars)
        #sys.exit()
        out, (h_n, c_n) = self.rnn(packed)
        unpacked = unpack_nested_sequence(out, indices)
        #print(unpacked.shape)
        #torch.einsum("b w c -> b c w", chars)
        #print(chars.shape, char_counts.shape)
        #packed = torch.nn.utils.rnn.pack_padded_sequence(chars, char_counts, batch_first=True, enforce_sorted=False)
        #print(chars.shape, char_counts.max())
        #sys.exit()
        #return einops.rearrange(unpacked, "s w c r -> s w (c r)") #, unpacked) #self.embeddings(chars).sum(2)
        return torch.einsum("s w c r -> s w r", unpacked)
        
class WordEncoder(torch.nn.Module):
    def __init__(self,
                 nwords,
                 word_emb_size):
        super(WordEncoder, self).__init__()
        self.word_emb_size = word_emb_size
        self.embeddings = torch.nn.Embedding(nwords + 2, word_emb_size, padding_idx=0)
    def forward(self, words, word_counts):
        return self.embeddings(words)
    @property
    def out_size(self):
        return self.word_emb_size

class Classifier(torch.nn.Module):
    def __init__(self,
                 nlabels,
                 char_emb_size,
                 word_emb_size):
        super(Classifier, self).__init__()
        self.out_layer = torch.nn.Linear(char_emb_size + word_emb_size, nlabels)
        #self.rnn = torch.nn.LSTM(input_size=char_emb_size, hidden_size=32, batch_first=True, bidirectional=True)        
        
    def forward(self, reps, word_counts):
        #packed, indices = pack_nested_sequence(echars, char_counts)
        #retval = torch.nn.utils.rnn.pack_padded_sequence(reps, word_counts, batch_first=True, enforce_sorted=False)
        #print(word_counts)
        #print(reps.shape, self.out_layer)
        return self.out_layer(reps)
        #print(reps.shape, word_counts.shape)
        #print(word_counts)
    
class Model(torch.nn.Module):
    def __init__(self, 
                 nchars, 
                 nwords,
                 nlabels,
                 max_word_length,
                 max_sent_length,
                 char_emb_size=2,
                 word_emb_size=2,
                 word_rep_size=128,
                 sent_rep_size=512):
        super(Model, self).__init__()

        #self.char_embeddings = torch.nn.Embedding(nchars + 2, char_emb_size, padding_idx=0)
        self.char_enc = CharEncoder(nchars, char_emb_size)
        self.word_enc = WordEncoder(nwords, word_emb_size)
        self.classifier = Classifier(nlabels, self.char_enc.out_size, self.word_enc.out_size) #emb_size, word_emb_size)
        #self.max_word_length = max_word_length
        #self.max_sent_length = max_sent_length
        
    def forward(self, chars, char_counts, words, word_counts):
        chars = self.char_enc(chars, char_counts)
        words = self.word_enc(words, word_counts)
        #print(words.shape, chars.shape)
        comb = torch.cat([words, chars], dim=2)        
        out = self.classifier(comb, word_counts)
        #print(chars.shape, words.shape, comb.shape) #, out.shape)
        return out


        #self.word_encoder = torch.nn.TransformerEncoderLayer(d_model=char_emb_size,
        #                                                     nhead=8,
        #                                                     dim_feedforward=word_rep_size)
        #self.sentence_encoder = torch.nn.TransformerEncoderLayer(d_model=word_emb_size + word_rep_size,
        #                                                         nhead=8,
        #                                                         dim_feedforward=sent_rep_size)

def collate(items):
    max_word_count = max([len(item["words"]) for item in items])
    max_char_count = max(sum([[len(cs) for cs in item["characters"]] for item in items], []))
    chars = torch.zeros(size=(len(items), max_word_count, max_char_count), dtype=int)
    words = torch.zeros(size=(len(items), max_word_count), dtype=int)
    labels = torch.zeros(size=(len(items), max_word_count), dtype=int)
    char_counts = torch.zeros(size=(len(items), max_word_count), dtype=int)
    word_counts = torch.zeros(size=(len(items),), dtype=int)
    for i, item in enumerate(items):
        word_counts[i] = len(item["words"])
        for j, word in enumerate(item["words"]):
            words[i, j] = word
            cs = item["characters"][j]
            char_counts[i, j] = len(cs)
            for k, c in enumerate(cs):
                chars[i, j, k] = c
            labels[i, j] = item["labels"][j]
    return (chars, char_counts, words, word_counts, labels)


def train_model(train, dev, output, rest):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest="epochs", type=int, default=10)
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true", default=False)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=2)
    args, rest = parser.parse_known_args(rest)
    label_counts = {}
    char2id = {}
    word2id = {}
    label2id = {None : 0}
    max_word_length = 0
    max_sent_length = 0
    train_data = []
    for item in train:
        max_sent_length = max(max_sent_length, len(item["tokens"]))
        datum = {"words" : [], "characters" : [], "labels" : []}
        for token in item["tokens"]:
            word = token["form"]
            max_word_length = max(max_word_length, len(word))
            label = token["language"]
            label2id[label] = label2id.get(label, len(label2id))
            word2id[word] = word2id.get(word, len(word2id) + 2)
            datum["words"].append(word2id[word])
            datum["labels"].append(label2id[label])
            chars = []
            for char in word:
                char2id[char] = char2id.get(char, len(char2id) + 2)
                chars.append(char2id[char])
            datum["characters"].append(chars)
        train_data.append(datum)
    dev_data = []
    for item in train:
        max_sent_length = max(max_sent_length, len(item["tokens"]))
        datum = {"words" : [], "characters" : [], "labels" : []}
        for token in item["tokens"]:
            word = token["form"]
            max_word_length = max(max_word_length, len(word))
            label = token["language"]
            datum["words"].append(word2id.get(word, 1))
            datum["labels"].append(label2id.get(label, 0))
            chars = []
            for char in word:
                chars.append(char2id.get(char, 1))
            datum["characters"].append(chars)
        dev_data.append(datum)
    model = Model(len(char2id), len(word2id), len(label2id), max_word_length, max_sent_length)

    train_loader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=args.batch_size,
                              collate_fn=collate)
    dev_loader = DataLoader(dev_data,
                            shuffle=True,
                            batch_size=args.batch_size,
                            collate_fn=collate)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(args.epochs):
        for chars, char_counts, words, word_counts, labels in train_loader:
            optimizer.zero_grad()
            if args.use_gpu == True:
                pass
            selector = words != 0
            out = model(chars, char_counts, words, word_counts)
            sm = torch.nn.functional.log_softmax(out, dim=2).reshape(out.shape[0], out.shape[2], out.shape[1])
            loss = torch.masked_select(torch.nn.functional.nll_loss(sm, labels, reduction="none"), selector).mean()
            optimizer.step()
        dev_loss = 0.0
        model.train(False)
        for chars, char_counts, words, word_counts, labels in dev_loader:
            if args.use_gpu == True:
                pass
            selector = words != 0
            out = model(chars, char_counts, words, word_counts)
            sm = torch.nn.functional.log_softmax(out, dim=2).reshape(out.shape[0], out.shape[2], out.shape[1])
            dev_loss += torch.masked_select(torch.nn.functional.nll_loss(sm, labels, reduction="none"), selector).mean().detach()            
        model.train(True)
        print(epoch+1, dev_loss)
    return pickle.dumps((model, char2id, word2id, label2id))


def apply_model(model, test, args):
    model, char2id, word2id, label2id = pickle.loads(model)
    test_data = []
    for item in test:
        max_sent_length = max(max_sent_length, len(item["tokens"]))
        datum = {"words" : [], "characters" : [], "labels" : []}
        for token in item["tokens"]:
            word = token["form"]
            max_word_length = max(max_word_length, len(word))
            label = token["language"]
            datum["words"].append(word2id.get(word, 1))
            datum["labels"].append(label2id.get(label, 0))
            chars = []
            for char in word:
                chars.append(char2id.get(char, 1))
            datum["characters"].append(chars)
        test_data.append(datum)
    test_loader = DataLoader(test_data,
                             shuffle=False,
                             batch_size=args.batch_size,
                             collate_fn=collate)
    for chars, char_counts, words, word_counts, labels in train_loader:
        if args.use_gpu == True:
            pass
        selector = words != 0
        out = model(chars, char_counts, words, word_counts)
        sm = torch.nn.functional.log_softmax(out, dim=2).reshape(out.shape[0], out.shape[2], out.shape[1])
        loss = torch.masked_select(torch.nn.functional.nll_loss(sm, labels, reduction="none"), selector).mean()

    #for i in range(len(test)):
    #    test[i]["scores"] = model.probabilities(test[i]["sequence"])
    return test
