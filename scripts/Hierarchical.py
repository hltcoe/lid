import math
import random
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
import functools
import einops
import logging
from sklearn.metrics import f1_score
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class BasicScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):

    def __init__(self, optimizer, early_stop, patience, *argv, **argdict) -> None:
        self.early_stop = early_stop
        self.num_bad_epochs_for_early_stop = 0
        self.last_epoch = 0
        self.cooldown_counter = 0
        self.patience = patience
        super(BasicScheduler, self).__init__(optimizer, *argv, **argdict)
        
    def step(self, metrics):
        is_reduce_rate = False
        is_early_stop = False
        is_new_best = False
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.is_better(current, self.best):
            self.best = current
            is_new_best = True
            self.num_bad_epochs = 0
            self.num_bad_epochs_for_early_stop = 0
        else:
            self.num_bad_epochs += 1
            self.num_bad_epochs_for_early_stop += 1
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
            self.num_bad_epochs_for_early_stop = 0
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            is_reduce_rate = True
        if self.num_bad_epochs_for_early_stop > self.early_stop:
            is_early_stop = True
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        return (is_reduce_rate, is_early_stop, is_new_best)

        
class Classifier(torch.nn.Module):
    def __init__(self,
                 nlabels,
                 word_rep_size):
        super(Classifier, self).__init__()
        self.hidden_layer = torch.nn.Linear(word_rep_size, 1024)
        self.out_layer = torch.nn.Linear(1024, nlabels)
        self.norm = torch.nn.BatchNorm1d(1024)
        self.dropout = torch.nn.Dropout(0.6)
    def forward(self, reps):
        return self.out_layer(self.dropout(torch.nn.functional.relu(self.norm(self.hidden_layer(reps)))))


class CharAttention(torch.nn.Module):
    def __init__(self, input_dim, heads, output_dim, nlayers=2):
        super(CharAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.encoder = torch.nn.MultiheadAttention(self.input_dim, nlayers, dropout=0.6)
    def forward(self, char_reps, char_counts, word_counts):
        rchar_reps = einops.rearrange(char_reps, "b w c r -> c (b w) r")
        output, attn = self.encoder(rchar_reps, rchar_reps, rchar_reps) #, word_mask)
        output = einops.rearrange(output, "c (b w) r -> b w c r", b=char_reps.shape[0])
        return torch.mean(output, dim=2)


class WordAttention(torch.nn.Module):
    def __init__(self, input_dim, heads, output_dim, nlayers=2):
        super(WordAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.encoder = torch.nn.MultiheadAttention(self.input_dim, nlayers, dropout=0.6)
    def forward(self, word_reps, word_mask):
        rword_reps = einops.rearrange(word_reps, "b w r -> w b r")
        output, attn = self.encoder(rword_reps, rword_reps, rword_reps) #, word_mask)
        output = einops.rearrange(output, "w b r -> b w r")
        return output


class Model(torch.nn.Module):
    def __init__(self,
                 nchars,
                 nwords,
                 nlabels,
                 char_emb_size=32,
                 word_emb_size=64,
                 ):
        super(Model, self).__init__()    
        self.word_emb_size = word_emb_size
        self.char_emb_size = char_emb_size
        self.word_emb = torch.nn.Embedding(nwords, word_emb_size, padding_idx=0)
        self.char_emb = torch.nn.Embedding(nchars, char_emb_size, padding_idx=0)
        self.char_attention = CharAttention(char_emb_size, 4, char_emb_size)
        self.word_attention = WordAttention(word_emb_size + char_emb_size, 4, word_emb_size)
        self.sentence_attention = WordAttention(word_emb_size + char_emb_size, 4, word_emb_size)
        self.token_classifier = Classifier(nlabels, self.word_emb_size + self.char_emb_size)
        self.sentence_classifier = Classifier(nlabels, self.word_emb_size + self.char_emb_size)
    def forward(self, chars, char_counts, words, word_counts):
        embedded_words = self.word_emb(words) # b x w x d
        chars = self.char_emb(chars)
        character_derived_words = self.char_attention(chars, char_counts, word_counts)
        noncontextual_word_reps = torch.cat([embedded_words, character_derived_words], 2)
        word_mask = words > 0
        contextual_word_reps = self.word_attention(noncontextual_word_reps, word_mask)
        sentence_reps = self.sentence_attention(noncontextual_word_reps, word_mask)
        sentence_reps = (torch.sum(sentence_reps, 1).T / word_counts).T
        return (
            self.token_classifier(contextual_word_reps.reshape(-1, contextual_word_reps.shape[-1])),
            self.sentence_classifier(sentence_reps), #self.sentence_classifier(sentence_reps),
        )


def collate(items, label_count, use_gpu, i2w={}, i2l={}):
    max_word_count = max([len(item["words"]) for item in items])
    max_char_count = max([max([len(cs) for cs in item["characters"]]) for item in items])
    chars = torch.zeros(size=(len(items), max_word_count, max_char_count), dtype=int)
    words = torch.zeros(size=(len(items), max_word_count), dtype=int)
    labels = torch.zeros(size=(len(items), max_word_count), dtype=int)
    distributions = torch.zeros(size=(len(items), label_count))
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
            distributions[i, item["labels"][j]] += 1.0
    distributions = (distributions.T / torch.sum(distributions, 1)).T
    return [x.cuda() if use_gpu == True else x for x in [chars, char_counts, words, word_counts, labels, distributions]]


class Empty():
    pass


class Unknown():
    pass


def train_model(train, dev, output, rest):
    logging.info("invoked")
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_observations", dest="training_observations", type=int, default=10000000)
    parser.add_argument("--dev_interval", dest="dev_interval", type=int, default=1000)
    parser.add_argument("--patience", dest="patience", type=int, default=10)
    parser.add_argument("--early_stop", dest="early_stop", type=int, default=20)
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true", default=False)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=128)
    args, rest = parser.parse_known_args(rest)
    label_counts = {}
    char2id = {Empty() : 0, Unknown() : 1}
    word2id = {Empty() : 0, Unknown() : 1}
    label2id = {Empty() : 0, Unknown(): 1}
    max_word_length = 0
    max_sent_length = 0
    random.shuffle(train)
    random.shuffle(dev)
    train_data = []
    logging.info("invoked")
    for item in train:
        max_sent_length = max(max_sent_length, len(item["tokens"]))
        datum = {"words" : [], "characters" : [], "labels" : []}
        for token in item["tokens"]:
            word = token["form"] #.lower()
            max_word_length = max(max_word_length, len(word))
            label = token["language"]
            label2id[label] = label2id.get(label, len(label2id))
            word2id[word] = word2id.get(word, len(word2id))
            datum["words"].append(word2id[word])
            datum["labels"].append(label2id[label])
            chars = []
            for char in word:
                char2id[char] = char2id.get(char, len(char2id))
                chars.append(char2id[char])
            datum["characters"].append(chars)
        train_data.append(datum)
    id2label = {v : k for k, v in label2id.items()}
    id2word = {v : k for k, v in word2id.items()}
    id2char = {v : k for k, v in char2id.items()}
    dev_data = []
    for item in dev:
        max_sent_length = max(max_sent_length, len(item["tokens"]))
        datum = {"words" : [], "characters" : [], "labels" : []}
        for token in item["tokens"]:
            word = token["form"]
            max_word_length = max(max_word_length, len(word))
            label = token["language"]
            datum["words"].append(word2id.get(word, 1))
            datum["labels"].append(label2id.get(label, 1))
            chars = []
            for char in word:
                chars.append(char2id.get(char, 1))
            datum["characters"].append(chars)
        dev_data.append(datum)

    logging.info("creating model")
    model = Model(
        len(char2id),
        len(word2id), 
        len(label2id)
    )
    logging.info("{} {} {}".format(len(char2id), len(word2id), len(label2id)))
    if args.use_gpu == True:
        model.cuda()
    logging.info("loaders")
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=functools.partial(collate, label_count=len(label2id), use_gpu=args.use_gpu)
    )
    dev_loader = DataLoader(
        dev_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=functools.partial(collate, label_count=len(label2id), use_gpu=args.use_gpu)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = BasicScheduler(optimizer, early_stop=args.early_stop, patience=args.patience)
    best_dev_score = None
    best_state = None
    observation_count = 0
    observation_count_since_dev = 0
    logging.info("starting training")
    stop = False
    while stop != True:
        model.train(True)
        for chars, char_counts, words, word_counts, labels, distributions in train_loader:
            #print(chars.shape)
            observation_count += words.shape[0]
            observation_count_since_dev += words.shape[0]
            optimizer.zero_grad()
            out, sentence_out = model(chars, char_counts, words, word_counts)
            sm = torch.nn.functional.log_softmax(out, 1)
            ssm = torch.nn.functional.log_softmax(sentence_out, 1)
            _, sguesses = torch.max(ssm, dim=1)
            _, sgolds = torch.max(distributions, dim=1)
            train_sentence_f1 = f1_score(sguesses.cpu(), sgolds.cpu(), average="macro")
            sentence_loss = torch.nn.functional.kl_div(ssm, distributions, reduction="batchmean")
            _, cpreds = torch.max(sm, dim=1)
            clabels = labels.flatten()
            selector = clabels > 1
            word_loss = torch.masked_select(
                torch.nn.functional.nll_loss(sm, clabels, reduction="none"),
                selector
            ).mean()
            
            #sl = torch.masked_select(loss, selector)
            #train_loss = sl.mean().item()
            train_guesses = torch.masked_select(cpreds, selector).detach().cpu().tolist()
            train_golds = torch.masked_select(clabels, selector).detach().cpu().tolist()
            train_acc = len([a for a, b in zip(train_guesses, train_golds) if a == b]) / len(train_guesses)
            train_f1 = f1_score(train_guesses, train_golds, average="macro")
            train_loss = word_loss + sentence_loss
            #sl = sl.mean() + sentence_loss
            train_loss.backward()
            optimizer.step()
            # logging.info("After %d observations train loss/acc/tf1/sf1 at %.2f/%.2f/%.2f/%.2f", 
            #              observation_count, 
            #              train_loss.detach().cpu(),
            #              train_acc,
            #              train_f1,
            #              train_sentence_f1
            # )
            if observation_count_since_dev >= args.dev_interval or observation_count > args.training_observations:
                observation_count_since_dev = 0
                dev_loss = 0.0
                dev_guesses, dev_golds = [], []
                dev_sguesses, dev_sgolds = [], []
                model.train(False)
                # names clash with outer training loop, but it's OK here
                for chars, char_counts, words, word_counts, labels, distributions in dev_loader:
                    dev_out, dev_sentence_out = model(chars, char_counts, words, word_counts)
                    ssm = torch.nn.functional.log_softmax(dev_sentence_out, 1)
                    dev_sguesses += torch.max(ssm, dim=1)[1].detach().cpu().tolist()
                    dev_sgolds += torch.max(distributions, dim=1)[1].detach().cpu().tolist()
                    sentence_loss = torch.nn.functional.kl_div(ssm, distributions, reduction="batchmean")
                    dev_loss += sentence_loss.cpu().detach()
                    dev_dists = torch.nn.functional.log_softmax(dev_out, 1)
                    cpreds = torch.max(dev_dists, dim=1)[1]
                    clabels = labels.flatten()
                    dev_selector = clabels > 1
                    loss = torch.nn.functional.nll_loss(dev_dists, clabels, reduction="none")            
                    dev_loss += torch.masked_select(loss, dev_selector).mean().cpu().detach()
                    dev_guesses += torch.masked_select(cpreds, dev_selector).cpu().detach().tolist()
                    dev_golds += torch.masked_select(clabels, dev_selector).cpu().detach().tolist()
                model.train(True)
                dev_sentence_f1 = f1_score(dev_sguesses, dev_sgolds, average="macro")
                dev_acc = len([a for a, b in zip(dev_guesses, dev_golds) if a == b]) / len(dev_guesses)
                dev_f1 = f1_score(dev_guesses, dev_golds, average="macro")
                score = -dev_f1
                reduce_rate, early_stop, new_best = scheduler.step(score) #dev_loss)
                logging.info("After %d observations dev loss/acc/tf1/sf1 at %.2f/%.2f/%.2f/%.2f",
                             observation_count,
                             dev_loss,
                             dev_acc,
                             dev_f1,
                             dev_sentence_f1
                )
                if new_best or best_dev_score == None:
                    logging.info("New best dev F-score: %.3f", abs(score))
                    best_dev_score = score
                    best_state = {k : v.clone().detach().cpu() for k, v in model.state_dict().items()}

                if reduce_rate == True:
                    logging.info("Reducing learning rate after no improvement for %d dev evaluations and reverting to previous best state",
                                 scheduler.patience)
                    model.load_state_dict(best_state)
                if early_stop == True:
                    logging.info("Stopping early after no improvement for %d dev evaluations", scheduler.early_stop)
                    stop = True
                    break
                if observation_count > args.training_observations:
                    logging.info("Stopping after %d observations", observation_count)
                    stop = True
                    break

    model.load_state_dict(best_state)
    dev_loss = 0.0
    dev_guesses, dev_golds = [], []
    dev_sguesses, dev_sgolds = [], []
    model.train(False)
    for chars, char_counts, words, word_counts, labels, distributions in dev_loader:
        dev_out, dev_sentence_out = model(chars, char_counts, words, word_counts)
        ssm = torch.nn.functional.log_softmax(dev_sentence_out, 1)
        dev_sguesses += torch.max(ssm, dim=1)[1].cpu().detach().tolist()
        dev_sgolds += torch.max(distributions, dim=1)[1].cpu().detach().tolist()

        sentence_loss = torch.nn.functional.kl_div(ssm, distributions, reduction="batchmean")
        dev_loss += sentence_loss.cpu().detach()
        dev_dists = torch.nn.functional.log_softmax(dev_out, 1)
        cpreds = torch.max(dev_dists, dim=1)[1]
        clabels = labels.flatten()
        dev_selector = clabels > 1
        loss = torch.nn.functional.nll_loss(dev_dists, clabels, reduction="none")            
        dev_loss += torch.masked_select(loss, dev_selector).mean().cpu().detach()
        dev_guesses += torch.masked_select(cpreds, dev_selector).cpu().detach().tolist()
        dev_golds += torch.masked_select(clabels, dev_selector).cpu().detach().tolist()
    model.train(True)
    dev_acc = len([a for a, b in zip(dev_guesses, dev_golds) if a == b]) / len(dev_guesses)
    dev_f1 = f1_score(dev_guesses, dev_golds, average="macro")
    dev_sentence_f1 = f1_score(dev_sguesses, dev_sgolds, average="macro")
    reduce_rate, early_stop, new_best = scheduler.step(score) #dev_loss)
    logging.info("Final dev loss/acc/tf1/sf1 at %.2f/%.2f/%.2f/%.2f",
                 dev_loss,
                 dev_acc,
                 dev_f1,
                 dev_sentence_f1
             )

    return pickle.dumps((best_state, char2id, word2id, label2id))


def apply_model(model, test, rest):
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true", default=False)
    parser.add_argument("--apply_batch_size", dest="batch_size", type=int, default=128)
    args, rest = parser.parse_known_args(rest)

    best_state, char2id, word2id, label2id = pickle.loads(model)
    id2char = {v : k for k, v in char2id.items()}
    id2word = {v : k for k, v in word2id.items()}
    id2label = {v : k for k, v in label2id.items()}
    model = Model(
        len(char2id),
        len(word2id), 
        len(label2id)
    )
    model.load_state_dict(best_state)
    if args.use_gpu == True:
        model.cuda()
    model.train(False)
    test_data = []
    max_sent_length = 0
    max_word_length = 0

    instances = []
    for item in test:
        instances.append(item)
        max_sent_length = max(max_sent_length, len(item["tokens"]))
        datum = {"words" : [], "characters" : [], "labels" : []}
        for token in item["tokens"]:
            word = token["form"]
            max_word_length = max(max_word_length, len(word))
            label = token["language"]
            datum["words"].append(word2id.get(word, 1))
            datum["labels"].append(label2id.get(label, 1))
            chars = []
            for char in word:
                chars.append(char2id.get(char, 1))
            datum["characters"].append(chars)
        test_data.append(datum)

    test_loader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=functools.partial(collate, label_count=len(label2id), use_gpu=args.use_gpu)
    )

    i = 0
    test_guesses, test_golds = [], []
    chunks = []
    for chars, char_counts, words, word_counts, labels, distributions in test_loader:
        out, sentence_out = model(chars, char_counts, words, word_counts)
        sm = torch.nn.functional.log_softmax(out, 1)
        ssm = torch.nn.functional.log_softmax(sentence_out, 1)
        _, sguesses = torch.max(ssm, dim=1)
        _, sgolds = torch.max(distributions, dim=1)
        dev_sentence_f1 = f1_score(sguesses.cpu(), sgolds.cpu(), average="macro")
        sentence_loss = torch.nn.functional.kl_div(ssm, distributions, reduction="batchmean")
        sm = einops.rearrange(sm, "(s w) r -> s w r", s=words.shape[0])
        chunks.append((sm.detach().cpu().tolist(), ssm.detach().cpu().tolist()))
    return (chunks, label2id)
