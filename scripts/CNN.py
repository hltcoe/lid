import argparse
import torch

class CNNModel(torch.nn.Module):
    def __init__(self, nchars, char_emb_size, max_width):
        pass
    def forward(self, x):
        pass

def train_model(train, dev, output, rest):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest="epochs", type=int, default=10)
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true", default=False)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=128)
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
    if args.use_gpu == True:
        model.cuda()

    train_loader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=args.batch_size,
                              collate_fn=functools.partial(collate, use_gpu=args.use_gpu))
    dev_loader = DataLoader(dev_data,
                            shuffle=True,
                            batch_size=args.batch_size,
                            collate_fn=functools.partial(collate, use_gpu=args.use_gpu))

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
