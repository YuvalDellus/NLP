import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import matplotlib.pyplot as plt

# import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"

LOG_LINEAR_EPOCHS = 20
LOG_LINEAR_RATE = 0.01
LSTM_EPOCHS = 4
LSTM_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 64
LSTM_HIDDEN_DIM = 100
DROPOUT = 0.5


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    avg_vec = np.zeros(embedding_dim)
    dev_factor = 0
    for word in sent.text:
        if word in word_to_vec:
            vec = word_to_vec[word]
            avg_vec += vec
            dev_factor += 1
    if dev_factor == 0:
        return avg_vec
    return avg_vec / dev_factor


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    one_hots = torch.tensor(np.zeros(len(word_to_ind)))
    dev_fac = len(sent.text)
    for word in sent.text:
        ind = word_to_ind[word]
        one_hot = get_one_hot(len(word_to_ind), ind)
        one_hots += one_hot

    return one_hots / dev_fac


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word_dic = dict()
    for ind, word in enumerate(words_list):
        word_dic[word] = ind
    return word_dic


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    mat = []

    for i in range(seq_len):
        vec = np.zeros(embedding_dim)
        if i < len(sent.text):
            if sent.text[i] in word_to_vec:
                vec = word_to_vec[sent.text[i]]
        mat.append(vec)

    return np.array(mat)


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()  # Important to call Modules constructor!!
        self.name = W2V_SEQUENCE
        self.drop = nn.Dropout(p=dropout)
        self.bi_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                               bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)
        return

    def forward(self, text):
        text = text.permute(1, 0, 2)
        _, (h_n, c_n) = self.bi_LSTM(text.float())
        h_n = h_n[0, :, :] + h_n[1, :, :]
        out = self.drop(h_n)
        out = self.linear(out)
        return out

    def predict(self, text):
        text = text.permute(1, 0, 2)
        _, (h_n, c_n) = self.bi_LSTM(text.float())
        h_n = h_n[0, :, :] + h_n[1, :, :]
        out = self.linear(h_n)
        out = nn.Sigmoid()(out)
        return out.round()


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()  # Important to call Modules constructor!!
        self.name = ONEHOT_AVERAGE
        self.linear1 = nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(self, x):
        h1 = self.linear1(x.float())
        return h1

    def predict(self, x):
        h1 = self.linear1(x.float())
        out = nn.Sigmoid()(h1)
        out = out.round()
        return out


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    correct = 0
    total = 0
    for i in range(len(preds)):
        if preds[i] == y[i]:
            correct += 1
        total += 1
    return correct / total


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    model.train()
    total_loss = 0
    total_acc = 0
    batches = 0
    for batch in data_iterator:
        batches += 1
        local_batch, local_labels = batch[0], batch[1]
        optimizer.zero_grad()
        pred = model(local_batch)
        target = local_labels.unsqueeze(1)
        loss = criterion(pred, target)
        pred = nn.Sigmoid()(pred)
        acc = binary_accuracy(pred.round(), target)
        total_loss += loss.item()
        total_acc += acc
        loss.backward()
        optimizer.step()
    return total_loss / batches, total_acc / batches


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    batches = 0
    for local_batch, local_labels in data_iterator:
        batches += 1
        pred = model(local_batch)
        target = local_labels.unsqueeze(1)
        loss = criterion(pred, target.float())
        pred = nn.Sigmoid()(pred)
        acc = binary_accuracy(pred.round(), local_labels)
        total_loss += loss.item()
        total_acc += acc
    return total_loss / batches, total_acc / batches


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    model.eval()
    preds = None
    for local_batch, local_labels in data_iter:
        pred = model.predict(local_batch.float())
        if preds is None:
            preds = pred.detach().numpy().flatten()
        else:
            preds = np.concatenate([preds, pred.detach().numpy().flatten()])
    return preds


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(n_epochs):
        t_loss, t_acc = train_epoch(model, data_manager.torch_iterators[TRAIN], optimizer, criterion)
        train_loss.append(t_loss)
        train_acc.append(t_acc)
        v_loss, v_acc = evaluate(model, data_manager.torch_iterators[VAL], criterion)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
        print(f"EPOCH {epoch}, train loss = {t_loss}, train accuracy = {t_acc}, val loss = {v_loss}, val accuracy ="
              f" {v_acc}")

    train_loss = np.array(train_loss)
    train_acc = np.array(train_acc)
    val_loss = np.array(val_loss)
    val_acc = np.array(val_acc)
    np.save(f"results\\train_loss_{model.name}", train_loss)
    np.save(f"results\\train_acc_{model.name}", train_acc)
    np.save(f"results\\val_loss_{model.name}", val_loss)
    np.save(f"results\\val_acc_{model.name}", val_acc)


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manger = DataManager(batch_size=BATCH_SIZE)
    arg = data_manger.sent_func_kwargs
    model = LogLinear(len(arg['word_to_ind']))
    model.double()
    train_model(model, data_manger, LogLinear, LOG_LINEAR_RATE, WEIGHT_DECAY)
    torch.save(model.state_dict(), f"models\\{model.name}")
    return


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manger = DataManager(W2V_AVERAGE, embedding_dim=W2V_EMBEDDING_DIM, batch_size=BATCH_SIZE)
    model = LogLinear(W2V_EMBEDDING_DIM)
    model.name = W2V_AVERAGE
    model.double()
    train_model(model, data_manger, LogLinear, LOG_LINEAR_RATE, WEIGHT_DECAY)
    torch.save(model.state_dict(), f"models\\{model.name}")
    return


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manger = DataManager(W2V_SEQUENCE, embedding_dim=W2V_EMBEDDING_DIM, batch_size=BATCH_SIZE)
    model = LSTM(W2V_EMBEDDING_DIM, LSTM_HIDDEN_DIM, dropout=DROPOUT, n_layers=1)
    train_model(model, data_manger, LSTM_EPOCHS, LSTM_RATE, WEIGHT_DECAY)
    torch.save(model.state_dict(), f"models\\{model.name}")
    return

def make_plot_train_test(x_range, train_values, validation_values, x_label, y_label, title, save_filepath):
    """
    Plots Train Validation values over epochs
    :param x_range: x axis range
    :param train_values: train set values
    :param validation_values: validation set values
    :param x_label: x axis label
    :param y_label: y axis label
    :param title: graph title
    :param save_filepath: path to save the graph
    """
    with plt.style.context('ggplot'):
        plt.figure()
        plt.plot(x_range, train_values, label='Train')
        plt.plot(x_range, validation_values, label='Validation')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_filepath, dpi=300)
        plt.show()


def get_special_set_results(pred, y, ind):
    """
    returns special set's accuracy
    :param pred: model's prediction
    :param y: true labels
    :param ind: special set indices in the set
    """
    special_pred = pred[ind]
    special_y = y[ind]
    acc = binary_accuracy(special_pred, special_y)
    return acc


def get_models():
    """
    returns the models architectures
    """
    one_hot_log = LogLinear(DataManager().get_input_shape()[0])
    avg_log = LogLinear(W2V_EMBEDDING_DIM)
    avg_log.name = W2V_AVERAGE
    lstm = LSTM(W2V_EMBEDDING_DIM, LSTM_HIDDEN_DIM, dropout=DROPOUT, n_layers=1)
    return [one_hot_log, avg_log, lstm]


def get_results(mode):
    """
    returns train and validation results for a given mode
    :param mode: model mode
    """
    train_loss = np.load(f"results\\train_loss_{mode}.npy")
    train_acc = np.load(f"results\\train_acc_{mode}.npy")
    val_loss = np.load(f"results\\val_loss_{mode}.npy")
    val_acc = np.load(f"results\\val_acc_{mode}.npy")
    return train_loss, train_acc, val_loss, val_acc


def test_results(model):
    """
    returns results over test set and special subsets
    :param model: model to predict with
    """
    model.load_state_dict(torch.load(f"models\\{model.name}"))
    model.eval()
    data_manager = DataManager(model.name, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    if model.name == ONEHOT_AVERAGE:
        data_manager = DataManager(model.name, batch_size=BATCH_SIZE)
    test_labels = data_manager.get_labels(TEST)
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc = evaluate(model, data_manager.torch_iterators[TEST], criterion)
    pred = get_predictions_for_data(model, data_manager.torch_iterators[TEST])
    negated = data_loader.get_negated_polarity_examples(data_manager.torch_datasets[TEST].data)
    rare = data_loader.get_rare_words_examples(data_manager.torch_datasets[TEST].data, data_manager.sentiment_dataset)
    negated_acc = get_special_set_results(pred, test_labels, negated)
    rare_acc = get_special_set_results(pred, test_labels, rare)
    return test_loss, test_acc, negated_acc, rare_acc


if __name__ == '__main__':
    print("LOG LINEAR MODEL")
    train_log_linear_with_one_hot()
    print("---------------------------------\n")
    print("LOG LINEAR MODEL W2V")
    train_log_linear_with_w2v()
    print("---------------------------------\n")
    print("LSTM MODEL")
    train_lstm_with_w2v()
    models = get_models()
    for model in models:
        train_loss, train_acc, val_loss, val_acc = get_results(model.name)
        make_plot_train_test(range(len(train_loss)), train_loss, val_loss, "EPOCHS", "LOSS", f"{model.name} LOSS",
                             f"figs\\Loss_{model.name}")
        make_plot_train_test(range(len(train_loss)), train_acc, val_acc, "EPOCHS", "ACCURACY", f"{model.name} ACCURACY",
                             f"figs\\Accuracy_{model.name}")
        test_loss, test_acc, negated_acc, rare_acc = test_results(model)
        print(f"accuracy on test_set for {model.name} - {test_acc}")
        print(f"loss on test_set for {model.name} - {test_loss}")
        print(f"accuracy on negated test_set for {model.name} - {negated_acc}")
        print(f"accuracy on rare test_set for {model.name} - {rare_acc}")
