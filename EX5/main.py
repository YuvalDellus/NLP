import nltk
from nltk.corpus import dependency_treebank
from sklearn.model_selection import train_test_split
from itertools import permutations
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx
from util import *
import numpy as np
import random

HAS_DEPENDANTS = ""
ITERATIONS = 2
TEST_SIZE = 0.1
EMBEDDING_DIM = 300


class Arc:
    """
    class to hold necessary information about each arc
    """

    def __init__(self, head, tail, weight):
        self.head = head
        self.tail = tail
        self.weight = weight


def index_corpus(sents):
    """
    indexes every word and every tag from the corpus to a specified index
    :param sents: the training set sentences
    :return: word_dic that holds every
    """
    words = set()
    tags = set()
    for sent in sents:
        for i in range(len(sent.nodes)):
            words.add(sent.nodes[i]['word'])
            tags.add(sent.nodes[i]['tag'])
    words = list(words)
    tags = list(tags)
    word_dic = {words[i]: i for i in range(len(words))}
    tag_dic = {tags[i]: i for i in range(len(tags))}
    weights_size = len(words) ** 2 + len(tags) ** 2
    return word_dic, tag_dic, weights_size


def feature_function(word_1, word_2, word_dic, tag_dic):
    """
    returns the index of the pairs (word_1, word_2) and (tag_1, tag_2) in the feature vector
    :param word_1: first (word, tag) tuple
    :param word_2: second (word, tag) tuple
    :param word_dic: word to index mapping
    :param tag_dic: tag to index mapping
    :return:
    """
    ind_1, ind_2 = -1, -1
    if word_1[0] in word_dic and word_2[0] in word_dic:
        ind_1 = (word_dic[word_1[0]] * len(word_dic)) + word_dic[word_2[0]]
    if word_1[1] in tag_dic and word_2[1] in tag_dic:
        ind_2 = (len(word_dic) * len(word_dic)) + (tag_dic[word_1[1]] * len(tag_dic)) + tag_dic[word_2[1]]
    return ind_1, ind_2


def get_weight(indices, w):
    """
    computes the weight of a given arc based on its' indices in the feature vector
    :param indices: indices in the feature vector
    :param w: current weight
    :return: arc's weight
    """
    weight = 0
    if indices[0] != -1:
        weight += w[indices[0]]
    if indices[1] != -1:
        weight += w[indices[1]]
    return weight


def get_arcs_for_sent(sent, word_dic, tag_dic, w):
    """
    returns all possible arcs and their weight for a given sentence
    :param sent:
    :param word_dic: word to index mapping
    :param tag_dic: tag to index mapping
    :param w: current weight
    :return: arcs for the given sentence
    """
    all_arcs = list(permutations(range(len(sent.nodes)), 2))
    arcs = []
    for arc in all_arcs:
        word_0, tag_0 = sent.nodes[arc[0]]['word'], sent.nodes[arc[0]]['tag']
        word_1, tag_1 = sent.nodes[arc[1]]['word'], sent.nodes[arc[1]]['tag']
        indices = feature_function((word_0, tag_0), (word_1, tag_1), word_dic, tag_dic)
        weight = -get_weight(indices, w)
        arcs.append(Arc(arc[0], arc[1], weight))
    return arcs


def update_vector(vector, indices):
    """
    updates given vector indices
    :param vector:
    :param indices:
    :return:
    """
    if indices[0] != -1:
        vector[indices[0]] += 1
    if indices[1] != -1:
        vector[indices[1]] += 1
    return vector


def get_update(sent, max_tree, weights_size, word_dic, tag_dic):
    """
    computes the vector needed to update the weight
    :param sent: current sentence
    :param max_tree: tree found by Chu Lio Edmonds
    :param weights_size:
    :param word_dic: word to index mapping
    :param tag_dic: tag to index mapping
    :return: the vector needed to update the weight
    """
    w_max = np.zeros(weights_size)
    w_gold = np.zeros(weights_size)
    for i in range(len(sent.nodes)):
        word_1 = sent.nodes[i]['word']
        tag_1 = sent.nodes[i]['tag']
        if HAS_DEPENDANTS in sent.nodes[i]['deps']:
            for ind in sent.nodes[i]['deps'][HAS_DEPENDANTS]:
                word_2, tag_2 = sent.nodes[ind]['word'], sent.nodes[ind]['tag']
                indices = feature_function((word_1, tag_1), (word_2, tag_2), word_dic, tag_dic)
                w_gold = update_vector(w_gold, indices)
    for arc in max_tree:
        head_word, head_tag = sent.nodes[max_tree[arc].head]['word'], sent.nodes[max_tree[arc].head]['tag']
        tail_word, tail_tag = sent.nodes[max_tree[arc].tail]['word'], sent.nodes[max_tree[arc].tail]['tag']
        indices = feature_function((head_word, head_tag), (tail_word, tail_tag), word_dic, tag_dic)
        w_max = update_vector(w_max, indices)
    return w_gold - w_max


def averaged_perceptron(iters, sents, word_dic, tag_dic, weights_size):
    """
    averaged perceptron algorithm implementation
    :param iters: number of iterations
    :param sents: training sentences
    :param word_dic: mapping between a word and its' index
    :param tag_dic: mapping between a tag and its' index
    :param weights_size
    :return: weights for inference stage
    """
    W = np.zeros(weights_size)
    w = np.zeros(weights_size)
    for i in range(iters):
        random.shuffle(sents)  # shuffles the training set
        for sent in sents:
            arcs = get_arcs_for_sent(sent, word_dic, tag_dic, w)
            max_tree = min_spanning_arborescence_nx(arcs, 0)
            W += w
            w = w + get_update(sent, max_tree, weights_size, word_dic, tag_dic)
    return W / (iters * len(sents))


def get_arcs_from_sent_tree(sent):
    """
    return arcs from a given sentence
    :param sent:
    :return: set of the sentence's arcs
    """
    arcs = set()
    for i in range(len(sent.nodes)):
        if HAS_DEPENDANTS in sent.nodes[i]['deps']:
            for ind in sent.nodes[i]['deps'][HAS_DEPENDANTS]:
                arc = (i, ind)
                arcs.add(arc)
    return arcs


def get_arcs_from_tree(tree):
    """
    :param tree:
    :return: set of the tree arcs
    """
    arcs = set()
    for arc in tree.keys():
        arcs.add((tree[arc].head, tree[arc].tail))
    return arcs


def score(sents, word_dic, tag_dic, w):
    """
    computes the accuracy of the weights found by the averaged perceptron algorithm
    :param sents: test sentences
    :param word_dic: mapping between a word and its' index
    :param tag_dic: mapping between a tag and its' index
    :param w: weight vector
    :return: accuracy
    """
    avg_acc = 0
    for sent in sents:
        arcs = get_arcs_for_sent(sent, word_dic, tag_dic, w)
        max_tree = min_spanning_arborescence_nx(arcs, 0)
        tree_arcs = get_arcs_from_tree(max_tree)
        sent_arcs = get_arcs_from_sent_tree(sent)
        equal_arcs = len(tree_arcs.intersection(sent_arcs))
        avg_acc += (equal_arcs / len(sent.nodes))
    return avg_acc / len(sents)


"""
#####################################
##### bonus implementation ##########
#####################################
"""


def averaged_perceptron_bonus(iters, sents, w_2_vec, tag_dic, weights_size):
    """
    averaged perceptron algorithm implementation
    :param iters: number of iterations
    :param sents: training sentences
    :param w_2_vec: mapping between a word and its' embedding vector
    :param tag_dic: mapping between a tag and its' index
    :param weights_size
    :return: weights for inference stage
    """
    W = np.zeros(weights_size)
    w = np.zeros(weights_size)
    for i in range(iters):
        random.shuffle(sents)  # shuffles the training set
        for j, sent in enumerate(sents):
            arcs = get_arcs_for_sent_bonus(sent, w_2_vec, w, tag_dic)
            max_tree = min_spanning_arborescence_nx(arcs, 0)
            W += w
            w = w + get_update_bonus(sent, max_tree, weights_size, w_2_vec, tag_dic)
    return W / (iters * len(sents))


def score_bonus(sents, w_2_vec, w, tag_dic):
    """
    computes the accuracy of the weights found by the averaged perceptron algorithm
    :param sents: test sentences
    :param w_2_vec: mapping between a word and its' embedding vector
    :param w: weight vector
    :param tag_dic: mapping between a tag and its' index
    :return: accuracy
    """
    avg_acc = 0
    for sent in sents:
        arcs = get_arcs_for_sent_bonus(sent, w_2_vec, w, tag_dic)
        max_tree = min_spanning_arborescence_nx(arcs, 0)
        tree_arcs = get_arcs_from_tree(max_tree)
        sent_arcs = get_arcs_from_sent_tree(sent)
        equal_arcs = len(tree_arcs.intersection(sent_arcs))
        avg_acc += (equal_arcs / len(sent.nodes))
    return avg_acc / len(sents)


def get_update_bonus(sent, max_tree, weights_size, w_2_vec, tag_dic):
    """
    computes the vector needed to update the weight
    :param sent: current sentence
    :param max_tree: tree found by Chu Lio Edmonds
    :param weights_size:
    :param w_2_vec: mapping between a word and its' embedding vector
    :param tag_dic: tag to index mapping
    :return: the vector needed to update the weight
    """
    w_max = np.zeros(weights_size)
    w_gold = np.zeros(weights_size)
    for i in range(len(sent.nodes)):
        word_1, tag_1 = sent.nodes[i]['word'], sent.nodes[i]['tag']
        if HAS_DEPENDANTS in sent.nodes[i]['deps']:
            for ind in sent.nodes[i]['deps'][HAS_DEPENDANTS]:
                word_2, tag_2 = sent.nodes[ind]['word'], sent.nodes[ind]['tag']
                feature_vec = pair_to_embedding((word_1, word_2), w_2_vec, (tag_1, tag_2), tag_dic)
                w_gold += feature_vec
    for arc in max_tree:
        head_word, head_tag = sent.nodes[max_tree[arc].head]['word'], sent.nodes[max_tree[arc].head]['tag']
        tail_word, tail_tag = sent.nodes[max_tree[arc].tail]['word'], sent.nodes[max_tree[arc].tail]['tag']
        feature_vec = pair_to_embedding((head_word, tail_word), w_2_vec, (head_tag, tail_tag), tag_dic)
        w_max += feature_vec
    return w_gold - w_max


def get_arcs_for_sent_bonus(sent, w_2_vec, w, tag_dic):
    """
    returns all possible arcs and their weight for a given sentence
    :param sent:
    :param w_2_vec: mapping between a word and its' embedding vector
    :param w: current weight
    :param tag_dic: tag to index mapping
    :return: arcs for the given sentence
    """
    all_arcs = list(permutations(range(len(sent.nodes)), 2))
    arcs = []
    for arc in all_arcs:
        word_0, tag_0 = sent.nodes[arc[0]]['word'], sent.nodes[arc[0]]['tag']
        word_1, tag_1 = sent.nodes[arc[1]]['word'], sent.nodes[arc[1]]['tag']
        feature_vec = pair_to_embedding((word_0, word_1), w_2_vec, (tag_0, tag_1), tag_dic)
        weight = - np.dot(feature_vec, w)
        arcs.append(Arc(arc[0], arc[1], weight))
    return arcs


def pair_to_embedding(words, w_2_vec, tags, tag_dic):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param words: arc words to embed
    :param w_2_vec: mapping between a word and its' embedding vector
    :param tags: POS tagging of the input words
    :param tag_dic: tag to index mapping
    :return: arc's feature vector
    """
    vec_1 = np.zeros(EMBEDDING_DIM)
    vec_2 = np.zeros(EMBEDDING_DIM)
    if words[0] in w_2_vec:
        vec_1 = w_2_vec[words[0]]
    if words[1] in w_2_vec:
        vec_2 = w_2_vec[words[1]]
    ind = (tag_dic[tags[0]] * len(tag_dic)) + tag_dic[tags[1]]
    vec_3 = np.zeros(len(tag_dic) ** 2)
    vec_3[ind] = 1
    feature_vec = np.concatenate((vec_1, vec_2, vec_3))
    return feature_vec


if __name__ == '__main__':
    parsed_sents = dependency_treebank.parsed_sents()
    train_sents, test_sents = train_test_split(parsed_sents, test_size=TEST_SIZE)
    word_dic, tag_dic, weights_size = index_corpus(train_sents)
    w = averaged_perceptron(ITERATIONS, train_sents, word_dic, tag_dic, weights_size)
    s = score(test_sents, word_dic, tag_dic, w)
    print(s)

    # bonus #

    words = list(word_dic.keys())
    w_2_vec = create_or_load_slim_w2v(words, False)
    weights_size_2 = 2 * EMBEDDING_DIM + len(tag_dic) ** 2
    w = averaged_perceptron_bonus(ITERATIONS, train_sents, w_2_vec, tag_dic, weights_size_2)
    s = score_bonus(test_sents, w_2_vec, w, tag_dic)
    print(s)