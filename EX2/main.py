from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from nltk.corpus import brown


def load_data():
    """
    loads brown corpus and divides it to words and tags
    :return: split train and test sets
    """
    tagged_sentences = brown.tagged_sents(categories='news')
    X = []
    Y = []
    for sent in tagged_sentences:
        x = []
        y = []
        for word in sent:
            x.append(word[0])
            y.append(word[1])
        X.append(x)
        Y.append(y)
    return train_test_split(X, Y, test_size=0.1, shuffle=False)


def most_likely_tag(X, Y):
    """
    :param X: words
    :param Y: tags
    :return: dictionary of dictionaries with words as key and tags as values and keys and
             the number of tags per words as values.
    """
    dic = dict()
    for sen in range(len(X)):
        for word in range(len(X[sen])):
            tag = Y[sen][word]
            if X[sen][word] not in dic:
                word_dic = dict()
                word_dic[tag] = 1
                dic[X[sen][word]] = word_dic
            else:
                word_dic = dic.get(X[sen][word])
                if tag in word_dic:
                    word_dic[tag] = word_dic[tag] + 1
                else:
                    word_dic[tag] = 1
    return dic


def tag_words(words, tags, dic):
    """
    this baseline model tags the test set
    :param words: words to tags
    :param tags: tags set
    :param dic: number of tags per word dictionary
    :return: error rates
    """
    unknown_words, unknown_words_correct = 0, 0
    known_words, known_words_correct = 0, 0
    for sen in range(len(words)):
        for word in range(len(words[sen])):
            tag = tags[sen][word]
            if words[sen][word] in dic:
                word_tags = dic.get(words[sen][word])
                max_key = max(word_tags, key=word_tags.get)
                if max_key == tag:
                    known_words_correct += 1
                known_words += 1
            else:
                unknown_words += 1
                if tag == 'NN':
                    unknown_words_correct += 1
    unknown_error = 1 - unknown_words_correct / unknown_words
    known_error = 1 - known_words_correct / known_words
    total_error = 1 - (known_words_correct + unknown_words_correct) / (known_words + unknown_words)
    return total_error, known_error, unknown_error


def emission_dic(X, Y):
    """
    calculates emission probabilities
    :param X: words train set
    :param Y: true tags for the train set
    :return: emission probabilities as a dictionary
    """
    dic = dict()
    word_set = set()
    for sen_index, sentence in enumerate(X):
        for word_index, word in enumerate(sentence):
            word_set.add(word)
            tag = Y[sen_index][word_index]
            if tag not in dic:
                tag_dic = dict()
                tag_dic[word] = 1
                tag_dic["sum"] = 1
                dic[tag] = tag_dic
            else:
                tag_dic = dic.get(tag)
                if word in tag_dic:
                    tag_dic[word] += 1
                else:
                    tag_dic[word] = 1
                tag_dic["sum"] += 1
    return dic, word_set


def emission(dic, word, tag, words_set, is_first):
    """
    :param dic: emission probabilities
    :param word: word to emit
    :param tag: tag to emit
    :param words_set: all words in the train set
    :return: emission probability
    """
    if word not in words_set:
        return 1e-24
    dic_tag = dic.get(tag)
    if tag in dic:
        if word in dic_tag:
            word_count = dic_tag.get(word)
            total_words = dic_tag.get("sum")
            return word_count / total_words
    return 1e-24


def transition_dic(tags):
    """
    calculates transition probabilities over the tags in the train set.
    :param tags: true tags of the train set
    :return: transition probabilities over the tags as dictionary
    """
    dic = dict()
    tag_set = set()
    for tags_sent in tags:
        for j, tag in enumerate(tags_sent):
            tup = ("START", tag)
            if 0 < j < len(tags_sent) - 1:
                tup = (tags_sent[j - 1], tag)
            if j == len(tags_sent) - 1:
                tup = (tag, "STOP")
            if tup in dic:
                dic[tup] = dic[tup] + 1
            else:
                dic[tup] = 1
            if tup[0] in dic:
                dic[tup[0]] = dic[tup[0]] + 1
            else:
                dic[tup[0]] = 1
                tag_set.add(tup[0])
    return dic, tag_set


def transition(dic, tag, prev_tag):
    """
    calculates transition probability from prev tag to tag
    :param dic: transition probabilities
    :param tag: current tag
    :param prev_tag: previous tag
    :return: transition probability from prev tag to tag
    """
    if (prev_tag, tag) in dic:
        tup_count = dic[(prev_tag, tag)]
        prev_count = dic[prev_tag]
        return tup_count / prev_count
    return 0


def viterbi(sent, transition_d, emission_d, word_set, S, emission_func):
    """
        implementation of the viterbi algorithm
        :param sent: sentence to tag
        :param transition_d: transition probabilities
        :param emission_d: emission probabilities
        :param word_set: set of all the words in the training set
        :param S: tags set
        :param emission_func: emission function to be used
        :param low_freq_d: low frequency words
        :return: tags for the given sentence
        """
    pi = np.zeros((len(S), len(sent)), dtype=np.float64)
    bp = np.zeros((len(S), len(sent) + 1), dtype=np.float64)
    for v in range(len(S)):
        t = transition(transition_d, S[v], "START")
        e = emission_func(emission_d, sent[0], S[v], word_set, True)
        pi[v][0] = t * e
    for i in range(1, len(sent)):
        for v in range(len(S)):
            p = 0
            e = emission_func(emission_d, sent[i], S[v], word_set, False)
            for u in range(len(S)):
                p_new = pi[u][i - 1] * transition(transition_d, S[v], S[u]) * e
                if p_new > p:
                    p = p_new
                    pi[v][i] = p_new
                    bp[v][i] = u
    best_cell = 0
    best_cell_index = 0
    for v in range(len(S)):
        if pi[v][-1] * transition(transition_d, "STOP", S[v]) > best_cell:
            best_cell = pi[v][-1]
            best_cell_index = v
    result = [S[best_cell_index]]
    next_tag = best_cell_index
    for i in reversed(range(len(sent) - 1)):
        next_tag = int(bp[next_tag][i + 1])
        result.append(S[next_tag])
    result.reverse()
    return result


def viterbi_tagging(X, y, transition_d, emission_d, word_set, S, emission_func):
    """
    :param X: test set words
    :param y: test set true tags
    :param transition_d: transition probabilities
    :param emission_d: emission probabilities
    :param word_set: set of all the words in the training set
    :param S: tags set
    :param emission_func: emission function to be used
    :return: error rates
    """
    unknown_words, unknown_words_correct = 0, 0
    known_words, known_words_correct = 0, 0
    y_pred = []
    for i in range(len(X)):
        y_sent_pred = viterbi(X[i], transition_d, emission_d, word_set, list(S), emission_func)
        y_pred.append(y_sent_pred)
        y_true = y[i]
        for j in range(len(X[i])):
            if X[i][j] in words_set:
                known_words += 1
                known_words_correct += (y_sent_pred[j] == y_true[j])
            else:
                unknown_words += 1
                unknown_words_correct += (y_sent_pred[j] == y_true[j])
    unknown_error = 1 - unknown_words_correct / unknown_words
    known_error = 1 - known_words_correct / known_words
    total_error = 1 - (known_words_correct + unknown_words_correct) / (known_words + unknown_words)
    return total_error, known_error, unknown_error, y_pred


def clear_tags(y):
    """
    clear + and - from given tags
    :param y: train set tags
    :return: cleared tags
    """
    for i in range(len(y)):
        for j in range(len(y[i])):
            tag = y[i][j]
            tag = tag.split("+")[0]
            tag = tag.split("-")[0]
            y[i][j] = tag
    return y


def print_error(total, known, unknown, title):
    """
    prints error rates
    :param total: total_error
    :param known: known_error
    :param unknown: unknown_error
    :param title: tagging method
    """
    print("error for " + title)
    print(f"total error: {total}, known words error: {known}, unknown words error: {unknown}")


def add_one_emission(dic, word, tag, word_set, is_first):
    """
    add one smoothing emission function
    :param is_first: is the word the first in its the sentence
    :param dic: emission probabilities
    :param word: word to emit
    :param tag: tag to emit
    :param word_set: all words in the train set
    :return: emission probability
    """
    dic_tag = dic.get(tag)
    total_words = dic_tag.get("sum") + len(word_set)
    if word in dic_tag:
        word_count = dic_tag.get(word) + 1
        return word_count / total_words
    else:
        return 1 / total_words


def count_words(X):
    """
    counts the number of words in the train set
    :param X: train set
    :return: number of words in the train set
    """
    words_dic = dict()
    for sent in X:
        for word in sent:
            if word in words_dic:
                words_dic[word] = words_dic[word] + 1
            else:
                words_dic[word] = 1
    return words_dic


def get_low_freq_words(words_dic):
    low_freq_set = set()
    for key in words_dic:
        if words_dic[key] < 2:
            low_freq_set.add(key)
    return low_freq_set


def pseudo_emission_dic(X, Y, words_dic, low_freq_set):
    """
    creates emission probabilities that takes into account unseen words low frequency words
    using pseudo words
    :param X: train set
    :param Y: train labels
    :param words_dic: words in train set
    :param low_freq_set: low frequency words
    :return: pseudo words emission probabilities
    """
    dic = dict()
    pseudo_word_set = set()
    for i in range(len(X)):
        for j in range(len(X[i])):
            tag = Y[i][j]
            word = X[i][j]
            if word in low_freq_set or word not in words_dic:
                word = get_pseudo_word(word, j)
            pseudo_word_set.add(word)
            if tag not in dic:
                tag_dic = dict()
                tag_dic[word] = 1
                tag_dic["sum"] = 1
                dic[tag] = tag_dic
            else:
                tag_dic = dic.get(tag)
                if word in tag_dic:
                    tag_dic[word] = tag_dic[word] + 1
                else:
                    tag_dic[word] = 1
                tag_dic["sum"] = tag_dic["sum"] + 1
    return dic, pseudo_word_set


def get_pseudo_word(word, is_first):
    """
    gets pseudo word type for a given word
    :param word: word to convert
    :param is_first: is it the first words in the sentence
    :return: pseudo word for the given word
    """
    if "$" in word:
        return "dollarAmount"
    if word.isnumeric():
        if len(word) == 4:
            return "fourDigit"
        return "numeric"
    if any(str.isdigit(c) for c in word):
        if ":" in word:
            return "digitAndColon"
        if "-" in word:
            return "digitAndDash"
        if "," in word:
            return "numeric"
        if "." in word:
            return "digitAndDot"
        return "mixedDigit"
    if word[-2:] == "'s":
        return "ApostropheS"
    if "." in word:
        return "charsAndDots"
    if "-" in word:
        return "charsAndDash"
    if word.islower():
        return word
    if word.isupper():
        return "upperCase"
    if is_first:
        return "firstWord"
    if word[0].isupper():
        return "initCap"
    return "else"


def add_one_pseudo_emission(dic, word, tag, word_set, is_first):
    """
    add one smoothing emission function with pseudo words
    :param is_first: is the word the first in its the sentence
    :param dic: emission probabilities
    :param word: word to emit
    :param tag: tag to emit
    :param word_set: all words in the train set
    :return: emission probability
    """
    dic_tag = dic.get(tag)
    total_words = dic_tag.get("sum") + len(word_set)
    if word not in word_set:
        word = get_pseudo_word(word, is_first)
    if word in dic_tag:
        word_count = dic_tag.get(word) + 1
        return word_count / total_words
    return 1 / total_words


def emission_pseudo(dic, word, tag, word_set, is_first):
    """
    :param dic: emission probabilities with pseudo words
    :param word: word to emit
    :param tag: tag to emit
    :param word_set: all words in the train set
    :param is_first: is the word the first in its the sentence
    :return: emission probability
    """
    if word not in word_set:
        word = get_pseudo_word(word, is_first)
    dic_tag = dic.get(tag)
    if word in dic_tag:
        word_count = dic_tag.get(word)
        total_words = dic_tag.get("sum")
        return word_count / total_words
    return 1e-24


# %%

def confusion_matrix(y_true, y_pred):
    tags_set = set()
    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            tags_set.add(y_test[i][j])
            tags_set.add(y_pred[i][j])
    tags = list(tags_set)
    idx_dic = dict()
    for i in range(len(tags)):
        idx_dic[tags[i]] = i
    mat = np.zeros([len(tags), len(tags)])
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            true_tag = y_true[i][j]
            pred_tag = y_pred[i][j]
            mat[idx_dic[true_tag]][idx_dic[pred_tag]] += 1

    df = pd.DataFrame(mat, index=tags, columns=tags)
    return df


if __name__ == '__main__':
    # a
    X_train, X_test, y_train, y_test = load_data()
    y_train = clear_tags(y_train)
    y_test = clear_tags(y_test)
    # b.1
    tags_per_word = most_likely_tag(X_train, y_train)
    # b.2
    total_error, known_error, unknown_error = tag_words(X_test, y_test, tags_per_word)
    print_error(total_error, known_error, unknown_error, "baseline tagging")
    # c.1
    emission_d, words_set = emission_dic(X_train, y_train)
    transition_d, tags_set = transition_dic(y_train)
    # c.2 + c.3
    tags_set.remove("START")
    total_error, known_error, unknown_error, y_pred = viterbi_tagging(X_test, y_test, transition_d, emission_d,
                                                                      words_set,
                                                                      tags_set, emission)
    print_error(total_error, known_error, unknown_error, "viterbi tagging")

    # d.1 + d.2
    total_error, known_error, unknown_error, y_pred = viterbi_tagging(X_test, y_test, transition_d, emission_d,
                                                                      words_set,
                                                                      tags_set, add_one_emission)
    print_error(total_error, known_error, unknown_error, "viterbi add-one tagging")
    # e.1
    words_d = count_words(X_train)
    low_freq = get_low_freq_words(words_d)
    pseudo_dic, pseudo_set = pseudo_emission_dic(X_train, y_train, words_d, low_freq)
    # e.2
    total_error, known_error, unknown_error, y_pred = viterbi_tagging(X_test, y_test, transition_d, pseudo_dic,
                                                                      pseudo_set,
                                                                      tags_set, emission_pseudo)
    print_error(total_error, known_error, unknown_error, "viterbi pseudo words tagging")

    # e.3
    total_error, known_error, unknown_error, y_pred = viterbi_tagging(X_test, y_test, transition_d, pseudo_dic,
                                                                      pseudo_set,
                                                                      tags_set, add_one_pseudo_emission)
    print_error(total_error, known_error, unknown_error, "viterbi pseudo add-one tagging")

    mat = confusion_matrix(y_test, y_pred)
    mat.to_csv("confusion matrix")
