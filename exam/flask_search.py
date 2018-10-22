from flask import Flask
from flask import render_template, request
from pymystem3 import Mystem
import string
from gensim import matutils
import numpy as np
from math import log
from collections import defaultdict
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle
from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec

mystem = Mystem()
app = Flask(__name__)

russian_stopwords = set(stopwords.words('russian'))

with open('info_data.pickle', 'rb') as f:
    info_data = pickle.load(f)
with open('word_count_del.pickle', 'rb') as f:
    word_count_del = pickle.load(f)
with open('word_count_not_del.pickle', 'rb') as f:
    word_count_not_del = pickle.load(f)
with open('vec_data_del.pickle', 'rb') as f:
    vec_data_del = pickle.load(f)
with open('vec_data_not_del.pickle', 'rb') as f:
    vec_data_not_del = pickle.load(f)

avgdl = np.mean([i['len'] for i in info_data.values()])
model_w2v = Word2Vec.load('araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model')
model_d2v = Doc2Vec.load('my_d2v_model')


def preprocessing(input_text, stopwords={}, del_stopwords=True, del_digit=True):

    words = [x.lower().strip(string.punctuation+'»«–…') for x in word_tokenize(input_text)]
    lemmas = [mystem.lemmatize(x)[0] for x in words if x]
    lemmas_arr = []
    for lemma in lemmas:
        if del_stopwords:
            if lemma in stopwords:
                continue
        if del_digit:
            if lemma.isdigit():
                continue
        lemmas_arr.append(lemma)
    return lemmas_arr


def get_w2v_vectors(lemmas, model):

    lemmas_vectors = []
    for lemma in lemmas:
        try:
            lemmas_vectors.append(model.wv[lemma])
        except:
            None
    if lemmas_vectors:
        doc_vec = sum(lemmas_vectors)
        normalized_vec = matutils.unitvec(doc_vec)
        return list(normalized_vec)
    else:
        return [0] * 300


def similarity(vec1, vec2):
    return np.dot(vec1, vec2)


def culc_sim_score(all_data, vec, model_type):

    answer = defaultdict(float)  # id : score

    for part in all_data:

        if model_type == 'word2v':
            sim = similarity(part['w2v_vec'], vec)
        elif model_type == 'doc2v':
            sim = similarity(part['d2v_vec'], vec)
        else: raise ValueError

        if answer[part['id']] == 0.0: answer[part['id']] = float('-inf')

        if sim > answer[part['id']]: answer[part['id']] = sim

    return answer


def search_w2v(string, model, info_data, vec_data, stopwords={}, amount=10, del_stop=True):

    if not isinstance(string, str):
        raise ValueError('enter correct data')

    words = preprocessing(string, stopwords=stopwords, del_stopwords=del_stop, del_digit=True)
    vec = get_w2v_vectors(words, model)
    answer = culc_sim_score(vec_data, vec, 'word2v')

    for index, ans in enumerate(sorted(answer.items(), reverse=True, key=lambda x: x[1])):
        if index >= amount: break
        yield (ans[0], info_data[ans[0]], ans[1])


def get_d2v_vectors(text, model):
    """Получает вектор документа"""
    return model.infer_vector(text)


def search_d2v(string, model, info_data, vec_data, stopwords={}, del_stop=False, amount=10):

    if not isinstance(string, str):
        raise ValueError('enter correct data')

    words = preprocessing(string, stopwords=stopwords, del_stopwords=del_stop, del_digit=True)
    vec = get_d2v_vectors(words, model)
    answer = culc_sim_score(vec_data, vec, 'doc2v')

    for index, ans in enumerate(sorted(answer.items(), reverse=True, key=lambda x: x[1])):
        if index >= amount: break
        yield (ans[0], info_data[ans[0]], ans[1])


def score_BM25(qf, dl, avgdl, k1, b, N, n):
    """
    Compute similarity score between search query and documents from collection
    :return: score

    qf - кол - во вхождений слова в документе
    dl - длина документа
    """

    tf = qf / dl
    idf = log(N - n + 0.5 / n + 0.5)
    a = (k1 + 1) * tf
    b = tf + k1*(1 - b + b*(dl / avgdl))

    return (a / b) * idf


def compute_sim(words, avgdl, doc, info_data, word_count, N):
    """
    Compute similarity score between search query and documents from collection
    :return: score
    """

    k1 = 2.0
    b = 0.75
    ans = 0

    for word in words:
        if word_count[word] != {}:

            try: qf = word_count[word][doc]
            except KeyError: qf = 0

            dl = info_data[doc]['len']
            n = len(word_count[word])
            ans += score_BM25(qf, dl, avgdl, k1, b, N, n)

    return ans


def get_search_result(text, avgdl, info_data, word_count, stopwords={}, del_stop=True, amount=10):
    """
    Compute sim score between search query and all documents in collection
    Collect as pair (doc_id, score)
    :param query: input text
    :return: list of lists with (doc_id, score)
    """

    if not isinstance(text, str):
        raise ValueError

    words = preprocessing(text, stopwords=stopwords, del_stopwords=del_stop, del_digit=True)
    answer = {}
    N = len(info_data)

    for doc in info_data:
        answer[doc] = compute_sim(words, avgdl, doc, info_data, word_count, N)

    for index, ans in enumerate(sorted(answer.items(), reverse=True, key=lambda x: x[1])):
        if index >= amount: break
        yield (ans[0], info_data[ans[0]], ans[1])


def merging_all_3(w2v, d2v, okapi, all_):

    ans = {}

    for item in all_:

        try: it_w = w2v[item][1]
        except KeyError: it_w = 0

        try: it_d = d2v[item][1]
        except KeyError: it_d = 0

        try: it_o = okapi[item][1]
        except KeyError: it_o = 0

        score = ((it_o * 0.8) + ((it_d * 0.2 + it_w + 0.8) / 2)*0.2) / 2
        ans[item] = score
    return ans


def search_w2_d2_ok(string, avgdl, model_w2v, model_d2v, info_data, vec_data, word_count, stopwords={}, del_stop=False, amount=10):

    w2v = {i[0]: (i[1], i[2]) for i in search_w2v(string, model_w2v, info_data, vec_data, stopwords=stopwords, amount=amount, del_stop=del_stop)}
    d2v = {i[0]: (i[1], i[2]) for i in search_d2v(string, model_d2v, info_data, vec_data, stopwords=stopwords, amount=amount, del_stop=del_stop)}
    okapi = {i[0]: (i[1], i[2]) for i in get_search_result(string, avgdl, info_data, word_count, stopwords=stopwords, del_stop=del_stop, amount=amount)}

    all_ = set(w2v.keys()) | set(d2v.keys()) | set(okapi.keys())
    answer = merging_all_3(w2v, d2v, okapi, all_)

    for index, ans in enumerate(sorted(answer.items(), reverse=True, key=lambda x: x[1])):
        if index >= amount: break
        yield (ans[0], info_data[ans[0]], ans[1])


def search(string, search_method, avgdl, model_w2v, model_d2v, info_data, vec_data_del, vec_data_not_del, word_count_del, word_count_not_del, amount=10, del_stop=True, stopwords={}):

    if search_method == 'inverted_index':
        if del_stop != 'True':
            search_result = get_search_result(string, avgdl, info_data, word_count_not_del, stopwords=stopwords, del_stop=False, amount=amount)
        else:
            search_result = get_search_result(string, avgdl, info_data, word_count_del, stopwords=stopwords, del_stop=True, amount=amount)

    elif search_method == 'word2vec':
        if del_stop != 'True':
            search_result = search_w2v(string, model_w2v, info_data, vec_data_not_del, stopwords=stopwords, amount=amount, del_stop=False)
        else:
            search_result = search_w2v(string, model_w2v, info_data, vec_data_del, stopwords=stopwords, amount=amount, del_stop=True)

    elif search_method == 'doc2vec':
        if del_stop != 'True':
            search_result = search_d2v(string, model_d2v, info_data, vec_data_not_del, stopwords=stopwords, amount=amount, del_stop=False)
        else:
            search_result = search_d2v(string, model_d2v, info_data, vec_data_del, stopwords=stopwords, amount=amount, del_stop=True)

    elif search_method == 'all':
        if del_stop != 'True':
            search_result = search_w2_d2_ok(string, avgdl, model_w2v, model_d2v, info_data, vec_data_not_del, word_count_not_del, del_stop=False, stopwords=stopwords, amount=amount)
        else:
            search_result = search_w2_d2_ok(string, avgdl, model_w2v, model_d2v, info_data, vec_data_del, word_count_del, del_stop=True, stopwords=stopwords, amount=amount)

    else:
        raise TypeError('unsupported search method')
    return search_result


@app.route('/', methods=['GET'])
def index():

    if request.args:
        query = request.args['words']
        stops = request.args['stops']
        amount = int(request.args['amount'])
        search_method = request.args['model']

        global avgdl, model_w2v, model_d2v, info_data, vec_data_del, vec_data_not_del, word_count_del, word_count_not_del, russian_stopwords

        try:
            result = search(query, search_method, avgdl, model_w2v, model_d2v, info_data, vec_data_del, vec_data_not_del, word_count_del, word_count_not_del, amount=amount, del_stop=stops, stopwords=russian_stopwords)
            return render_template('result.html', name=result)
        except: return render_template('error.html')

    return render_template('Index.html')


@app.route('/result', methods=['GET'])
def result():

    if request.args:
        query = request.args['words']
        stops = True
        amount = 10
        search_method = 'inverted_index'

        global avgdl, model_w2v, model_d2v, info_data, vec_data_del, vec_data_not_del, word_count_del, word_count_not_del, russian_stopwords

        try:
            result = search(query, search_method, avgdl, model_w2v, model_d2v, info_data, vec_data_del, vec_data_not_del, word_count_del, word_count_not_del, amount=amount, del_stop=stops, stopwords=russian_stopwords)
            return render_template('result.html', name=result)
        except: return render_template('error.html')

    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=False)
