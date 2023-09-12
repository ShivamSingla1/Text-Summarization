

from flask import Flask, render_template, request
import numpy as np
import nltk

nltk_data_path = 'C:/nltk_data'
nltk.data.path.append(nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)

import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['input_text']
    lines = int(request.form['lines'])

    sentences = sent_tokenize(text)

    sentences_clean = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in sentences]
    stop_words = stopwords.words('english')
    sentence_tokens = [[words for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]

    w2v = Word2Vec(sentence_tokens, vector_size=10, min_count=1, epochs=1000)
    sentence_embeddings = [[w2v.wv[word][0] for word in words] for words in sentence_tokens]
    max_len = max([len(tokens) for tokens in sentence_tokens])
    sentence_embeddings = [np.pad(embedding, (0, max_len - len(embedding)), 'constant') for embedding in sentence_embeddings]

    similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
    for i, row_embedding in enumerate(sentence_embeddings):
        for j, column_embedding in enumerate(sentence_embeddings):
            similarity_matrix[i][j] = 1 - spatial.distance.cosine(row_embedding, column_embedding)

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    top_sentence = {sentence: scores[index] for index, sentence in enumerate(sentences)}
    top = dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:lines])

    summarized_text = "\n".join(top.keys())

    return render_template('index.html', input_text=text, summarized_text=summarized_text, lines=lines)

if __name__ == '__main__':
    app.run(debug=True)