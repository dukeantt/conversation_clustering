from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import re
import string
import unicodedata

df = pd.read_csv('data/ExpectedConversation_27_04.csv')

sentences = []
sentences2 = []
short_input_texts = []
short_bot_texts = []
short_entities = []
short_actions = []

for index, value in df.iterrows():
    text = str(value['input_text'])
    if "scontent.xx.fbcdn.net" in str(value['input_text']):
        text = re.sub('\[|\]|\'|\n|\{|\}', '', str(value['cv_outputs']))
        # text = text.split(',')[0].replace('object_type:', '')
        if text.split(',')[0].replace('object_type:', '') == ' ':
            text = text.split(',')[2].replace('product_name:', '')
        else:
            text = text.split(',')[0].replace('object_type:', '')

    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    sentences.append(str(text).split())
    sentences2.append(str(text))

for index, value in df.iterrows():
    short_input_text = str(value['input_text'])
    if "scontent.xx.fbcdn.net" in short_input_text:
        short_input_text = re.sub('\[|\]|\'|\n|\{|\}', '', str(value['cv_outputs']))
        if short_input_text.split(',')[0].replace('object_type:', '') == ' ':
            short_input_text = short_input_text.split(',')[2].replace('product_name:', '')
        else:
            short_input_text = short_input_text.split(',')[0].replace('object_type:', '')

    short_bot_text = str(value['bot_text']) if "scontent.xx.fbcdn.net" not in str(value['bot_text']) else "url"
    if len(short_input_text) > 50:
        n = int(len(short_input_text) / 50)
        short_input_text = " ".join([short_input_text[50 * x:50 * (x + 1)] + "-" + "<br>" for x in range(n)])
    if len(short_bot_text) > 50:
        n = int(len(short_bot_text) / 50)
        short_bot_text = " ".join([short_bot_text[50 * x:50 * (x + 1)] + "-" + "<br>" for x in range(n)])
    short_entity = str(value['entities'])
    if len(short_entity) > 50:
        n = int(len(short_entity) / 50)
        short_entity = " ".join([short_entity[50 * x:50 * (x + 1)] + "-" + "<br>" for x in range(n)])
    short_entity = short_entity if "scontent.xx.fbcdn.net" not in short_entity else "url"
    short_actions = str(value['action_1']) if "scontent.xx.fbcdn.net" not in str(value['action_1']) else "url"
    short_input_texts.append(short_input_text)
    short_bot_texts.append(short_bot_text)
    short_entities.append(short_entity)

tfidf_vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\w+")
tfidf = tfidf_vectorizer.fit_transform(sentences2)
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names())

model = Word2Vec(sentences, size=20, window=3, min_count=1, workers=4)
vectors_list = []
for sent_index, sent_value in enumerate(sentences):
    if (len(sent_value) == 0):
        sentence_length = 1
    else:
        sentence_length = len(sent_value)

    vector = 0
    for word_index, word_value in enumerate(sent_value):
        try:
            word_tfidf = tfidf_df.loc[sent_index][word_value]
        except KeyError:
            print(word_value)
            word_tfidf = 1
        word_vector_improved = model.wv[word_value] * word_tfidf
        vector += word_vector_improved
    vector = vector / sentence_length
    try:
        vector = vector.tolist()
    except:
        vector = [float(0)] * 20
    vectors_list.append(vector)
    x = 0

tsne = TSNE(n_components=2)
tsne_vectors = tsne.fit_transform(vectors_list)

# scores = []
# for k in range(2,20):
#     x = k
#     kmeans = KMeans(n_clusters=x, random_state=0)
#     kmeans = kmeans.fit(tsne_vectors)
#     labels = kmeans.labels_
#     score = silhouette_score(tsne_vectors, labels)
#     inertia = kmeans.inertia_
#     scores.append((k, score,inertia))
#
# scores_df = pd.DataFrame(scores, columns=['k', 'silhouette_score','inertia'])
# scores_df.to_csv("data/custom_embedding_scores.csv", index=False)

gm = GaussianMixture(n_components=12, n_init=10, covariance_type="spherical").fit(tsne_vectors)
hc = AgglomerativeClustering(n_clusters=12, affinity='euclidean', linkage='ward').fit_predict(tsne_vectors)
kmeans = KMeans(n_clusters=12, random_state=0).fit(tsne_vectors)

vectors_df = pd.DataFrame(data=tsne_vectors, columns=["x", "y"])
df = pd.merge(df, vectors_df, right_index=True, left_index=True)
df['combine_text'] = sentences2
df['kmeans_cluster'] = kmeans.labels_
df['gm_cluster'] = gm.predict(tsne_vectors)
df['hc_cluster'] = hc
df['short_input_texts'] = short_input_texts
df['short_bot_texts'] = short_bot_texts
df['short_entities'] = short_entities

df.to_csv("data/custom_embedding_12_clusters.csv", index=False)

x = 0
