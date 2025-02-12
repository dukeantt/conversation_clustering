from gensim.models import FastText
from fse.models import Average
from fse import IndexedList
import pandas as pd
import re
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

sentences = []
sentences2 = []
short_input_texts = []
short_bot_texts = []
short_entities = []
short_actions = []

df = pd.read_csv('data/ExpectedConversation_27_04.csv')
for index, value in df.iterrows():
    sentence = str(value['input_text']) + ' ' + str(value['intent']) + ' ' + str(value['entities']) + ' ' + \
               str(value['action_1']) + ' ' + str(value['bot_text'])
    sentence = re.sub('\[|\]|\"|\'', '', sentence)
    sentences.append(sentence.split())
    sentences2.append(sentence)
    short_input_text = str(value['input_text']) if "scontent.xx.fbcdn.net" not in str(
        value['input_text']) else "url"
    short_bot_text = str(value['bot_text']) if "scontent.xx.fbcdn.net" not in str(value['bot_text']) else "url"
    if len(short_input_text) > 50:
        n = int(len(short_input_text) / 50)
        short_input_text = " ".join([short_input_text[50 * x:50 * (x + 1)] + "-" + "<br>" for x in range(n)])
    if len(short_bot_text) > 50:
        n = int(len(short_bot_text) / 50)
        short_bot_text = " ".join([short_bot_text[50 * x:50 * (x + 1)] + "-" + "<br>" for x in range(n)])
    short_entity = str(value['entities']) if "scontent.xx.fbcdn.net" not in str(value['entities']) else "url"
    short_actions = str(value['action_1']) if "scontent.xx.fbcdn.net" not in str(value['action_1']) else "url"

    short_input_texts.append(short_input_text)
    short_bot_texts.append(short_bot_text)
    short_entities.append(short_entity)

ft = FastText(sentences, min_count=1, size=10)

model = Average(ft)
model.train(IndexedList(sentences))

vectors_list = model.sv.vectors.tolist()  # 10 dimensions vectors
# tsne = TSNE(n_components=3)
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
# scores_df = pd.DataFrame(scores, columns=['k', 'silhouette_score', 'inertia'])
# scores_df.to_csv("data/scores.csv", index=False)

kmeans12 = KMeans(n_clusters=12, random_state=0).fit(tsne_vectors)
kmeans4 = KMeans(n_clusters=4, random_state=0).fit(tsne_vectors)
gm = GaussianMixture(n_components=6, n_init=10, covariance_type="diag").fit(tsne_vectors)
hc = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward').fit_predict(tsne_vectors)


# vectors_df = pd.DataFrame(data=tsne_vectors, columns=["x", "y","z"])
vectors_df = pd.DataFrame(data=tsne_vectors, columns=["x", "y"])
df = pd.merge(df, vectors_df, right_index=True, left_index=True)
df['combine_text'] = sentences2
df['kmeans_cluster_12'] = kmeans12.labels_
df['kmeans_cluster_4'] = kmeans4.labels_
df['gm_cluster'] = gm.predict(tsne_vectors)
df['hc_cluster'] = hc
df['short_input_texts'] = short_input_texts
df['short_bot_texts'] = short_bot_texts
df['short_entities'] = short_entities

df.to_csv("data/tsne_vectors_2d_with_multiple_method.csv", index=False)
x = 0
