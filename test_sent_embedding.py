from gensim.models import FastText
from fse.models import Average
from fse import IndexedList
import pandas as pd
import re
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans


sentences = []
sentences2 = []
input_text = []
bot_text = []
intent = []
entities = []
action = []

df = pd.read_csv('data/ExpectedConversation_27_04.csv')
for index, value in df.iterrows():
    sentence = str(value['input_text']) + ' ' + str(value['intent']) + ' ' + str(value['entities']) + ' ' + str(
        value['action_1']) + ' ' + \
               str(value['bot_text']) + ' ' + str(value['input_text'])
    sentence = re.sub('\[|\]|\"|\'', '', sentence)
    sentences.append(sentence.split())
    sentences2.append(sentence)
    input_text.append(value['input_text'])
    bot_text.append(value['bot_text'])
    intent.append(value['intent'])
    entities.append(value['entities'])
    action.append(value['action_1'])

ft = FastText(sentences, min_count=1, size=10)

model = Average(ft)
model.train(IndexedList(sentences))

model.sv.similarity(0, 2)
vectors_list = model.sv.vectors.tolist()
tsne = TSNE(n_components=2)
tsne_vectors = tsne.fit_transform(vectors_list)
kmeans = KMeans(n_clusters=4, random_state=0).fit(tsne_vectors)

df2 = pd.DataFrame(data=tsne_vectors, columns=["x", "y"])
df2['text'] = sentences2
df2['input_text'] = input_text
df2['bot_text'] = bot_text
df2['intent'] = intent
df2['entities'] = entities
df2['action'] = action
df2['cluster'] = kmeans.labels_
df2.to_csv("data/tsne_vectors5.csv", index=False)
x = 0
