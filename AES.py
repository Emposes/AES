import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from tensorflow import keras
from tensorflow.keras import layers

import pathlib
from collections import Counter
from statistics import mean

import language_check
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns 
import textstat as ts
from textblob import TextBlob
from textblob import Word
import sklearn

nltk.download('punkt'); nltk.download('brown'); nltk.download('averaged_perceptron_tagger')
sns.set(style="ticks", color_codes=True)

#Load dataset
df_train = pd.read_json('C:/Users/Win/Desktop/Doctorat/Poli/SII/Referate/data/referate-train.json', encoding="utf8") 

df_dev = pd.read_json('C:/Users/Win/Desktop/Doctorat/Poli/SII/Referate/data/referate-dev.json', encoding="utf8") 

new= [df_train, df_dev]

df = pd.concat(new, ignore_index=True)

from tensorflow.keras.preprocessing.text import Tokenizer
#Tokenize category
category_tokenizer = Tokenizer()
category_tokenizer.fit_on_texts(df['category'])

df['category_code'] = np.array(category_tokenizer.texts_to_sequences(df['category']))

df['text'].replace('', np.nan, inplace=True)

df = df.dropna(axis='rows')

df = df[df['text'].notna()]

df.category.value_counts().rename_axis('category').reset_index(name='counts').sort_values(
    'category').plot(kind='bar', x="category", y="counts")

df['grade2'] = df.grade.round(0)

df['grade'].describe().transpose()

df.groupby('category')['grade'].describe().transpose()

meta_features = ['essay_length', 'avg_sentence_length', 'avg_word_length']
grammar_features = [ 'syntax_errors']

df.reindex(columns=meta_features + grammar_features, fill_value=np.zeros)
essays = df['text'].values

def add_meta_feature_columns(index, df, blob):
    
    # Essay Length (number of words)
    df.at[index, 'essay_length'] = len(blob.words)

    # Average Sentence Length
    sentence_lengths = [len(sentence.split(' ')) for sentence in blob.sentences]
    df.at[index, 'avg_sentence_length'] = mean(sentence_lengths)

    # Average Word Length
    word_lengths = [len(word) for word in blob.words]
    df.at[index, 'avg_word_length'] = mean(word_lengths)

 
def add_grammar_feature_columns(index, df, blob, essay):
    
    # Number of possible spelling and grammatical Mistakes
    print("Processed %5d essays for correctness..." % (index + 1), end="\r")
    languageTool = language_check.LanguageTool('ro')
    df.at[index, 'syntax_errors'] = len(languageTool.check(essay))
    
print("Adding feature Columns...")

df[df.index.duplicated()]


for i in range(df.shape[0]):
    blob = TextBlob(essays[i])

    add_meta_feature_columns(i, df, blob)
    
    add_grammar_feature_columns(i, df, blob, essays[i])
    
print("\nDone!")


#df.to_pickle('C:/Users/Win/Desktop/Doctorat/Poli/SII/Referate/data/training_set.pkl')

dataset = pd.read_pickle('C:/Users/Win/Desktop/Doctorat/Poli/SII/Referate/data/training_set.pkl')

dataset = dataset.dropna(axis='rows')

dataset = dataset.dropna(axis='columns').drop(columns=['text'])
dataset = dataset.dropna(axis='columns').drop(columns=['filename'])
dataset = dataset.dropna(axis='columns').drop(columns=['category'])
dataset = dataset.dropna(axis='columns').drop(columns=['index'])
dataset = dataset.dropna(axis='columns').drop(columns=['grade2'])

from sklearn import preprocessing
x = dataset.iloc[:,1:6].values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dataset2 = pd.DataFrame(x_scaled)

train_dataset = dataset2.sample(frac=0.8,random_state=42)
test_dataset = dataset2.drop(train_dataset.index)

train_labels = train_dataset.pop('grade')

scaler = preprocessing.MinMaxScaler()
train_grade_new = scaler.fit_transform(dataset[['grade']])
train_labels = train_labels.round(0)
train_labels2 = scaler.transform([train_labels])
test_labels = test_dataset.pop('grade')

import seaborn as sns
sns.displot(train_labels2[0])
# box-cox transform
from numpy.random import seed
from numpy.random import randn
from numpy import exp
from scipy.stats import boxcox
from matplotlib import pyplot
# seed the random number generator
seed(42)
# transform to be exponential
data23 = exp(train_labels2[0])
# power transform
data23 = boxcox(data23, 0)

mini_batch1 = 100
mini_batch2 = 10

def build_model():
    model = keras.Sequential([
    layers.Dense(mini_batch1, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(mini_batch2, activation='relu'),
    layers.Dense(mini_batch1, activation='relu'),
    layers.Dense(mini_batch2, activation='relu'),
    layers.Dense(mini_batch1, activation='relu'),
    layers.Dense(mini_batch2, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation = 'sigmoid')
  ])

    optimizer = tf.keras.optimizers.RMSprop(5e-5)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

model=0
model = build_model()

model.summary()

EPOCHS = 5000
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

early_history = model.fit(train_dataset, data23, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=2, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])


loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} domain1_score".format(mae))


test_predictions = model.predict(test_dataset)
test_predictions = scaler.inverse_transform(test_predictions)

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [grade]')
plt.ylabel('Predictions [grade]')
lims = [0, 10]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

#Test Accuracy
sklearn.metrics.mean_squared_error(test_labels, test_predictions)


df_test = pd.read_json('C:/Users/Win/Desktop/Doctorat/Poli/SII/Referate/data/referate-test.json', encoding="utf8") 
essays = df_test['text'].values
df_test['category_code'] = np.array(category_tokenizer.texts_to_sequences(df_test['category']))
df_test['text'].replace('', np.nan, inplace=True)
df_test = df_test.dropna(axis='rows')

for i in range(485, df_test.shape[0]+1):
    blob = TextBlob(essays[i])

    add_meta_feature_columns(i, df_test, blob)
    
    add_grammar_feature_columns(i, df_test, blob, essays[i])
    
print("\nDone!")

dataset_test = df_test.dropna(axis='rows')

dataset_test = dataset_test.dropna(axis='columns').drop(columns=['text'])
dataset_test = dataset_test.dropna(axis='columns').drop(columns=['filename'])
dataset_test = dataset_test.dropna(axis='columns').drop(columns=['category'])
dataset_test = dataset_test.dropna(axis='columns').drop(columns=['index'])

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(dataset_test)
dataset_test2 = pd.DataFrame(x_scaled)

test_predictions = model.predict(dataset_test2)
test_predictions = scaler.inverse_transform(test_predictions)
df_test['grade'] = test_predictions
df_test = df_test.to_json()
df_test.to_json(r'C:/Users/Win/Desktop/Doctorat/Poli/SII/Referate/data/data.json')

df_test2 = pd.read_json('C:/Users/Win/Desktop/Doctorat/Poli/SII/Referate/data/data.json', encoding="utf8") 

