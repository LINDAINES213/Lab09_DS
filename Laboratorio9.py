import os
import opendatasets as od
import streamlit as st
import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display, clear_output
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
from nltk import ngrams
from collections import Counter
import seaborn as sns
from matplotlib_venn import venn2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.colors import ListedColormap
 
def printText(df, stop = 10):
    for i, t in enumerate(df):
        print(i, t)
        if i >= stop:
            break

if not os.path.isdir('nlp-getting-started'):
    od.download("https://www.kaggle.com/c/nlp-getting-started/data")

 
dataTrain = pd.read_csv('nlp-getting-started/train.csv')
dataTest = pd.read_csv('nlp-getting-started/test.csv')
dataSampleSubmission = pd.read_csv('nlp-getting-started/sample_submission.csv')

 
text = dataTrain['text']

 
textWithoutUrl = text.str.replace(r'http\S+|www\S+', '', regex=True)
textLowerCase = textWithoutUrl.str.lower()
textWithouthSpecialCharacthers = textLowerCase.str.replace('@', '')
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace('#', '')
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace("'", "") 
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace(r'\?+', '?', regex=True)
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace(r'\!+', '!', regex=True)
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace('&amp;', 'and')
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace('\n', ' ')
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace(r'(\.{3,})', 'THREEPOINTSIDENFIFIER', regex=True)
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace(r'(.)\1{'+str(2)+',}', r'\1' * 2, regex=True)
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace('THREEPOINTSIDENFIFIER', '...')
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace('.',' ')
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace(',',' ')
textWithouthSpecialCharacthers = textWithouthSpecialCharacthers.str.replace(r'\s{2,}', ' ', regex=True)

nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

def removeStopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopWords]  
    return ' '.join(filtered_words)

textWithoutStopwords = textWithouthSpecialCharacthers.apply(removeStopwords)
textWithoutStopwords = textWithoutStopwords.str.replace(r'[^\w\s]','',regex=True)

nltk.download('punkt_tab')
tokenizedText = textWithoutStopwords.apply(word_tokenize)
allWords = [word for words in tokenizedText for word in words]
wordCounts = Counter(allWords)

def mostFrequentWords(tokenizedTextColumn, targetColumn):

    disasterTweets = tokenizedTextColumn[targetColumn == 1]
    nonDisasterTweets = tokenizedTextColumn[targetColumn == 0]
    
    def countWords(texts):
        wordsList = [word for tokens in texts for word in tokens]
        wordCounts = Counter(wordsList)
        return wordCounts
    
    disasterWordCounts = countWords(disasterTweets)
    nonDisasterWordCounts = countWords(nonDisasterTweets)
    disasterWordFreqDf = pd.DataFrame(disasterWordCounts.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
    nonDisasterWordFreqDf = pd.DataFrame(nonDisasterWordCounts.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
    
    return disasterWordFreqDf, nonDisasterWordFreqDf

disasterWordFreqDf, nonDisasterWordFreqDf = mostFrequentWords(tokenizedText, dataTrain['target'])

# Función para generar nube de palabras
def plotWordCloud(option):
    if option == 'Desastres':
        title = 'Desastres'
        text = ' '.join(disasterWordFreqDf['Word'])
    else:
        title = 'No Desastres'
        text = ' '.join(nonDisasterWordFreqDf['Word'])
    
    # Usar una paleta apta para daltónicos con seaborn
    palette = sns.color_palette("colorblind", as_cmap=False)
    cmap = ListedColormap(palette.as_hex())
    
    # Crear la nube de palabras con la paleta de colores y un fondo blanco
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=cmap).generate(text)
    
    # Mostrar la nube de palabras con Streamlit
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)

# Función para generar histograma de palabras más frecuentes
def plotTopWordsHistogram(option):
    if option == 'Desastres':
        title = 'Desastres'
        topWords = disasterWordFreqDf.head(10)
    else:
        title = 'No Desastres'
        topWords = nonDisasterWordFreqDf.head(10)
    
    # Crear el histograma
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Frequency', y='Word', data=topWords, palette='colorblind', ax=ax)
    ax.set_xlabel('Frecuencia')
    ax.set_ylabel('Palabras')
    ax.set_title(f'Top 10 Palabras Más Frecuentes - {title}')
    ax.legend().remove()
    st.pyplot(fig)


def bigramAnalysis(tokenizedTextColumn, targetColumn, label):

    if label == 'Desastres':
        selectedTweets = tokenizedTextColumn[targetColumn == 1]
    else:
        selectedTweets = tokenizedTextColumn[targetColumn == 0]
    
    def generate_ngrams(selectedTweets, n):
        ngramsList = [ngram for tokens in selectedTweets for ngram in ngrams(tokens, n)]
        ngramCounts = Counter(ngramsList)
        ngramFreqDf = pd.DataFrame(ngramCounts.items(), columns=[f'{n}-gram', 'Frequency']).sort_values(by='Frequency', ascending=False)
        topNgrams = ngramFreqDf.head(10)
        return topNgrams
    
    # Generar bi-gramas y tri-gramas
    bigrams = generate_ngrams(selectedTweets, 2)
    
    # Usar paleta colorblind para los colores de las barras
    palette = sns.color_palette("colorblind", 1)
    
    # Graficar bi-gramas
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh([' '.join(ngram) for ngram in bigrams['2-gram']], bigrams['Frequency'], color=palette[0])
    ax.set_xlabel('Frecuencia')
    ax.set_ylabel('Bi-gramas')
    ax.set_title(f'Top 10 Bi-gramas - {label}')
    ax.invert_yaxis()
    st.pyplot(fig)
    
# Función para análisis de n-gramas
def trigramAnalysis(tokenizedTextColumn, targetColumn, label):

    if label == 'Desastres':
        selectedTweets = tokenizedTextColumn[targetColumn == 1]
    else:
        selectedTweets = tokenizedTextColumn[targetColumn == 0]
    
    def generate_ngrams(selectedTweets, n):
        ngramsList = [ngram for tokens in selectedTweets for ngram in ngrams(tokens, n)]
        ngramCounts = Counter(ngramsList)
        ngramFreqDf = pd.DataFrame(ngramCounts.items(), columns=[f'{n}-gram', 'Frequency']).sort_values(by='Frequency', ascending=False)
        topNgrams = ngramFreqDf.head(10)
        return topNgrams
    
    # Generar bi-gramas y tri-gramas
    trigrams = generate_ngrams(selectedTweets, 3)
    
    # Usar paleta colorblind para los colores de las barras
    palette = sns.color_palette("colorblind", 1)
    
    # Graficar tri-gramas
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh([' '.join(ngram) for ngram in trigrams['3-gram']], trigrams['Frequency'], color=palette[0])
    ax.set_xlabel('Frecuencia')
    ax.set_ylabel('Tri-gramas')
    ax.set_title(f'Top 10 Tri-gramas - {label}')
    ax.invert_yaxis()
    st.pyplot(fig)

# Configurar la página para que use el ancho completo
st.set_page_config(layout="wide")  # Añade esta línea para hacer que el diseño sea "wide"

# Crear un layout con 2 columnas (que ahora ocuparán todo el ancho de la página)
col1, col2 = st.columns(2)

# Primera columna: Nube de palabras
with col1:
    # Título y selección de categoría
    st.title('Minería de textos - Clasificación de tweets')

    # Seleccionar entre "Desastres" y "No Desastres"
    option = st.selectbox('Seleccionar categoría:', ['Desastres', 'No Desastres'])
    st.subheader('Nube de palabras')
    plotWordCloud(option)

# Segunda columna: Top 10 palabras más frecuentes
with col2:
    st.subheader('Top 10 Palabras más frecuentes')
    plotTopWordsHistogram(option)

    st.subheader('Bi-Gramas')
    bigramAnalysis(tokenizedText, dataTrain['target'], label=option)

# Trigrams en la cuarta columna

col3, col4, col5 = st.columns(3)
with col5:
    st.subheader('Tri-Gramas')
    trigramAnalysis(tokenizedText, dataTrain['target'], label=option)