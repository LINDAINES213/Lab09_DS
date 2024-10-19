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
import plotly.express as px

# Función para generar histograma interactivo de palabras más frecuentes
def plotTopWordsHistogramInteractive(option):
    if option == 'Desastres':
        title = 'Desastres'
        topWords = disasterWordFreqDf.head(10)
    else:
        title = 'No Desastres'
        topWords = nonDisasterWordFreqDf.head(10)

    # Crear el histograma interactivo con Plotly
    fig = px.bar(topWords, 
                 x='Frequency', 
                 y='Word', 
                 orientation='h',  # Barra horizontal
                 title=f'Top 10 Palabras Más Frecuentes - {title}',
                 labels={'Frequency': 'Frecuencia', 'Word': 'Palabra'},
                 color='Frequency',  # Color basado en frecuencia
                 color_continuous_scale='Blues')  # Escala de color

    fig.update_layout(
        xaxis_title="Frecuencia",
        yaxis_title="Palabras",
        template="plotly_white",
        hovermode="closest"  # Habilitar hover
    )

    # Mostrar la gráfica en Streamlit
    st.plotly_chart(fig)


def bigramAnalysisInteractive(tokenizedTextColumn, targetColumn, label):

    if label == 'Desastres':
        selectedTweets = tokenizedTextColumn[targetColumn == 1]
    else:
        selectedTweets = tokenizedTextColumn[targetColumn == 0]

    def generate_ngrams(selectedTweets, n):
        ngramsList = [ngram for tokens in selectedTweets for ngram in ngrams(tokens, n)]
        ngramCounts = Counter(ngramsList)
        ngramFreqDf = pd.DataFrame(ngramCounts.items(), columns=[f'{n}-gram', 'Frequency']).sort_values(by='Frequency', ascending=False)
        return ngramFreqDf.head(10)

    # Generar bi-gramas
    bigrams = generate_ngrams(selectedTweets, 2)

    # Crear el gráfico interactivo
    fig = px.bar(bigrams, 
                 x='Frequency', 
                 y=bigrams['2-gram'].apply(lambda x: ' '.join(x)),  # Unir los bigramas
                 orientation='h',
                 title=f'Top 10 Bi-gramas - {label}',
                 labels={'Frequency': 'Frecuencia', '2-gram': 'Bi-grama'},
                 color='Frequency',
                 color_continuous_scale='Oranges')

    fig.update_layout(
        xaxis_title="Frecuencia",
        yaxis_title="Bi-gramas",
        template="plotly_white",
        hovermode="closest"
    )

    # Mostrar la gráfica en Streamlit
    st.plotly_chart(fig)
    
# Función para análisis de n-gramas
def trigramAnalysisInteractive(tokenizedTextColumn, targetColumn, label):

    if label == 'Desastres':
        selectedTweets = tokenizedTextColumn[targetColumn == 1]
    else:
        selectedTweets = tokenizedTextColumn[targetColumn == 0]

    def generate_ngrams(selectedTweets, n):
        ngramsList = [ngram for tokens in selectedTweets for ngram in ngrams(tokens, n)]
        ngramCounts = Counter(ngramsList)
        ngramFreqDf = pd.DataFrame(ngramCounts.items(), columns=[f'{n}-gram', 'Frequency']).sort_values(by='Frequency', ascending=False)
        return ngramFreqDf.head(10)

    # Generar tri-gramas
    trigrams = generate_ngrams(selectedTweets, 3)

    # Crear el gráfico interactivo
    fig = px.bar(trigrams, 
                 x='Frequency', 
                 y=trigrams['3-gram'].apply(lambda x: ' '.join(x)),  # Unir los trigramas
                 orientation='h',
                 title=f'Top 10 Tri-gramas - {label}',
                 labels={'Frequency': 'Frecuencia', '3-gram': 'Tri-grama'},
                 color='Frequency',
                 color_continuous_scale='Reds')

    fig.update_layout(
        xaxis_title="Frecuencia",
        yaxis_title="Tri-gramas",
        template="plotly_white",
        hovermode="closest"
    )

    # Mostrar la gráfica en Streamlit
    st.plotly_chart(fig)

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
    plotTopWordsHistogramInteractive(option)

    st.subheader('Bi-Gramas')
    bigramAnalysisInteractive(tokenizedText, dataTrain['target'], label=option)

# Trigrams en la cuarta columna

col3, col4, col5 = st.columns(3)
with col5:
    st.subheader('Tri-Gramas')
    trigramAnalysisInteractive(tokenizedText, dataTrain['target'], label=option)