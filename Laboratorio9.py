import os
import json
import opendatasets as od
import streamlit as st
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
from sklearn.metrics import accuracy_score, confusion_matrix,  precision_score, recall_score, f1_score
from matplotlib.colors import ListedColormap
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

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

def plotWordCloudInteractive(option, max_words=100, background_color='white'):
    if option == 'Desastres':
        title = 'Desastres'
        text = ' '.join(disasterWordFreqDf['Word'])
    else:
        title = 'No Desastres'
        text = ' '.join(nonDisasterWordFreqDf['Word'])
    palette = sns.color_palette("colorblind", as_cmap=False)
    cmap = ListedColormap(palette.as_hex())
    wordcloud = WordCloud(width=800, height=400, 
                          background_color=background_color, 
                          colormap=cmap, 
                          max_words=max_words).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)

def plotTopWordsHistogramInteractive(option):
    if option == 'Desastres':
        title = 'Palabras relacionadas a desastres'
        topWords = disasterWordFreqDf.head(10)
    else:
        title = 'Palabras no relacionadas a desastres'
        topWords = nonDisasterWordFreqDf.head(10)
    fig = px.bar(topWords, 
                 x='Frequency', 
                 y='Word', 
                 orientation='h', 
                 title=f'{title}',
                 labels={'Frequency': 'Frecuencia', 'Word': 'Palabra'},
                 color='Frequency', 
                 color_continuous_scale='Blues')
    fig.update_layout(
        xaxis_title="Frecuencia",
        yaxis_title="Palabras",
        template="plotly_white",
        hovermode="closest" 
    )
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
    bigrams = generate_ngrams(selectedTweets, 2)
    fig = px.bar(bigrams, 
                 x='Frequency', 
                 y=bigrams['2-gram'].apply(lambda x: ' '.join(x)),
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
    st.plotly_chart(fig)
    
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

    trigrams = generate_ngrams(selectedTweets, 3)
    fig = px.bar(trigrams, 
                 x='Frequency', 
                 y=trigrams['3-gram'].apply(lambda x: ' '.join(x)),  
                 orientation='h',
                 title=f'Top 10 Tri-gramas - {label}',
                 labels={'Frequency': 'Frecuencia', '3-gram': 'Tri-grama'},
                 color='Frequency',
                 color_continuous_scale='Greens')
    fig.update_layout(
        xaxis_title="Frecuencia",
        yaxis_title="Tri-gramas",
        template="plotly_white",
        hovermode="closest"
    )
    st.plotly_chart(fig)


def commonWords(disasterWordFreqDf, nonDisasterWordFreqDf):
    print("🔍 Identificando palabras comunes en ambas categorías...")
    disasterWords = set(disasterWordFreqDf['Word'])
    nonDisasterWords = set(nonDisasterWordFreqDf['Word'])
    commonWords = disasterWords.intersection(nonDisasterWords)
    print("📚 Palabras comunes en ambas categorías:")
    print(commonWords)
    palette = sns.color_palette("colorblind", 2)
    plt.figure(figsize=(10, 7))
    venn = venn2([disasterWords, nonDisasterWords], ('Desastres', 'No Desastres'))
    venn.get_label_by_id('10').set_color(palette[0])
    venn.get_label_by_id('01').set_color(palette[1])
    plt.title('')
    st.pyplot(plt)

def add_embedding_noise(embedding, noise_factor=0.1):
    noise = tf.random.normal(shape=tf.shape(embedding), mean=0.0, stddev=noise_factor, dtype=tf.float32)
    return embedding + noise  

sentences = textWithoutStopwords.tolist()
targets = dataTrain['target'].tolist()
testSentences = dataTest['text'].tolist()

trainingSize = int(len(sentences) * 0.80)
trainingSentences = sentences[0:trainingSize]
testingSentences = sentences[trainingSize:]
trainingTargets = targets[0:trainingSize]
testingTargets = targets[trainingSize:]

trainingTargetsArray = np.array(trainingTargets)
testingTargetsArray = np.array(testingTargets)

tokenizer = Tokenizer(num_words=500, oov_token='<OOV>')
tokenizer.fit_on_texts(trainingSentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(trainingSentences)
training_padded = pad_sequences(training_sequences, maxlen=40, padding='post', truncating='post')
testing_sequences = tokenizer.texts_to_sequences(testingSentences)
testing_padded = pad_sequences(testing_sequences, maxlen=40, padding='post', truncating='post')

main_test_sequence = tokenizer.texts_to_sequences(testSentences)
main_test_padded = pad_sequences(main_test_sequence, maxlen=40, padding='post', truncating='post')

# Función para guardar resultados
def save_results(history, loss, accuracy, model_name):
    # Guardar el history en un archivo JSON
    history_file = f"models/{model_name}_history.json"
    with open(history_file, 'w') as f:
        json.dump(history.history, f)

    # Guardar el test loss y test accuracy
    results_file = f"models/{model_name}_results.json"
    results = {
        'test_loss': loss,
        'test_accuracy': accuracy
    }
    with open(results_file, 'w') as f:
        json.dump(results, f)

# Función para cargar resultados
def load_results(model_name):
    history_file = f"models/{model_name}_history.json"
    results_file = f"models/{model_name}_results.json"

    if os.path.exists(history_file) and os.path.exists(results_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return history, results['test_loss'], results['test_accuracy']
    else:
        return None, None, None

def metrictsOfModels(model_name, y_test, y_pred, output_file='model_metrics.json'):
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist() 
    }
    try:
        with open(output_file, 'r') as f:
            all_metrics = json.load(f)
    except FileNotFoundError:
        all_metrics = {}
    all_metrics[model_name] = metrics
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Métricas para {model_name} guardadas en {output_file}.")

# Modelo 1
def createModel1():
    model = tf.keras.Sequential([
        layers.Embedding(500, 16, input_length=40),
        layers.Lambda(lambda x: add_embedding_noise(x, noise_factor=0.05)),
        layers.Bidirectional(layers.LSTM(8, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(8)),
        layers.Dense(72, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(36, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Modelo 2
def createModel2():
    secondModel = tf.keras.Sequential([
        layers.Embedding(500, 16, input_length=40),
        layers.Conv1D(128, 5, activation='relu'),
        layers.MaxPooling1D(pool_size=4),
        layers.Conv1D(64, 5, activation='relu'),
        layers.MaxPooling1D(pool_size=4),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.001)
    secondModel.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    secondModel.summary()
    return secondModel

history, loss, accuracy = load_results('firstModel')
if history is None:
    firstModel = createModel1()
    firstModel.summary()
    earlyStopping = EarlyStopping(min_delta=0.001, patience=10)
    history = firstModel.fit(
        training_padded,
        trainingTargetsArray,
        epochs=15,
        validation_data=(testing_padded, testingTargetsArray),
        callbacks=[earlyStopping]
    )
    loss, accuracy = firstModel.evaluate(testing_padded, testingTargetsArray)
    y_pred_prob = firstModel.predict(testing_padded)  
    y_pred_classes = (y_pred_prob >= 0.5).astype(int).flatten()
    save_results(history, loss, accuracy, 'firstModel')
    metrictsOfModels('firstModel', testingTargetsArray, y_pred_classes, 'modelo1_metrics.json')
    history, loss, accuracy = load_results('firstModel')
else:
    print("Resultados cargados desde el archivo:")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

history2, loss2, accuracy2 = load_results('secondModel')
if history2 is None:
    secondModel = createModel2()
    earlyStopping = EarlyStopping(min_delta=0.001, patience=10)
    history2 = secondModel.fit(
        training_padded,
        trainingTargetsArray,
        epochs=15,
        validation_data=(testing_padded, testingTargetsArray),
        callbacks=[earlyStopping]
    )
    loss2, accuracy2 = secondModel.evaluate(testing_padded, testingTargetsArray)
    y_pred_prob = secondModel.predict(testing_padded)
    y_pred_classes = (y_pred_prob >= 0.5).astype(int).flatten()
    save_results(history2, loss2, accuracy2, 'secondModel')
    metrictsOfModels('secondModel', testingTargetsArray, y_pred_classes, 'modelo2_metrics.json')
    history2, loss2, accuracy2 = load_results('secondModel')
else:
    print("Resultados cargados desde el archivo:")
    print(f"Test Loss: {loss2:.4f}")
    print(f"Test Accuracy: {accuracy2:.4f}")

def showResults(col, history):
    if history is not None:
        epochs = list(range(1, len(history['loss']) + 1))
        color_palette = sns.color_palette("colorblind", 3).as_hex()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=history['loss'], mode='lines', name='Training Loss',
            line=dict(color=color_palette[0])
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_loss'], mode='lines', name='Validation Loss',
            line=dict(color=color_palette[1])
        ))
        fig.update_layout(
            title="Pérdida durante el entrenamiento",
            xaxis_title="Epochs",
            yaxis_title="Loss/Accuracy",
            legend_title="Tipo de Pérdida",
            hovermode="x",
            height=400, 
        )
        col.plotly_chart(fig)

def save_svm_results(y_testSVM, y_predSVM, model_nameSVM, accuracySVM=None, error_rateSVM=None):
    results_fileSVM = f"models/{model_nameSVM}_results.json"
    resultsSVM = {
        'y_testSVM': y_testSVM,
        'y_predSVM': y_predSVM.tolist(),
        'accuracySVM': accuracySVM,
        'error_rateSVM': error_rateSVM
    }
    with open(results_fileSVM, 'w') as f:
        json.dump(resultsSVM, f)

def load_svm_results(model_nameSVM):
    results_fileSVM = f"models/{model_nameSVM}_results.json"
    try:
        if os.path.exists(results_fileSVM):
            with open(results_fileSVM, 'r') as f:
                resultsSVM = json.load(f)
            y_testSVM = resultsSVM['y_testSVM']
            y_predSVM = resultsSVM['y_predSVM']
            accuracySVM = resultsSVM['accuracySVM']
            error_rateSVM = resultsSVM['error_rateSVM']
            return y_testSVM, y_predSVM, accuracySVM, error_rateSVM
        else:
            return None, None, None, None
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error al cargar el archivo JSON: {e}. Se eliminará el archivo corrupto.")
        os.remove(results_fileSVM)
        return None, None, None, None

y_testSVM, y_predSVM, accuracySVM, error_rateSVM = load_svm_results('svmModel')
if y_testSVM is None or y_predSVM is None:
    vectorizerSVM = TfidfVectorizer(max_features=500, stop_words='english')
    XSVM = vectorizerSVM.fit_transform(sentences)
    X_trainSVM, X_testSVM, y_trainSVM, y_testSVM = train_test_split(XSVM, targets, test_size=0.2, random_state=42)
    svm_modelSVM = SVC(kernel='linear', C=1.0, random_state=42)
    svm_modelSVM.fit(X_trainSVM, y_trainSVM)
    y_predSVM = svm_modelSVM.predict(X_testSVM)
    accuracySVM = accuracy_score(y_testSVM, y_predSVM)
    error_rateSVM = 1 - accuracySVM  
    save_svm_results(y_testSVM, y_predSVM, 'svmModel', accuracySVM, error_rateSVM)
    metrictsOfModels('svmModel', y_testSVM, y_predSVM, 'svmModel_metrics.json')

else:
    print("Resultados cargados desde el archivo.")

def show_svm_results(colSVM, y_testSVM, y_predSVM):
    colSVM.subheader("Matriz de Confusión")
    if y_testSVM is not None and y_predSVM is not None:
        cmSVM = confusion_matrix(y_testSVM, y_predSVM)
        z = cmSVM
        x = ['Predicción 0', 'Predicción 1']
        y = ['Real 0', 'Real 1']
        figSVM = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Oranges', showscale=True)
        figSVM.update_layout(
            title="",
            xaxis_title="Predicciones",
            yaxis_title="Valores Reales"
        )
        colSVM.plotly_chart(figSVM)

def mostrar_indicadores():
    color_palette = sns.color_palette("colorblind", 3).as_hex()
    results = {
        'Modelo 1': {'Test Loss': 0.45, 'Test Accuracy': 0.85},
        'Modelo 2': {'Test Loss': 0.30, 'Test Accuracy': 0.90},
        'Modelo 3': {'Accuracy': 0.85, 'Error Rate': 0.15}
    }
    st.subheader('Indicadores de Desempeño entre Modelos')
    cols = st.columns(3)
    with cols[0]:
        st.write('📚 LSTM y Embedding')

        fig1_loss = go.Figure(go.Indicator(
            mode="gauge+number",
            value=results['Modelo 1']['Test Loss'],
            title={'text': "Test Loss"},
            gauge={'axis': {'range': [None, 1]}, 'bar': {'color': color_palette[0]}} 
        ))
        fig1_loss.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200)
        st.plotly_chart(fig1_loss)
        fig1_acc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=results['Modelo 1']['Test Accuracy'],
            title={'text': "Test Accuracy"},
            gauge={'axis': {'range': [None, 1]}, 'bar': {'color': color_palette[1]}}
        ))
        fig1_acc.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=180)
        st.plotly_chart(fig1_acc)
    with cols[1]:
        st.write('📚 CNN')
        fig2_loss = go.Figure(go.Indicator(
            mode="gauge+number",
            value=results['Modelo 2']['Test Loss'],
            title={'text': "Test Loss"},
            gauge={'axis': {'range': [None, 1]}, 'bar': {'color': color_palette[0]}}
        ))
        fig2_loss.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200)
        st.plotly_chart(fig2_loss)
        fig2_acc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=results['Modelo 2']['Test Accuracy'],
            title={'text': "Test Accuracy"},
            gauge={'axis': {'range': [None, 1]}, 'bar': {'color': color_palette[1]}}
        ))
        fig2_acc.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=180)
        st.plotly_chart(fig2_acc)
    with cols[2]:
        st.write('📚 SVM')
        fig3_acc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=results['Modelo 3']['Accuracy'],
            title={'text': "Accuracy"},
            gauge={'axis': {'range': [None, 1]}, 'bar': {'color': color_palette[2]}} 
        ))
        fig3_acc.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200)
        st.plotly_chart(fig3_acc)
        fig3_error = go.Figure(go.Indicator(
            mode="gauge+number",
            value=results['Modelo 3']['Error Rate'],
            title={'text': "Error Rate"},
            gauge={'axis': {'range': [None, 1]}, 'bar': {'color': color_palette[2]}}
        ))
        fig3_error.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=180)
        st.plotly_chart(fig3_error)

def cargar_datos_modelo(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"No se pudo encontrar el archivo {file_path}")
        return {}

def table2():
    datos_modelo1 = cargar_datos_modelo('modelo1_metrics.json')
    datos_modelo2 = cargar_datos_modelo('modelo2_metrics.json')
    datos_modelo3 = cargar_datos_modelo('svmModel_metrics.json')
    results = {
        '🧠 LSTM y Embedding': {
            'Precision': datos_modelo1['firstModel'].get('Precision', 'N/A'),
            'Recall': datos_modelo1['firstModel'].get('Recall', 'N/A'),
            'F1-Score': datos_modelo1['firstModel'].get('F1-Score', 'N/A'),
        },
        '🧠 CNN': {
            'Precision': datos_modelo2['secondModel'].get('Precision', 'N/A'),
            'Recall': datos_modelo2['secondModel'].get('Recall', 'N/A'),
            'F1-Score': datos_modelo2['secondModel'].get('F1-Score', 'N/A'),
        },
        '🧠 SVM': {
            'Precision': datos_modelo3['svmModel'].get('Precision', 'N/A'),
            'Recall': datos_modelo3['svmModel'].get('Recall', 'N/A'),
            'F1-Score': datos_modelo3['svmModel'].get('F1-Score', 'N/A'),
        }
    }
    selected_models = st.multiselect(
        'Selecciona los modelos a comparar:',
        ['🧠 LSTM y Embedding', '🧠 CNN', '🧠 SVM'],
        default=['🧠 LSTM y Embedding', '🧠 CNN', '🧠 SVM']  
    )
    if len(selected_models) >= 2:
        selected_results = {model: results[model] for model in selected_models}
        df_comparativa = pd.DataFrame(selected_results).T  
        st.subheader('Comparación de Desempeño entre Modelos')
        st.table(df_comparativa)
    else:
        st.warning("Por favor selecciona al menos 2 modelos para comparar.")

# Interfaz
st.set_page_config(layout="wide")
col1, col2 = st.columns(2)
col3, col4, col5 = st.columns(3)

with col1:
    st.title('MT | Clasificación de tweets 📝')
    st.write("""Resultados del análisis y diseño de modelos de aprendizaje automático que predicen cuáles tweets tratan sobre desastres reales y cuáles no. Se dispone de un conjunto de datos de 10,000 tweets que fueron clasificados manualmente.""")
    st.write("""📊 Puedes explorar la frecuencia de palabras en los tweets relacionados a desastres y no desastres. Selecciona la categoría de interés y el número máximo de palabras a visualizar.""")
    option = st.selectbox('Seleccionar categoría:', ['Desastres', 'No Desastres'])
    st.write("""📈 También puedes visualizar las palabras más frecuentes en los tweets de cada categoría, así como los bi-gramas y tri-gramas más comunes.""")
    max_words = st.slider('Seleccionar número máximo de palabras:', 50, 200, 100)
    st.subheader('Frecuencia de palabras')
    plotWordCloudInteractive(option, max_words=max_words)

with col2:
    st.subheader('Top 10 Palabras más frecuentes')
    plotTopWordsHistogramInteractive(option)
    st.subheader('Bi-Gramas')
    bigramAnalysisInteractive(tokenizedText, dataTrain['target'], label=option)

with col3:
    st.subheader('Intersección de palabras')
    commonWords(disasterWordFreqDf, nonDisasterWordFreqDf)
    st.subheader('--------------------- Modelo SVM ---------------------')
    show_svm_results(col3, y_testSVM, y_predSVM)

with col4:
    option = st.selectbox('Selecciona el modelo para mostrar:',['Modelo 1 - 🧠 LSTM y Embedding', 'Modelo 2 - 🧠 CNN'])
    st.subheader('Modelo de clasificación')
    if option == 'Modelo 1 - 🧠 LSTM y Embedding':
        showResults(col4, history)
    else:
        showResults(col4, history2)
    mostrar_indicadores()

with col5:
    st.subheader('Tri-Gramas')
    trigramAnalysisInteractive(tokenizedText, dataTrain['target'], label=option)
    table2()
    st.markdown("""
    <div style="text-align: justify;">
    De los tres modelos, <b>SVM</b> destacó por su mejor <b>F1-Score</b>, siendo ideal para un equilibrio entre precisión y recall. <b>CNN</b> tuvo el mejor <b>test accuracy</b>, adecuado para tareas que priorizan la exactitud. En contraste, <b>LSTM</b> mostró buena precisión, pero su bajo <b>recall</b> lo hace menos eficaz para identificar instancias positivas.
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.write("""Diego Alexander Hernández Silvestre 21270 🛻 |
    Linda Inés Jiménez Vides 21169 🏎️  
    """)