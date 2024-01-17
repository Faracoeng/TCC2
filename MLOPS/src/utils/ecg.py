import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from database import *
from sqlalchemy import text

import tensorflow as tf


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from models.autoencoder import AnomalyDetector

autoencoder = AnomalyDetector()

# Classe para representar os 140 pontos do ECG, e realziar as devidas tratativas no dados
class ECG:
    # Inferencia
    def get_ECG_inference_data(self):
        try:
            engine_DB = get_engine()
            #ecg_query = text("SELECT * FROM ECG WHERE dt_measure=CURDATE()")
            #ecg_query = text("SELECT * FROM ECG")
            ecg_query = text("SELECT * FROM ECG ORDER BY dt_measure DESC LIMIT 1")
            dataframe = pd.read_sql(ecg_query, engine_DB)
            # Pegando apenas os 140 pontos do ECG
            dataframe = dataframe.iloc[:, :-2]
            raw_data = dataframe.values
            logger.info("Dados diários para inferência obtidos da base de dados")
            #print(dataframe.head())
            return raw_data

        except Exception as e:
            logger.error(f"Erro ao obter ECG's diários da base de dados para inferência: {str(e)}")

    # Modelo
    def get_ECG_train_data(self, begin_date, end_date):
        try:
            engine_DB = get_engine()
            ecg_query = text(f"SELECT * FROM ECG WHERE dt_measure BETWEEN '{begin_date}' AND '{end_date}'")
            #ecg_query = text("SELECT * FROM ECG")
            dataframe = pd.read_sql(ecg_query, engine_DB)
            # Pegando apenas os 140 pontos do ECG mais o rotulo
            dataframe = dataframe.iloc[:, :-2]
            raw_data = dataframe.values
            logger.info("Dados obtidos da base de dados com sucesso")
            #print(dataframe.head())
            return raw_data

        except Exception as e:
            logger.error(f"Erro ao obter ECG's da base de dados: {str(e)}")
    # Os dados do dataset foram rotulados para 0 e 1, onde:
    # 0 = anomalia
    # 1 = normal 
    # existem de 0 a 140 colunas, onde a ultima representa o label
    #Modelo
    def get_labels(self, raw_data):
        # Pegando o label
        labels = raw_data[:, -1]
        return labels

    #Modelo//Inferencia
    def get_ecg_points(self, raw_data):
        # Pegando os 140  dados de cada ECG
        data = raw_data[:, 0:-1]
        return data


    #  A função train_test_split() do numpy é usada para dividir 
    # um conjunto de dados em conjuntos de treinamento e teste, 
    # onde test_size indica a proporção de dados a serem usados para 
    # teste (neste caso, 20%), e random_state é uma semente para garantir reprodutibilidade.

    # raw_data é uma matriz que contém tanto os dados quanto os rótulos, 
    # os conjuntos resultantes (train_data, test_data, train_labels, test_labels) 
    # são utilizados para treinar e testar um modelo de aprendizado de máquina.
    #Modelo
    def get_train_test_split(self, data, labels, test_size, random_state):
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, random_state=random_state)#(data, labels, test_size=0.2, random_state=21)
        return train_data, test_data, train_labels, test_labels
    #Modelo//Inferencia
    def get_min_max_val(self, train_data):
        # Agora normalizando os dados entre 0 e 1 na amplitude.

        # Calculando o valor mínimo e máximo dos dados de 
        # treinamento (train_data) usando as funções tf.reduce_min e tf.reduce_max. 
        # Esses valores serão usados para normalizar os dados.
        min_val = tf.reduce_min(train_data)
        max_val = tf.reduce_max(train_data)
        return max_val, min_val

    def normalize_data(self, data, max_val, min_val):

        # normalizando os dados de treinamento (train_data) e de teste (test_data). 
        # A normalização é realizada subtraindo o valor mínimo (min_val) de cada 
        # elemento dos dados e, em seguida, dividindo pelo intervalo (diferença entre o valor máximo e mínimo).
        #  Isso resulta em todos os valores dos dados estando na faixa de 0 a 1.
        data = (data - min_val) / (max_val - min_val)
        #test_data = (test_data - min_val) / (max_val - min_val)


        # convertendo os dados normalizados para o tipo de dados float32. 
        # Isso é feito para garantir que os dados estejam no formato apropriado para cálculos numéricos 
        # e para compatibilidade com operações específicas do TensorFlow.
        data = tf.cast(data, tf.float32)
 
        return data


    def normal_train_test_generate(self, train_data, test_data, train_labels, test_labels):
            # Criando conjuntos de dados (normal_train_data e normal_test_data) 
            # contendo apenas os exemplos que têm rótulos verdadeiros, ou seja, 
            # os exemplos rotulados como normais. Isso é feito indexando os 
            # conjuntos de dados originais (train_data e test_data) usando os rótulos booleanos correspondentes.
            train_labels = train_labels.astype(bool)
            test_labels = test_labels.astype(bool)
            normal_train_data = train_data[train_labels]
            normal_test_data = test_data[test_labels]
            return normal_train_data, normal_test_data
    
    def anomalous_train_test_generate(train_data, test_data, train_labels, test_labels):
        # Criando conjuntos de dados (anomalous_train_data e anomalous_test_data) 
        # contendo apenas os exemplos que têm rótulos falsos, ou seja, 
        # os exemplos rotulados como anormais. O operador ~ é usado para 
        # inverter os valores booleanos em train_labels e test_labels, selecionando assim os exemplos anormais.

        anomalous_train_data = train_data[~train_labels]
        anomalous_test_data = test_data[~test_labels]
        return anomalous_train_data, anomalous_test_data