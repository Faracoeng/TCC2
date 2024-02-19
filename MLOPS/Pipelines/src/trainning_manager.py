import os
from models.autoencoder import AnomalyDetector
from datetime import datetime
from utils.ecg import ECG
from utils.model import *
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging.config
from database import *
import pickle

# Teste:
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

#logging.config.fileConfig('logging.conf')
# No docker
logging.config.fileConfig('/app/src/logging.conf')
logger = logging.getLogger()

autoencoder = AnomalyDetector()
ecg = ECG()


def get_environment_variables():
    try:
        processing_variables = {
            "begin_date": os.environ.get('BEGIN_DATE'),#"2023-11-24",
            "end_date":  os.environ.get('END_DATE'),#"2023-11-28",
            "test_size": os.environ.get('TEST_SIZE'),#0.2,
            "random_state": os.environ.get('RANDOM_STATE'),#21,
            "optimizer": os.environ.get('OPTIMIZER'),#"adam",
            "loss_function": os.environ.get('LOSS_FUNCTION'),#"mae", 
            "epochs": os.environ.get('EPOCHS'),#20,
            "batch_size": os.environ.get('BATCH_SIZE')#512
        }
        logger.info("Variáveis de ambiente de processamento dos dados carregadas com sucesso")

    except Exception as e:
        logger.error(f"Erro ao obter variáveis de ambiente de processamento dos dados: {str(e)}")

    return processing_variables

def get_train_data():
    try:
        start_date = get_environment_variables()["begin_date"],#"2023-11-24" 
        end_date = get_environment_variables()["end_date"],#"2023-11-28"
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        raw_data = ecg.get_ECG_train_data(start_date, end_date)
        logger.info("Dados de treinamento obtidos da base de dados")
        return raw_data
    except Exception as e:
        logger.error("Certifique-se de que as datas estão no formato 'YYYY-MM-DD'.")

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def get_model_stats(predictions, labels):
    logger.info("Accuracy = {}".format(accuracy_score(labels, predictions)))
    logger.info("Precision = {}".format(precision_score(labels, predictions)))
    logger.info("Recall = {}".format(recall_score(labels, predictions)))
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)

    return accuracy, precision, recall

def train_manager(model_tag):
    try:

        raw_data = get_train_data()
        labels = ecg.get_labels(raw_data)
        data = ecg.get_ecg_points(raw_data)

        train_data, test_data, train_labels, test_labels = ecg.get_train_test_split(data, labels, float(get_environment_variables()['test_size']), int(get_environment_variables()['random_state']))
     
        max_value, min_value = ecg.get_min_max_val(train_data)
        
        # Convertendo o tensor para um tipo de dados Python nativo
        max_value_python = max_value.numpy()
        min_value_python = min_value.numpy()

        normalazed_train_data = ecg.normalize_data(train_data, max_value, min_value)
        normalazed_test_data = ecg.normalize_data(test_data, max_value, min_value)

        # O autoencoder sera treinado usando apenas os ritmos normais, 
        # que são rotulados neste conjunto de dados como 1. Separando os ritmos normais dos ritmos anormais:
        # Convertendo os rótulos de treinamento (train_labels) e rótulos de teste (test_labels) para o tipo de dados booleano (bool).
        #  Isso é feito para garantir que os rótulos sejam interpretados como valores booleanos verdadeiros ou falsos.
        normal_train_data, normal_test_data = ecg.normal_train_test_generate(normalazed_train_data, normalazed_test_data, train_labels, test_labels)
        #print(normal_train_data)
        # Adicionando Keras no TensorFlow especificando o otimizador e a função de perda para definir o modelo através da função compile()
        #optimizer='adam': O otimizador é o algoritmo utilizado para 
        # ajustar os pesos do modelo durante o treinamento, com o objetivo
        #  de minimizar a função de perda. Neste caso, o otimizador Adam adapta as 
        # taxas de aprendizado dos parâmetros do modelo durante o treinamento.

        #loss='mae': A função de perda (loss) é uma métrica que o modelo tentará minimizar
        #  durante o treinamento. 'mae' refere-se à "Mean Absolute Error" (Erro Médio Absoluto),
        #  que é uma medida da média das diferenças absolutas entre as previsões do modelo e os 
        # rótulos reais. Neste contexto de autoencoder, o modelo tentará minimizar a diferença 
        # absoluta média entre as entradas e suas reconstruções.
        
        #autoencoder.compile(optimizer='adam', loss='mae') 
        autoencoder.compile(optimizer=get_environment_variables()['optimizer'], loss=get_environment_variables()['loss_function'])
        # Treinando por 20 epocas
        # normal_train_data são os dados de treinamento.

        #epochs=20: Indica o número de vezes que o modelo passará por todo o conjunto de 
        # treinamento durante o treinamento. Neste caso, o modelo será treinado por 20 épocas.

        #batch_size=512: Define o tamanho do lote usado para cada atualização de peso durante o treinamento. 
        # Isso significa que o modelo será atualizado a cada 512 exemplos. 
        # O tamanho do lote é uma consideração importante para a eficiência do treinamento.

        # validation_data=(test_data, test_data): Define os dados de validação usados durante o treinamento.
        #  Os dados de validação não são usados para treinar o modelo, mas sim para avaliar o desempenho 
        # do modelo em um conjunto de dados separado. Aqui, test_data é usado para validação.

        # shuffle=True: Embaralha os dados antes de cada época. 
        # Isso para garantir que o modelo não aprenda a ordem específica dos exemplos.
        history = autoencoder.fit(normal_train_data, normal_train_data, 
                epochs=int(get_environment_variables()['epochs']),
                batch_size=int(get_environment_variables()['batch_size']),
                validation_data=(normalazed_test_data, normalazed_test_data),
                #validation_data=(test_data, test_data),
                shuffle=True)
        
        
        
        # # Parte de Detecção de Anomalias

        # # Detecte anomalias calculando se a perda de reconstrução 
        # # é maior que um limiar fixo. Neste tutorial, você calculará 
        # # o erro médio médio para exemplos normais do conjunto de treinamento 
        # # e classificará exemplos futuros como anômalos se o erro de reconstrução 
        # # for maior que um desvio padrão do conjunto de treinamento.

        # # normal_train_data é um conjkunto de dados especifico para treinamento, 
        # # no qual deve conter apenas dados normais, sem anomalias

        # # As reconstruções dos dados de treinamento são calculadas e o erro de reconstrução
        # # é definido, então em tf.keras.losses, é utilizado o erro medio "mae".
        # # Isso vai definir um erro medio de treinamento, ou seja
        # # uma margem de segurança para definir um limiar de erro gerado no treinamento 
        # # que sera referencia nas inferencias de anomalias.
        normal_train_data_reconstructions = autoencoder.predict(normal_train_data)
        normal_train_data_reconstructions_train_loss = tf.keras.losses.mae(normal_train_data_reconstructions, normal_train_data)

        # # Com isso se define um threshold, existem diversas estratégias aqui sera utilizada esta.
        # # É definido pela média dos erros de treinamento mais o desvio padrão dos erros de treinamento.
        threshold = np.mean(normal_train_data_reconstructions_train_loss) + np.std(normal_train_data_reconstructions_train_loss)
        print("Threshold: ", threshold)

        preds = predict(autoencoder, normalazed_test_data, threshold)
        
        accuracy, precision, recall = get_model_stats(preds, test_labels)

        # Salve os pesos do modelo
        model_weights = autoencoder.get_weights()
        # Salvar no banco max/min value
        # Salvar no banco threshold
        # Salvar no banco modelo

        # Salvar no banco max/min value, threshold e modelo
        save_model_to_database(model_tag, max_value_python, min_value_python, threshold, model_weights, accuracy, precision, recall)
       
        #return autoencoder, train_data, test_data, train_labels, test_labels, normal_train_data, normal_test_data 

    except Exception as e:
        logger.error(f"Erro ao criar modelo: {e}")



def save_model_to_database(tag, max_value, min_value, threshold, model_weights, accuracy, precision, recall):
    try:
        #engine = create_database_engine(db_origem)
        Base.metadata.create_all(engine_Pipelines)

        Session = sessionmaker(bind=engine_Pipelines)
        session = Session()

        # Serializar os pesos do modelo
        model_weights_serialized = pickle.dumps(model_weights)

        model_info = Model(
            tag=tag,
            max_value=max_value,
            min_value=min_value,
            threshold=threshold,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            model_weights=model_weights_serialized 
        )

        session.add(model_info)
        session.commit()

        logger.info(f"Modelo e estatisticas salvas na base com a tag '{tag}'")

    except Exception as e:
        logger.error(f"Erro ao salvar o modelo na base: {str(e)}")





if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Por favor, forneça o nome do modelo como argumento.")
        sys.exit(1)

    model_tag_argument = sys.argv[1]
    logger.info(f"Argumento do modelo recebido: {model_tag_argument}")
    train_manager(model_tag_argument)