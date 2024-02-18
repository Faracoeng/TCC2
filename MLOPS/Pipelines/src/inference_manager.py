from models.autoencoder import AnomalyDetector
from datetime import datetime
from utils.ecg import ECG
from utils.model import *
from utils.fastapi import *
import tensorflow as tf
import logging.config
from database import *
import pickle
import time
# PARA TESTE
import matplotlib.pyplot as plt
import numpy as np
from trainning_manager import *
#from front_end import *

logging.config.fileConfig('logging.conf')
# No docker
#logging.config.fileConfig('/app/src/logging.conf')
logger = logging.getLogger()

autoencoder = AnomalyDetector()
ecg = ECG()

def load_model_from_database(tag):
    try:
        engine = create_database_engine(db_origem)
        Session = sessionmaker(bind=engine)
        session = Session()

        model_info = session.query(Model).filter_by(tag=tag).first()
        if model_info:
            # Desserializar os pesos do modelo
            model_weights_deserialized = pickle.loads(model_info.model_weights)

            # Criar um novo modelo
            new_model = AnomalyDetector()

            # Chamar a função build() ou passar um conjunto de dados de treinamento
            # para o modelo antes de definir os pesos
            new_model.build((1, 140))  # Substitua input_dim pelo número de recursos do seu modelo

            # Definir os pesos
            new_model.set_weights(model_weights_deserialized)

            return new_model

    except Exception as e:
        logger.error(f"Erro ao realizar inferência: {str(e)}")
        return None

def get_model_max_min_values(tag):
    try:
        #Session = sessionmaker(bind=engine_orig)
        session = get_session_Pipelines() #Session()

        model_info = session.query(Model).filter_by(tag=tag).first()

        if model_info:
            return model_info.max_value, model_info.min_value
        else:
            logger.error(f"Modelo com a tag '{tag}' não encontrado na base de dados.")
            return None, None

    except Exception as e:
        logger.error(f"Erro ao realizar inferência: {str(e)}")
        return None, None

def get_model_threshold(tag):
    try:
        
        #Session = sessionmaker(bind=engine_orig)
        session = get_session_Pipelines() #Session()

        model_info = session.query(Model).filter_by(tag=tag).first()

        if model_info:
            return model_info.threshold
        else:
            logger.error(f"Modelo com a tag '{tag}' não encontrado na base de dados.")
            return None

    except Exception as e:
        logger.error(f"Erro ao realizar inferência: {str(e)}")
        return None

def get_api_environment():
    try:
        api_envs = {
            "api_port": "5000",#os.environ.get('API_PORT'),
            "api_host": "localhost",#os.environ.get('API_HOST'),
            "predictions_route": "/predictions",#os.environ.get('PREDICT_ROUTE'),
            "ecg_route": "/ecg"#os.environ.get('ECG_ROUTE')
        }
        logger.info("Variáveis de ambiente da API carregadas com sucesso")

    except Exception as e:
            logger.error(f"Erro ao obter variáveis de ambiente da API: {str(e)}")

    return api_envs

def api_client(data, host, port, route):
    try:
        client = FastAPIClient()
        client.set_host(host)
        client.set_port(port)
        logger.info(f"Enviando dados para a rota {route}.")
        response = client.send_api_post(route, data)
        logger.info("Resposta da API: %s", response)
        logger.info("Post enviado com sucesso para a API")
    except Exception as e:
        logger.error(f"Erro ao enviar dados para a API: {e}")

def inference_manager(model_tag):

    data= ecg.get_ECG_inference_data()
    print(data)
    max_value, min_value = get_model_max_min_values(model_tag)
    print(max_value)
    print(min_value)
    # Convertendo o tensor para um tipo de dados Python nativo
    #max_value_inference = max_value.numpy()
    #min_value_inference = min_value.numpy()
    
    # Converter o tensor TensorFlow em um array NumPy
    normalazed_inference_data = ecg.normalize_data(data, max_value, min_value)
    #print(normalazed_inference_data)
    # Dados de validação
    #tensor = tf.constant([[0.6557866, 0.23823199, 0.09261786, 0.00404374, 0., 0.13859433, 0.33756405, 0.39341703, 0.48074687, 0.59964925, 0.6171354, 0.5974244, 0.608207, 0.6255347, 0.5968119, 0.6204715, 0.6184385, 0.616504, 0.6244116, 0.6096653, 0.6005445, 0.60949856, 0.61701095, 0.60398585, 0.60058343, 0.6077576, 0.6019527, 0.5881163, 0.5999749, 0.56631666, 0.5651372, 0.5817577, 0.57147187, 0.5632972, 0.5546873, 0.569688, 0.57254976, 0.57474476, 0.58706576, 0.5820871, 0.5972296, 0.60187036, 0.62669706, 0.6372963, 0.65385884, 0.64941543, 0.66934747, 0.66300243, 0.6784773, 0.6826204, 0.6840448, 0.6921553, 0.6889558, 0.70773715, 0.6928871, 0.7196199, 0.7126665, 0.70329547, 0.70068544, 0.7107365, 0.7255777, 0.7235762, 0.7355486, 0.74178785, 0.73832136, 0.747157, 0.74659276, 0.76908624, 0.76147276, 0.7650764, 0.7599358, 0.7666093, 0.7685559, 0.7740345, 0.7784174, 0.77988243, 0.7754697, 0.7616591, 0.7712477, 0.7676765, 0.7606266, 0.7451672, 0.7412053, 0.7451208, 0.73961717, 0.73584336, 0.73748374, 0.7288561, 0.736305, 0.7456426, 0.7303382, 0.72506374, 0.707149, 0.71523845, 0.7190325, 0.7136336, 0.72578484, 0.7376195, 0.76332206, 0.80546945, 0.85357016, 0.8666188, 0.89362997, 0.9346589, 0.98057365, 1., 0.9796889, 0.9702994, 0.9495639, 0.9072417, 0.8655404, 0.82671213, 0.74749696, 0.676652, 0.6431136, 0.6391139, 0.63560396, 0.6339185, 0.6283134, 0.6336423, 0.6380438, 0.623501, 0.6285979, 0.6241552, 0.6171591, 0.6125752, 0.6321151, 0.6335911, 0.62868893, 0.6481243, 0.6977576, 0.79494005, 0.81668514, 0.79567677, 0.7620936, 0.7127378, 0.7081753, 0.6920793, 0.81541544, 0.7028011]], shape=(1, 140), dtype=tf.float32)

    try:
        print("oi")
        # Carregar o modelo
        model = load_model_from_database(model_tag)
        #print(model.weights())
        #model_tag = "testedomingo"
        #model, train_data, test_data, train_labels, test_labels, normal_train_data, normal_test_data  = train_manager(model_tag)
        
        tensor = tf.constant([ 0.5594925, 0.554532, 0.4411452, 0.29301804, 0.20673993, 0.13617955,
    0.23012452, 0.3582632, 0.4119776, 0.43589172, 0.54114586, 0.5615606,
    0.54728544, 0.53517616, 0.5461514, 0.55051035, 0.5386099, 0.54031754,
    0.5396282, 0.5340332, 0.5346705, 0.5284389, 0.5306547, 0.52619225,
    0.51002693, 0.5160073, 0.50864756, 0.4989328, 0.5039928, 0.49604866,
    0.4829037, 0.4687068, 0.46690515, 0.4598262, 0.44844124, 0.44661075,
    0.4511718, 0.44730267, 0.43892437, 0.441952, 0.44708136, 0.46447277,
    0.47347665, 0.48422554, 0.49670303, 0.49174502, 0.516449, 0.5333947,
    0.5300341, 0.5370003, 0.54380476, 0.5428671, 0.55238664, 0.5385617,
    0.54539156, 0.54864514, 0.5563505, 0.56435835, 0.5547315, 0.5641891,
    0.56863415, 0.5663354, 0.5608818, 0.5647332, 0.5602431, 0.57787484,
    0.5875947, 0.59140855, 0.595485, 0.5982221, 0.60393333, 0.6164843,
    0.6339836, 0.6299771, 0.63317496, 0.62009984, 0.6378259, 0.6246068,
    0.6230249, 0.62436765, 0.6158669, 0.6252278, 0.6185154, 0.61229885,
    0.61180586, 0.6189899, 0.60373217, 0.6026029, 0.6038754, 0.6085527,
    0.60905504, 0.59532374, 0.60156167, 0.60111207, 0.59593785, 0.6008833,
    0.58934414, 0.6033332, 0.59252363, 0.5931327, 0.6045386, 0.59668803,
    0.5995459, 0.59796077, 0.60235035, 0.6004622, 0.61265093, 0.66024756,
    0.7008846, 0.6947817, 0.68232197, 0.67445105, 0.6607629, 0.6563071,
    0.65899837, 0.6529679, 0.6232044, 0.5794365, 0.56646353, 0.5692847,
    0.56394845, 0.566073, 0.56956583, 0.5672482, 0.56816804, 0.57102937,
    0.566824, 0.56152886, 0.5617077, 0.58328736, 0.57942736, 0.6216721,
    0.71476597, 0.6967847, 0.69656545, 0.70875186, 0.6906813, 0.6397422,
    0.56505466, 0.61712885], dtype=tf.float32)
                    
        # expandindo a dimensão do segundo tensor
        tensor = tf.expand_dims(tensor, axis=0)
        if model:
            
            api_environment = get_api_environment()

            #encoded_data = model.encoder(normalazed_inference_data).numpy()
            #reconstructions_data = model.decoder(encoded_data).numpy()
            #print(normalazed_inference_data)
            #print(normal_train_data[0])
            #new_normal_train_data = tf.expand_dims(normal_train_data[0], axis=0)
            predictions = model.predict(normalazed_inference_data)

            data_reconstructions_loss = tf.keras.losses.mae(predictions, normalazed_inference_data)
            print(data_reconstructions_loss)
            # print(data_reconstructions_loss)
            # plt.hist(data_reconstructions_loss[None,:], bins=50)
            # plt.xlabel("Train loss")
            # plt.ylabel("No of examples")
            # plt.show()

            #print(data_reconstructions_loss)
            #print(reconstructions_data)
            #print(predictions)
            # MEU TESTE-----------------------------
            plt.plot(normalazed_inference_data[0], 'b')
            plt.plot(predictions[0], 'r')
            plt.fill_between(np.arange(140), predictions[0], normalazed_inference_data[0], color='lightcoral')
            plt.legend(labels=["Input", "Reconstruction", "Error"])
            plt.show()
            #-----------------------------------------

            # Realizar inferência nos novos dados
            #predictions = model.predict(normalazed_inference_data)
            #data_reconstructions_loss = tf.keras.losses.mae(predictions[0], normalazed_inference_data[0])
            #inference_threshold = np.mean(data_reconstructions_loss) + np.std(data_reconstructions_loss)
            model_threshold = get_model_threshold(model_tag)

            #plt.hist(data_reconstructions_loss[None,:], bins=50)
            #plt.xlabel("Train loss")
            #plt.ylabel("No of examples")
            #plt.show()

            # Se o erro da inferência for maior que o threshold do modelo utilizado, então é uma anomalia
            print(f"data_reconstructions_loss é:   ---> {data_reconstructions_loss}")
            print(f"model_threshold é:   ---> {model_threshold}")

            #print(data_reconstructions_loss)
            # Se o erro da inferência for maior que o threshold do modelo utilizado, então é uma anomalia
            anomaly_detected = float(data_reconstructions_loss) > float(model_threshold)
            if anomaly_detected:
                logger.info("Anomalia detectada")
            #print(anomaly_detected)
            #print(tf.math.less(data_reconstructions_loss, model_threshold))

            # Montar os dados para API (ECG)
            inference_ecg_data = {
                "dt_measure": datetime.utcnow().isoformat(),
                "is_anomalous": anomaly_detected,
                "model_tag": model_tag,
                "values": normalazed_inference_data[0]  # Convertendo o array NumPy para uma lista
            }

            # Enviando dados para a API (ECG)

            #api_client(inference_ecg_data, api_environment['api_host'], api_environment['api_port'], api_environment['ecg_route'])
            
            # Enviar dados para a API (predictions)
            prediction_data = {
                "dt_measure": datetime.utcnow().isoformat(),
                "model_tag": model_tag,
                "values": [],  
            }


            # Adicione os valores à lista
            enumerated_predictions = list(enumerate(predictions[0]))
            for i, value in enumerated_predictions:
                prediction_data["values"].append(float(value))

            #api_client(prediction_data, api_environment['api_host'], api_environment['api_port'], api_environment['predictions_route'])
         
            time.sleep(2)
            
            #plot_and_update_data()

            logger.info("Inferência concluída com sucesso")
        else:
            logger.error("Falha ao realizar inferência. Modelo não encontrado.")
            return None

    except Exception as e:
        logger.error(f"Erro ao realizar inferencia: {str(e)}")
        return None



if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Por favor, forneça o nome do modelo como argumento que deseja utilizar na inferência.")
        sys.exit(1)

    model_tag = sys.argv[1]
    logger.info(f"Argumento do modelo recebido: {model_tag}")
    inference_manager(model_tag)