from models.autoencoder import AnomalyDetector
from datetime import datetime
from utils.ecg import ECG
from utils.model import *
from utils.fastapi import *
import tensorflow as tf
import logging.config
from database import *
import pickle

logging.config.fileConfig('logging.conf')
# No docker
#logging.config.fileConfig('/app/src/logging.conf')
logger = logging.getLogger()

autoencoder = AnomalyDetector()
ecg = ECG()

def load_model_from_database(tag):
    try:
        engine = create_database_engine(database_configs)
        Session = sessionmaker(bind=engine)
        session = Session()

        model_info = session.query(Model).filter_by(tag=tag).first()

        if model_info:
            model = AnomalyDetector()
            # Certifique-se de que os pesos do modelo são criados chamando a função build() ou
            # passando um conjunto de dados de treinamento durante a criação do modelo.
            model.build((1, 140))  # Substitua input_dim pelo número de recursos do seu modelo
            model.set_weights(pickle.loads(model_info.model_weights))
            return model
        else:
            logger.error(f"Modelo com a tag '{tag}' não encontrado na base de dados.")
            return None

    except Exception as e:
        logger.error(f"Erro ao realizar inferência: {str(e)}")
        return None

def get_model_threshold(tag):
    try:
        engine_DB = get_engine()
        Session = sessionmaker(bind=engine_DB)
        session = Session()

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
    
    client = FastAPIClient()
    client.set_host(host)
    client.set_port(port)
    client.send_api_post(route, data)
    logger.info("Post enviado com sucesso para a API")

def inference_manager(model_tag):

    data= ecg.get_ECG_inference_data()
    try:
        # Carregar o modelo
        model = load_model_from_database(model_tag)

        if model:
            
            api_environment = get_api_environment()
            # Realizar inferência nos novos dados
            predictions = model.predict(data)
            data_reconstructions_loss = tf.keras.losses.mae(predictions, data)
            #inference_threshold = np.mean(data_reconstructions_loss) + np.std(data_reconstructions_loss)
            model_threshold = get_model_threshold(model_tag)

            # Se o erro da inferência for maior que o threshold do modelo utilizado, então é uma anomalia
            print(f"data_reconstructions_loss é:   ---> {data_reconstructions_loss}")
            print(f"model_threshold é:   ---> {model_threshold}")
            #print(predictions)

            # Se o erro da inferência for maior que o threshold do modelo utilizado, então é uma anomalia
            anomaly_detected = float(data_reconstructions_loss) > float(model_threshold)
            # Montar os dados para API (ECG)
            inference_ecg_data = {
                "dt_measure": datetime.utcnow().isoformat(),
                "is_anomalous": anomaly_detected,
                "model_tag": model_tag,
                "values": data[0].tolist()  # Convertendo o array NumPy para uma lista
            }
            #print(inference_ecg_data)


            # Enviando dados para a API (ECG)
            api_client(inference_ecg_data, api_environment['api_host'], api_environment['api_port'], api_environment['ecg_route'])

            # Enviar dados para a API (predictions)
            prediction_data = {
                "dt_measure": datetime.utcnow().isoformat(),
                "model_tag": model_tag,
                "values": [],  # Adicione esta linha para incluir a lista de valores
            }

            # Adicione os valores à lista
            enumerated_predictions = list(enumerate(predictions[0]))
            for i, value in enumerated_predictions:
                prediction_data["values"].append(float(value))

            
            api_client(prediction_data, api_environment['api_host'], api_environment['api_port'], api_environment['predictions_route'])


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