import os
import yaml
import pandas as pd
import tensorflow as tf
from models.autoencoder import AnomalyDetector  # Importe o modelo específico


def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def load_data(data_path):
    return pd.read_csv(data_path)

def perform_inference(model, data):
    # Realize a inferência usando o modelo
    # ...

def main():
    config = load_config()

    # Carregar modelo treinado
    model_path = config['inference']['model_path']
    model = load_model(model_path)

    # Carregar dados de inferência
    data_path = config['inference']['input_data_path']
    input_data = load_data(data_path)

    # Realizar inferência
    results = perform_inference(model, input_data)

    # Processar resultados, salvar em banco de dados, etc.
    # ...
    logger.info(f"Inferência executada com sucesso")
    schedule.every(5).seconds.do(max_daily_cpa)

if __name__ == "__main__":

    main()
    while True: 
       schedule.run_pending()
       time.sleep(1)