import requests
from datetime import datetime
import logging

class FastAPIClient:
    def __init__(self):
        #Valores default
        self.base_url = "http://localhost:5000"
        self.host = "localhost"
        self.port = "5000"

    def set_host(self, host):
        self.host = host
        self.base_url = f"http://{self.host}:{self.port}"

    def set_port(self, port):
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"

    def send_api_post(self, route, data):
         # Remove barras extras no início da rota
        route = route.lstrip('/')
        try:
            response = requests.post(f"{self.base_url}/{route}", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao enviar post para a API: {str(e.response.text)}")
            return None

    # def get_prediction_api(self, prediction_id):
    #     try:
    #         response = requests.get(f"{self.base_url}/predictions/{prediction_id}")
    #         response.raise_for_status()
    #         return response.json()
    #     except requests.exceptions.RequestException as e:
    #         logging.error(f"Erro ao obter predição da API: {str(e)}")
    #         return None

    # def get_ecg_api(self, inference_ecg_id):
    #     try:
    #         response = requests.get(f"{self.base_url}/inference_ecg/{inference_ecg_id}")
    #         response.raise_for_status()
    #         return response.json()
    #     except requests.exceptions.RequestException as e:
    #         logging.error(f"Erro ao obter inferência ECG da API: {str(e)}")
    #         return None
        
    def get_base_url(self):
        return self.base_url


