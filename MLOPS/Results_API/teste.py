import random
from datetime import datetime
import requests


def test_hello_world():
    url = "http://localhost:5000/hello"  
    response = requests.get(url)

    if response.status_code == 200:
        print("Sucesso!")
        print("Resposta da API:", response.json())
    else:
        print(f"Falha! Código de status: {response.status_code}")
# Função para enviar dados para a API
def send_to_api(data, api_url):
    try:
        response = requests.post(api_url, json=data)
        response.raise_for_status()  # Lança uma exceção para erros HTTP
        return response.json()
    except Exception as e:
        print(f"Erro ao enviar post para a API: {str(e)}")
        return None

# Criar valores aleatórios para o campo 'values'
random_values = [random.uniform(-1, 1) for _ in range(140)]

# Dados de inferência ECG
inference_data = {
    'dt_measure': '2024-01-26T20:28:00.999737',
    'anomalous': True,
    'model_tag': 'Model-v0.0.1',
    'values': random_values
}

# URL da rota da API
api_url = "http://localhost:5000/ecg"

# Enviar dados para a API
response = send_to_api(inference_data, api_url)

if response:
    print("Resposta da API:", response)
else:
    print("Falha ao obter resposta da API.")


test_hello_world()