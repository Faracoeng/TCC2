from database import *
import pandas as pd
from sqlalchemy import text
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Carregar a configuração do logger a partir do arquivo logging.conf
#logging.config.fileConfig('logging.conf')
#logger = logging.getLogger('fastapi')

def get_ECG_inference_data():
    try:
        engine_DB = get_engine()
        ecg_query = text("SELECT * FROM Inference_ECG ORDER BY dt_measure DESC LIMIT 1")
        dataframe = pd.read_sql(ecg_query, engine_DB)
       # Extrair os pontos do ECG, a tag do modelo e a informação de anomalia
        pontos_ecg = dataframe.iloc[:, 4:].values.flatten()
        model_tag = dataframe.iloc[:, 3].iloc[0]
        is_anomalous = dataframe.iloc[:, 2].iloc[0]
        #raw_data = dataframe.values
        logger.info("Dados diários para inferência obtidos da base de dados para Frontend")
        return pontos_ecg, model_tag, is_anomalous

    except Exception as e:
        #print(f"Erro ao obter ECG's para Frontend: {str(e)}")
        logger.error(f"Erro ao obter ECG's para Frontend: {str(e)}")


def get_predictions():
    try:
        engine_DB = get_engine()
        ecg_query = text("SELECT * FROM Predictions ORDER BY dt_measure DESC LIMIT 1")
        dataframe = pd.read_sql(ecg_query, engine_DB)
        # Extrair pontos do ECG, tag do modelo e informação de anomalia
        pontos_predictions = dataframe.iloc[:, 3:].values.flatten()
        model_tag = dataframe.iloc[:, 2].iloc[0]
        #raw_data = dataframe.values
        logger.info("Dados diários para inferência obtidos da base de dados para Frontend")
        return pontos_predictions, model_tag

    except Exception as e:
        #print(f"Erro ao obter Predictions para Frontend: {str(e)}")
        logger.error(f"Erro ao obter Predictions para Frontend: {str(e)}")


pontos_ecg, model_tag, is_anomalous = get_ECG_inference_data()
pontos_predictions, model_tag_predictions = get_predictions()

#print(pontos_ecg)
#print(model_tag)
#print(is_anomalous)

#print("----------------Predictiosn----------------------------------")
#print(pontos_predictions)
#print(model_tag_predictions)



# Calcular o erro absoluto entre o ECG original e as previsões
erro = np.abs(pontos_ecg - pontos_predictions)

# Criar gráfico
fig, ax = plt.subplots()
ax.plot(pontos_ecg, label='ECG Original')
ax.plot(pontos_predictions, label='Previsões')
ax.plot(erro, label='Erro Absoluto', linestyle='--', color='red')

# Adicionar legendas e rótulos
ax.legend()
ax.set_xlabel('Ponto no ECG')
ax.set_ylabel('Valor')
ax.set_title('Comparação entre ECG Original e Previsões com Erro Absoluto')

# Mostrar gráfico
st.pyplot(fig)

# Adicionar um checkbox na interface do usuário
is_anomalo = st.checkbox("O ECG é Anômalo?")

# Lógica para enviar feedback para o backend
if is_anomalo:
    # Enviar feedback para o backend que o ECG é considerado anômalo
    # Aqui você deve adicionar a lógica específica para o seu aplicativo
    st.write("O ECG é considerado anômalo. Feedback enviado para o backend.")
else:
    # Enviar feedback para o backend que o ECG não é considerado anômalo
    # Aqui você deve adicionar a lógica específica para o seu aplicativo
    st.write("O ECG não é considerado anômalo. Feedback enviado para o backend.")
