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
        ecg_query = text("SELECT * FROM Inference_ECG ORDER BY dt_measure DESC LIMIT 1")
        dataframe = pd.read_sql(ecg_query, engine_Results)
       # Extrair os pontos do ECG, a tag do modelo e a informação de anomalia
        pontos_ecg = dataframe.iloc[:, 4:].values.flatten()
        model_tag = dataframe.iloc[:, 3].iloc[0]
        is_anomalous = dataframe.iloc[:, 2].iloc[0]
        #raw_data = dataframe.values
        logger.info("Dados diários para inferência obtidos da base de dados para Frontend")
        return pontos_ecg, model_tag, is_anomalous, dataframe

    except Exception as e:
        #print(f"Erro ao obter ECG's para Frontend: {str(e)}")
        logger.error(f"Erro ao obter ECG's para Frontend: {str(e)}")


def get_predictions():
    try:
        ecg_query = text("SELECT * FROM Predictions ORDER BY dt_measure DESC LIMIT 1")
        dataframe = pd.read_sql(ecg_query, engine_Results)
        # Extrair pontos do ECG, tag do modelo e informação de anomalia
        pontos_predictions = dataframe.iloc[:, 3:].values.flatten()
        model_tag = dataframe.iloc[:, 2].iloc[0]
        #raw_data = dataframe.values
        logger.info("Dados diários para inferência obtidos da base de dados para Frontend")
        return pontos_predictions, model_tag

    except Exception as e:
        #print(f"Erro ao obter Predictions para Frontend: {str(e)}")
        logger.error(f"Erro ao obter Predictions para Frontend: {str(e)}")


pontos_ecg, model_tag, is_anomalous, original_ecg_dataframe = get_ECG_inference_data()
pontos_predictions, model_tag_predictions = get_predictions()


def plot_and_update_data():
    # Obter dados do ECG e previsões
    pontos_ecg, model_tag, is_anomalous, original_ecg_dataframe = get_ECG_inference_data()
    pontos_predictions, model_tag_predictions = get_predictions()

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

    # Adicionar um botão "OK"
    if st.button("OK"):
        # Lógica para enviar feedback para o backend com base no valor do checkbox
        # Para atualizar se ECG é anôlamo ou não
        if is_anomalo:
                original_ecg_dataframe.iloc[0, 2] = 1  
        else:
            original_ecg_dataframe.iloc[0, 2] = 0  
        # Remover a primeira coluna (coluna 'id')
        original_ecg_dataframe = original_ecg_dataframe.drop(columns=[0])


        # Reorganizar as colunas para corresponder à estrutura da tabela ECG
        original_ecg_dataframe = original_ecg_dataframe[[col for col in range(1, 141)] + ['dt_measure']]
        # Atualizar o banco de dados ou fazer qualquer ação necessária com o dataframe atualizado
        # (substitua esta parte pelo código específico do seu aplicativo)
        st.write("Feedback enviado para o backend. Conjunto de dados atualizado e inserido na base de treinamento:")
        st.write(original_ecg_dataframe)


plot_and_update_data()