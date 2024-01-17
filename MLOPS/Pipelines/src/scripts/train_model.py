
def train_model(autoencoder, ecg, train_data, test_data):
    """
    Função para treinar o modelo autoencoder
    :param ecg: Objeto da classe ECG
    :param train_data: Dados de treino
    :param test_data: Dados de teste
    :return: None
    """
  
    # Trazer as variaveis de ambiente
    autoencoder.compile(optimizer='adam', loss='mae')
    autoencoder.fit(train_data, train_data, epochs=20, batch_size=512, validation_data=(test_data, test_data), shuffle=True)
    
    # Salvar o modelo treinado
    autoencoder.save('../models/autoencoder_model'+'/'+ecg.begin_date+'_'+ecg.end_date)



