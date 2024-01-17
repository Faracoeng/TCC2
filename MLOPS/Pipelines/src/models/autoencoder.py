
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
# Toda a serie temporal de cada ECG, deve ter um número padronizados 
# de pontos, para que possa ser utilizado na rede neural, aqui serão utilizado 140 pontos no eixo X

#Ja no eixo Y, será normalizado entre 0 e 1 para representar a aplitude do sinal
# Então na fonte dos dados, será necessário fazer um tratamento para que todos os 
# sinais tenham 140 pontos e estejam normalizados entre 0 e 1, a partir disso treina a rede neural.


# Para este dimensionamento, de 140 pontos de entrada, passando pela rede neural
# Deve reduzir a dimensionalidade para 32, depois para 16 e depois para 8.
# Representando uma redução de 140 para 8bytes, ou seja, 17 vezes menor.
# sendo necessário 3 camadas para isso.

# Depois descompactar, ou seja, aumentar a dimensionalidade de 8 para 16, depois para 32 e depois para 140.
# Estabelecendo um modelo que saiba fazer esse processo para ECG's que saibamos que não são anomalos, 
# aprendemos a criar um modelo do que seria um ECG normal, e quando passarmos um ECG anomalo, o erro no processo de compressão e descompressão será maior,
# E baseados em um limear de erro, podemos identificar se o ECG é normal ou anomalo.

# Essa é a classe que representa a rede neural que compacta e  descompacta os dados
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

