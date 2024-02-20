# Imagem base
FROM python:3.9

# Configuração do ambiente
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Diretório de trabalho no container
WORKDIR /app

# Copiar os arquivos de código-fonte para o container
COPY . /app

# Instalar as dependências
RUN apt update && apt install -y vim
RUN pip3 install -r src/requirements.txt

# Define o comando de inicialização do Streamlit com CORS desabilitado e mantém o contêiner em execução
CMD ["sleep", "infinity"]

