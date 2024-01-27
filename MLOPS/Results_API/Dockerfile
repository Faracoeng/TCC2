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
RUN pip3 install -r requirements.txt

# Comando para iniciar a aplicação
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]

