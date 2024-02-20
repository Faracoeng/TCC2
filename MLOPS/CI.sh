#!/bin/bash

# Verifique se foi fornecida uma tag como argumento
if [ -z "$1" ]; then
    echo "Por favor, forneça uma tag como argumento."
    exit 1
fi

# Verifique se foi fornecido um nome de usuário como argumento
if [ -z "$2" ]; then
    echo "Por favor, forneça um nome de usuário como argumento."
    exit 1
fi

# Verifique se foi fornecida uma senha como argumento
if [ -z "$3" ]; then
    echo "Por favor, forneça uma senha como argumento."
    exit 1
fi
# Defina a tag do Git
TAG=$1
NEXUS_USER=$2
NEXUS_PASS=$3
# Variaveis
NEXUS_HOST="localhost"


# Gere uma nova tag no Git
git tag -a $TAG -m "Versão $TAG"
#git push origin $tag

# Construindo a imagem Docker do pipelines
docker build -t $NEXUS_HOST:8882/pipelines:$TAG -f Pipelines/Dockerfile ./Pipelines
echo "docker push $NEXUS_HOST:8082/pipelines:$TAG"

# Construindo a imagem Docker da API de resultados
docker build -t $NEXUS_HOST:8082/results_api:$TAG -f Results_API/Dockerfile ./Results_API
docker push $NEXUS_HOST:8082/results_api:$TAG

# Gerando tar.gz do projeto
tar -czvf Pipelines_$TAG.tar.gz Pipelines
tar -czvf Results_API_$TAG.tar.gz Results_API

# Subindo tar.gz do projeto para o Nexus
curl -v -u $NEXUS_USER:$NEXUS_PASS --upload-file Pipelines_$TAG.tar.gz http://$NEXUS_HOST:8081/repository/ECG/Pipelines/Pipelines_$TAG.tar.gz
curl -v -u $NEXUS_USER:$NEXUS_PASS --upload-file Results_API_$TAG.tar.gz http://$NEXUS_HOST:8081/repository/ECG/Results_API/Results_API_$TAG.tar.gz
                                                                               
echo "Script concluído com sucesso!"
