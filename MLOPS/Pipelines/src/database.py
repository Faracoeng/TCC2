import logging.config
import sys
import os
#SQL
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

#logging.config.fileConfig('logging.conf')
# No docker
logging.config.fileConfig('/app/src/logging.conf')
logger = logging.getLogger()
    
#  Carregando variáveis de ambiente versão Dockerizada

# try:
#     db_origem = {
#         "host":  "localhost",
#         "port": 3306,
#         "charset": "utf8",
#         "database": "Datawarehouse",
#         "user": "datawarehouse_user",
#         "password": "admin123"
#     }

#     db_destino = {
#         "host": "localhost",
#         "port": 3306,
#         "charset": "utf8",
#         "database": "Results",
#         "user": "datawarehouse_user",
#         "password": "admin123"
#     }


try:
    db_origem = {
        "host": os.environ.get('ORIGEM_MYSQL_HOST'),
        "port": os.environ.get('ORIGEM_MYSQL_PORT'),
        "charset": os.environ.get('ORIGEM_MYSQL_CHARSET'),
        "database": os.environ.get('ORIGEM_MYSQL_DATABASE'),
        "user": os.environ.get('ORIGEM_MYSQL_USER'),
        "password": os.environ.get('ORIGEM_MYSQL_PASSWORD')
    }

    db_destino = {
        "host": os.environ.get('DESTINO_MYSQL_HOST'),
        "port": os.environ.get('DESTINO_MYSQL_PORT'),
        "charset": os.environ.get('DESTINO_MYSQL_CHARSET'),
        "database": os.environ.get('DESTINO_MYSQL_DATABASE'),
        "user": os.environ.get('DESTINO_MYSQL_USER'),
        "password": os.environ.get('DESTINO_MYSQL_PASSWORD')
    }

    logger.info("Variáveis de ambiente carregadas para origem e destino")
except Exception as e:
    logger.error(f"Erro ao obter variáveis de ambiente database: {str(e)}")

def create_database_engine(config):
    try:
        db_url = (
            f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?charset={config['charset']}"
        )
        engine = create_engine(db_url, connect_args={"charset": "utf8mb4"})
        logger.info(f"Conectado à base de dados em {config['host']}:{config['port']}")
        return engine
    except SQLAlchemyError as e:
        logger.error(f'Erro ao conectar ao banco de dados: {e}')
        return None


# Criando engines para bancos de dados de origem e destino
engine_Pipelines = create_database_engine(db_origem)
engine_Results = create_database_engine(db_destino)

# Criando as sessões
def get_session_Pipelines():
    Session_Orig = sessionmaker(autocommit=False, autoflush=False, bind=engine_Pipelines)
    
    return Session_Orig()

def get_session_Results():
    Session_Dest = sessionmaker(autocommit=False, autoflush=False, bind=engine_Results)
    
    return Session_Dest()




