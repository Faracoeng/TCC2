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
try:
    database_configs = {
        "host": os.environ.get('ORIGEM_MYSQL_HOST'),
        #"host": "localhost",    
        "port": os.environ.get('ORIGEM_MYSQL_PORT'),
        #"port": "3306",
        "charset": os.environ.get('ORIGEM_MYSQL_CHARSET'),
        #"charset": "utf8",
        "database": os.environ.get('ORIGEM_MYSQL_DATABASE'),
        #"database": "Results",
        "user": os.environ.get('ORIGEM_MYSQL_USER'),
        #"user": "datawarehouse_user",
        "password": os.environ.get('ORIGEM_MYSQL_PASSWORD')
        #"password": "admin123"
    }
    logger.info("Variáveis de ambiente database carregadas com sucesso")

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
def get_engine():
    try:
        engine = create_database_engine(database_configs)
        logger.info(f"Engine obtida com sucesso")
    except SQLAlchemyError as e:
        logger.error(f'Erro ao obter engine: {e}')
        return None
    return engine
engine = create_database_engine(database_configs)

# Criando as sessões
def get_session():
    try:
        Session_Orig = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info(f"Session obtida com sucesso")
    except SQLAlchemyError as e:
        logger.error(f'Erro ao obter Session_Orig: {e}')
        return None
    return Session_Orig()