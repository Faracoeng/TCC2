from fastapi import FastAPI

from database import *
from models.ecg import *
from models.predictions import *
from schemas.ecg import *
from schemas.predictions import *
from routes.results_route import router


# Carregar a configuração do logger a partir do arquivo logging.conf
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('fastapi')


app = FastAPI()

@app.on_event("startup")
def startup():
    logger.info("FastAPI iniciada com sucesso!")
    print("FastAPI iniciada com sucesso!")
    try:
        InferenceECG.metadata.create_all(bind=engine)
        Predictions.metadata.create_all(bind=engine)
    except Exception as e:
        print("Erro ao criar as tabelas do banco de dados:", e)
        logger.error("Erro ao criar as tabelas do banco de dados: %s", str(e))

@app.on_event("shutdown")
def shutdown():
    logger.info("Encerrando a aplicação e fechando a sessão do banco de dados.")
    #Session_Orig.close()  # Close the database session.
    

app.include_router(router)

#if __name__ == "__main__":
    #uvicorn.run("fastapi_code:app")
#    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)