from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.ecg import InferenceECG
from models.predictions import Predictions
from schemas.ecg import InferenceECGCreateSchema
from schemas.predictions import PredictionsCreateSchema
import logging.config
from database import get_session

# Carregar a configuração do logger a partir do arquivo logging.conf
#logging.config.fileConfig('logging.conf')
logging.config.fileConfig('/app/logging.conf')
logger = logging.getLogger('fastapi')

router = APIRouter()

@router.post("/predictions", response_model=PredictionsCreateSchema)
def create_prediction(prediction_data: PredictionsCreateSchema, db: Session = Depends(get_session)):
    try:
        prediction = Predictions(**prediction_data.dict())
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        logger.info(f"Predição criada com sucesso: {prediction}")
        #return prediction
    except Exception as e:
        import traceback
        traceback.print_exc()  # Imprime o traceback completo no console
        logger.error(f"Erro ao criar a Predição: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao criar a Predição")


@router.post("/ecg", response_model=InferenceECGCreateSchema)  
def create_inference_ecg(inference_data: InferenceECGCreateSchema, db: Session = Depends(get_session)):
    try:
        inference_ecg = InferenceECG(**inference_data.dict())
        db.add(inference_ecg)
        db.commit()
        db.refresh(inference_ecg)
        logger.info("Inferência ECG criada com sucesso")
        #return inference_ecg
    except Exception as e:
        import traceback
        traceback.print_exc()  # Imprime o traceback completo no console
        logger.error(f"Erro ao criar a Inferência ECG: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao criar a Inferência ECG")