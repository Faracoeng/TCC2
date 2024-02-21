from datetime import datetime
from pydantic import BaseModel, ValidationError, validator
from typing import List

class InferenceECGCreateSchema(BaseModel):
    values: List[float]
    dt_measure: datetime
    is_anomalous: int
    model_tag: str

    #@validator("values")
    #def check_values_length(cls, values):
    #    if len(values) != 140:
    #        raise ValueError("A lista 'values' deve ter exatamente 140 itens.")
    #    return values
