from sqlalchemy import Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Predictions(Base):
    __tablename__ = 'Predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    dt_measure = Column(DateTime, nullable=True)
    model_tag = Column(String(80))  # tag para identificar modelo utilizado na inferência

    # Adicione colunas para os valores no array (assumindo que o array tem 140 elementos)
    for i in range(140):
        locals()[f'value_{i}'] = Column(Float, default=None)

    def __init__(self, dt_measure, model_tag, values):
        self.dt_measure = dt_measure
        self.model_tag = model_tag
        for i, value in enumerate(values):
            setattr(self, f'value_{i}', value)
    def __repr__(self):
        return f'<Predictions(id={self.id}, timestamp={self.dt_measure}, tag={self.model_tag}, value_0={self.value_0}, value_1={self.value_1}, ...)>'
