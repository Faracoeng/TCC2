from sqlalchemy import Column, Integer, Float, DateTime, Boolean, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class InferenceECG(Base):
    __tablename__ = 'Inference_ECG'

    id = Column(Integer, primary_key=True, autoincrement=True)
    dt_measure = Column(DateTime, nullable=True)
    is_anomalous = Column(Boolean, default=False)
    model_tag = Column(String(80))  # Tag do modelo

    # Adicionando colunas para os números 0 até 139
    for i in range(140):
        locals()[str(i)] = Column(Float, default=None)

    def __init__(self, dt_measure, is_anomalous, model_tag, values):
        self.dt_measure = dt_measure
        self.is_anomalous = is_anomalous
        self.model_tag = model_tag
        for i, value in enumerate(values):
            setattr(self, str(i), value)

    def __repr__(self):
        return f'<InferenceECG(id={self.id}, dt_measure={self.dt_measure}, anomalous={self.anomalous}, tag={self.model_tag}, ...)>'
