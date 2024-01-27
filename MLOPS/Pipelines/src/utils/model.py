
from sqlalchemy import Column, Integer, String, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Model(Base):
    __tablename__ = 'Model'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    tag = Column(String(80), primary_key=True)
    max_value = Column(Float)
    min_value = Column(Float)
    threshold = Column(Float)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    model_weights = Column(LargeBinary)