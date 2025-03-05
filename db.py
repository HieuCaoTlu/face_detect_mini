from sqlalchemy import create_engine, Column, Integer, String, PickleType
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class EmbeddingData(Base):
    __tablename__ = 'embedding_data'

    id = Column(Integer, primary_key=True)
    label = Column(String, nullable=False)
    embedding = Column(PickleType, nullable=False) 

engine = create_engine('sqlite:///embeddings.db')
Base.metadata.create_all(engine)