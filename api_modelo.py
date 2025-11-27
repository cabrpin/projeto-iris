import os
import logging
import datetime
from venv import logger
import jwt
from functools import wraps # Import wraps for decorator functionality
from flask import Flask, request, jsonify
import joblib
import numpy as np
from sqlalchemy import Column, Integer, Float, DateTime, String, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker    

JWT_SECRET_KEY = "MEUSEGREDOAQUI"
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600

   
DB_URL = 'sqlite:///predictions.db'
engine = create_engine(DB_URL, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predict_class = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

#Cria as tabelas do banco em produção usar o Alembic
Base.metadata.create_all(engine)

logging.basicConfig(level=logging.INFO)
logging.getLogger('api_modelo')


# Carrega o modelo treinado
model = joblib.load('modelo_iris.pkl')
logger.info("Modelo carregado com sucesso.")


app = Flask(__name__)
predictions_cache = {}

#autenticacao simples jwt
TEST_USERNAME = "admin"
TEST_PASSWORD = "secret"

@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "Iris prediction API", "status": "ok"})

def create_token(username):
    payload ={
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        #Pega o token do header Authorization: Bearer <token>
        #decodificar e checar expiração
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = data.get('username')
    password = data.get('password')

    if username == TEST_USERNAME and password == TEST_PASSWORD:
        token = create_token(username)
        return jsonify({"token": token})
    else:
        return jsonify({"message": "Credenciais Invalidas"}), 401
    
@app.route("/predict", methods=["POST"])
@token_required
def predict():
    """
    Endpoint protegido por token para obter predição.
    Corpo json:
    {
        sepal_length: 5.1,
        sepal_width: 3.5,
        petal_length: 1.4,
        petal_width: 0.2  
    }
   """
    data = request.get_json(force=True)
    try:
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])
    except(ValueError, KeyError) as e:
        logger.error(f"Dados de entrada invalidos: {e}") 
        return jsonify({"error": "Dados invalidos, verifique os parametros"}), 400

    #verificar se ja existe no cache
    features = (sepal_length, sepal_width, petal_length, petal_width)
    if features in predictions_cache:
        logger.info("Cache hit para %s.", features)
        predicted_class = predictions_cache[features]
    else:
    #Rodar o modelo
        input_data = np.array([features])
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])    
    #armazenar no cache 
        predictions_cache[features] = predicted_class
        logger.info("Cache miss para %s. Predicao realizada.", features)

    # Salvar no banco de dados
    db = SessionLocal()
    new_prediction = Prediction(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
        predict_class=predicted_class
    )
    db.add(new_prediction)
    db.commit()
    db.close()
    logger.info("Predicao salva no banco de dados.") 
    return jsonify({ "prediction": predicted_class})


@app.route("/predictions", methods=["GET"])
@token_required
def get_predictions():
    """
    Lista as predições armazenadas no banco de dados.
    Parametros opcionais via query string:
    - limit (int): quantos registros retornar padrao 10.
    - offset (int): a partir de qual registros começar, padrão 0.
    Exemplo:
     /predictions?limit=5&offset=10
    """
    limit = int(request.args.get("limit", 10))
    offset = int(request.args.get("offset", 0))
    db = SessionLocal()
    preds = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset).all()
    db.close()
    results = []
    for p in preds:
        results.append({
            "id": p.id,
            "sepal_length": p.sepal_length,
            "sepal_width": p.sepal_width,
            "petal_length": p.petal_length,
            "petal_width": p.petal_width,
            "predict_class": p.predict_class,
            "created_at": p.created_at.isoformat()
        })  
    return jsonify(results)
    
if __name__ == '__main__':
    app.run(debug=True) 