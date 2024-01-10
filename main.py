from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import script
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## uvicorn api:app --reload --host 134.206.228.152 --port 8000


app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Classe pour les données entrantes pour la prédiction
class rating_obj(BaseModel):
    matrix: list  # La matrice doit être passée sous forme de liste de listes
    user_id : int


class POIData(BaseModel):
    Nom_du_POI: Optional[str]
    Categories_de_POI: Optional[str]
    Latitude: Optional[float]
    Longitude: Optional[float]
    Adresse_postale: Optional[str]
    Code_postal_et_commune: Optional[str]
    Periodes_regroupees: Optional[str]
    Covid19_mesures_specifiques: Optional[float]
    Covid19_est_en_activite: Optional[float]
    Covid19_periodes_d_ouvertures_confirmees: Optional[float]
    Createur_de_la_donnee: Optional[str]
    SIT_diffuseur: Optional[str]
    Date_de_mise_a_jour: Optional[str]
    Contacts_du_POI: Optional[str]
    Classements_du_POI: Optional[str]
    Description: Optional[str]
    URI_ID_du_POI: Optional[str]


"""class DataFrameInput(BaseModel):
    data: List[POIData]
# Initialisation et entraînement du modèle
try:
    model1 = script.model_notes()
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_clusters")
async def update_clusters(input_data: DataFrameInput):
    try:
        rows = [row.dict() for row in input_data.data]
        df = pd.DataFrame(rows)
        p=model1.predict_cluster(df)  # Ré-entraînement ou mise à jour des clusters
        return p.to_json(orient='records')  # Conversion DataFrame en JSON
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))"""

@app.post("/predict_ratings")
async def predict_ratings(data: rating_obj):
    try:
        logger.info("Début de la prédiction des notes")
        
        # Conversion de JSON en DataFrame
        model2 = script.model_notes()
        user_article_df = np.array(data.matrix)
        predictions = model2.fit_predict_ratings(user_article_df,data.user_id)
        logger.info("Prédiction terminée")
        # Convertir les prédictions en format approprié pour la réponse JSON
        predictions_json = []
        for pred in predictions:
            predictions_json.append({'uid': pred.uid, 'iid': pred.iid, 'rating': pred.est})
            predictions_json = sorted(predictions_json, key=lambda pred: pred["rating"], reverse=True)
        logger.info("Réponse JSON préparée")

        return predictions_json

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))



