from fastapi import FastAPI
import torch
import pandas as pd
from app.models.financialData import FinancialData
from app.utils.preprocessing import preprocessing
from app.utils.pred_score import pred_score

app = FastAPI()

@app.post()
def predict(data: FinancialData):
    dataTensor = preprocessing(data)
    return pred_score(dataTensor)