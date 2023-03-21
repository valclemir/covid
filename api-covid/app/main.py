from fastapi import FastAPI
from pydantic import BaseModel
from pycaret import classification
import numpy as np 
import pandas as pd 

app = FastAPI() 


class ParamsModelUTI(BaseModel):
    primeiro_atendimento: int
    dias_internacao: int 
    fat_imp_sintomas: int
    sexo: int
    fx_etar: int
    cd_cor: int
    cd_eciv: int 


class ParamsModelObito(BaseModel):
    primeiro_atendimento: int
    dias_internacao: int 
    fat_imp_sintomas: int
    sexo: int
    fx_etar: int
    cd_cor: int
    cd_eciv: int 
    usou_uti: int     


def load_model(model_name):

    model = classification.load_model(model_name="model/" + model_name) 
    return model 


@app.post("/uti/")    
async def covid(item: ParamsModelUTI):

    model = load_model("model-covid-uti")
    
    X = np.array([[
                item.primeiro_atendimento,
                item.dias_internacao,
                item.fat_imp_sintomas,
                item.sexo,
                item.fx_etar,
                item.cd_cor,
                item.cd_eciv
    ]])


    columns = ['PRI_ATEND', 
               'DIAS_INTERNACAO', 
               'FAT_IMP_SINNTOMAS', 
               'SEXO',
               'FX_ETAR', 
               'CD_COR', 
               'CD_ECIV']

    

    df = pd.DataFrame(X, columns=columns)

    y_pred = np.where(model.predict(df) == 1, "sim", "não")

    results = {"usa uti": str(y_pred[0])}
    return results




@app.post("/obitos/")    
async def covid(item: ParamsModelObito):

    model = load_model("model-covid-obito")
    
    X = np.array([[
                item.primeiro_atendimento,
                item.dias_internacao,
                item.fat_imp_sintomas,
                item.sexo,
                item.fx_etar,
                item.cd_cor,
                item.cd_eciv,
                item.usou_uti
    ]])


    columns = ['PRI_ATEND', 
               'DIAS_INTER', 
               'FAT_IMP_SIN', 
               'SEXO',
               'FX_ETAR', 
               'CD_COR', 
               'CD_ECIV', 
               'USOU_UTI'
               ]

    

    df = pd.DataFrame(X, columns=columns)

    print(df)

    y_pred = np.where(model.predict(df) == 1, "sim", "não")

    results = {"obito": str(y_pred[0])}
    return results