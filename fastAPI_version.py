from model import ModelPipeline
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import os
from fastapi.responses import HTMLResponse
import uvicorn

REG_MODEL = joblib.load(f'{os.getcwd()}/RandomForest_regression.joblib')
CLASS_MODEL = joblib.load(f'{os.getcwd()}/rnd_clf_THIS_MODEL.joblib')


app = FastAPI()
templates = Jinja2Templates(directory='pages')
app.mount('/static', StaticFiles(directory='static'), name='static')

class Item(BaseModel):
    HB: float
    Ultimate_strength: float
    E: float
    ro: float
    c: float


@app.post('/predict')
async def predict_materia(items: list[Item]):
    mdl = ModelPipeline(REG_MODEL, CLASS_MODEL)
    list_objs = []
    for item in items:
        obj = {
            'HB': item.HB,
            'Ultimate_strength': item.Ultimate_strength,
            'E': item.E,
            'ro': item.ro,
            'c': item.c
        }
        list_objs.append(obj)
    return mdl(list_objs)

@app.post('/', response_class=HTMLResponse)
async def predict_materia_home(request: Request, HB=Form(), ultimate_strength=Form(), E=Form(), ro=Form(), c=Form()):
    mdl = ModelPipeline(REG_MODEL, CLASS_MODEL)
    list_obj = [
        {
            'HB': HB,
            'Ultimate_strength': ultimate_strength,
            'E': E,
            'ro': ro,
            'c': c
        }
    ]
    y = mdl(list_obj)[0]

    output = ''
    for k in y.keys():
        output += k + f'<sub>{y[k]}</sub>'

    return templates.TemplateResponse('index.html', context={'request': request, 'output': output})

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/about', response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse('about.html', {'request': request})

@app.get('/api', response_class=HTMLResponse)
async def api(request: Request):
    return templates.TemplateResponse('api.html', {'request': request})


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)