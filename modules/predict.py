import os
import dill
import json
import pandas as pd
import logging

from os import listdir
from datetime import datetime

# Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - текущая директория при локальном запуске
path = os.environ.get('PROJECT_PATH', '..')


def predict():
    # загрузим обученную модель
    pkl_files = listdir(f'{path}/data/models')
    logging.info(f'Model is loaded as {pkl_files[-1]}')
    with open(f'{path}/data/models/{pkl_files[-1]}', 'rb') as file:
        model = dill.load(file)

    data = pd.DataFrame()
    # список файлов в директории data/test
    tests = listdir(f'{path}/data/test')
    for test in tests:
        with open(f'{path}/data/test/{test}') as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            # cделаем предсказания для датафрейма
            y = model.predict(df)
            df['predict'] = y[0]
            # объединим предсказания в один датафрейм
            data = pd.concat([data, df[['id', 'price', 'predict']]], ignore_index=True)

    # сохраним предсказания в csv-формат
    preds_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    data.to_csv(preds_filename)
    logging.info(f'Model is saved as {preds_filename}')


if __name__ == '__main__':
    predict()
