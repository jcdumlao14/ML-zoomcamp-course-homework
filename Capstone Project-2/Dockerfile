FROM python:3.9.12-slim
RUN pip install pipenv

WORKDIR / app

COPY ["Pipfile","Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py","model_rf_t=04.bin","./"]

EXPOSE 8888

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:8888", "predict:app"]