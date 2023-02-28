FROM python:3.9.0

RUN pip install \
    bokeh==3.0.3\
    numpy \
    boto3 \
    pandas \
    scikit-learn \
    Pillow&& \
    mkdir /mlflow/

EXPOSE 5000

CMD bokeh serve \
    --show dashboard_ui.py \
    --port 8080 \