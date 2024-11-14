"""
Airflow DAG for training stock price prediction model
Place this file in your Airflow DAGs folder
"""

from datetime import datetime, timedelta
import os
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.http_sensor import HttpSensor
from airflow.models import Variable

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_streamlit_running():
    """Check if Streamlit app is running by making a request to its health endpoint"""
    try:
        response = requests.get('http://localhost:8501/healthz')
        return response.status_code == 200
    except:
        return False

def train_model(**context):
    """Execute the model training script"""
    import sys
    sys.path.append('Stock-HMM')  # Replace with your project path
    
    from training import train_and_save_model
    train_and_save_model()
    
    # Updating last training timestamp
    Variable.set('last_model_training', datetime.now().isoformat())

dag = DAG(
    'stock_price_model_training',
    default_args=default_args,
    description='Train stock price prediction model when Streamlit is running',
    schedule_interval=timedelta(days=1),  
    catchup=False
)

# Sensor to check if Streamlit is running
streamlit_sensor = HttpSensor(
    task_id='check_streamlit_running',
    http_conn_id='streamlit_connection', 
    endpoint='/healthz',
    poke_interval=300,  # Check every 5 minutes
    timeout=60 * 60 * 2,  # Timeout after 2 hours
    mode='reschedule',  # Release worker slot while waiting
    dag=dag
)

# Model training task
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

streamlit_sensor >> train_model_task