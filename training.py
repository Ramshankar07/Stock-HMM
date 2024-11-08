"""Train hidden Markov model (HMM) on stock price data with Airflow integration"""

import json
import pickle
import os
from datetime import datetime
from typing import Tuple
import logging

import numpy as np
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from utils import compute_returns, make_stock_price_df

RANDOM_STATE = check_random_state(33)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(X: np.ndarray, max_n_state: int, n_train_init: int
    ) -> Tuple[StandardScaler, GaussianHMM]:
    """Fit scaler and HMM to training data."""
    logger.info('Model training started...')
    best_aic = None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    for n_state in range(1, max_n_state + 1):
        best_ll = None
        for _ in range(n_train_init):
            model = GaussianHMM(n_components=n_state, random_state=RANDOM_STATE)
            model.fit(X_scaled)
            ll = model.score(X_scaled)
            if not best_ll or best_ll < ll:
                best_ll = ll
                best_model = model
        aic = best_model.aic(X_scaled)
        logger.info(f'# of hidden states: {n_state}, AIC: {aic}')
        if not best_aic or aic < best_aic:
            best_aic = aic
            final_model = best_model
    logger.info('Model training completed')
    return scaler, final_model

def train_and_save_model():
    """Main function to train and save the model, called by Airflow"""
    try:
        logger.info('Starting model training process')
        
        # Load configuration
        with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)
        
        start_date = config['start_training_date']
        max_n_state = config['max_n_state']
        n_train_init = config['n_train_init']
        end_date = datetime.strftime(datetime.today(), '%Y-%m-%d')

        # Get and prepare data
        df = make_stock_price_df(start_date, end_date)
        df_ret = compute_returns(df)
        X = df_ret.values

        # Train model
        scaler, hmm = train_model(X, max_n_state, n_train_init)

        # Save training dates
        start_date = datetime.strftime(df.index[0], '%Y-%m-%d')
        end_date = datetime.strftime(df.index[-1], '%Y-%m-%d')
        
        os.makedirs('model', exist_ok=True)
        
        with open('model/training_dates.json', 'w') as file:
            json.dump({'start_date': start_date, 'end_date': end_date}, file)

        # Save model artifacts
        with open('model/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        with open('model/hmm.pkl', 'wb') as file:
            pickle.dump(hmm, file)
            
        logger.info('Model training and saving completed successfully')
        return True
        
    except Exception as e:
        logger.error(f'Error in model training: {str(e)}')
        raise

if __name__ == '__main__':
    train_and_save_model()