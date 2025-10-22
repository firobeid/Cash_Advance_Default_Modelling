import xgboost as xgb    
from sklearn.base import BaseEstimator, ClassifierMixin    
from typing import Tuple  
from sklearn.metrics import roc_curve   
import numpy as np
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def ks_stat(y_pred, dtrain)-> Tuple[str, float]:
    y_true = dtrain.get_label()  
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)  
    ks_stat = max(tpr - fpr)  
    return 'ks_stat', ks_stat 


class XGBoostClassifierWithEarlyStoppingNoCV(BaseEstimator, ClassifierMixin):    
    def __init__(self, nfolds=5, early_stopping_rounds=50, **params):    
        self.nfolds = nfolds
        self.early_stopping_rounds = early_stopping_rounds
        for param, value in params.items():
            setattr(self, param, value)
        
        self.evals_result = {}    
        self.bst = None
        self.best_iteration = None
        self.best_score = None
        
    def get_params(self, deep=True):
        excluded_attrs = {'evals_result', 'bst', 'best_iteration', 'best_score'}
        params = {key: value for key, value in self.__dict__.items()
                 if not key.startswith('_') and key not in excluded_attrs}
        return params
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def _get_xgb_params(self):
        param_mapping = {
            'learning_rate': 'eta',
            'reg_lambda': 'lambda',
            'reg_alpha': 'alpha',
            'random_state': 'seed'
        }
        
        params = self.get_params()
        xgb_params = {}
        
        params.pop('nfolds', None)
        params.pop('early_stopping_rounds', None)
        
        for param, value in params.items():
            xgb_param = param_mapping.get(param, param)
            xgb_params[xgb_param] = value
            
        return xgb_params
    
    def fit(self, X, y):    
        dtrain = xgb.DMatrix(X, label=y)
        params = self._get_xgb_params()
        try:
            n_estimators = getattr(self, 'n_estimators', params['n_estimators'])
            params.pop('n_estimators', None)
        except:
            n_estimators = getattr(self, 'n_estimators', 1000)
        
        # Train final model directly without CV
        self.bst = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            feval=ks_stat,
            verbose_eval=True, maximize=True
        )
        
        return self
    
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        predictions = self.bst.predict(dtest)
        return np.vstack((1 - predictions, predictions)).T
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
