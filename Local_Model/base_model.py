import xgboost as xgb    
from sklearn.base import BaseEstimator, ClassifierMixin    
from typing import Tuple  
from sklearn.metrics import roc_curve   

def ks_stat(y_pred, dtrain)-> Tuple[str, float]:
    y_true = dtrain.get_label()  
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)  
    ks_stat = max(tpr - fpr)  
    return 'ks_stat', ks_stat 
    
class XGBoostClassifierWithEarlyStopping(BaseEstimator, ClassifierMixin):    
    def __init__(self, nfolds=5, early_stopping_rounds=50, max_rounds=1000,**params):    
        self.params = params    
        self.evals_result = {}    
        self.bst = None  
        self.nfolds = nfolds
        self.early_stopping_rounds = early_stopping_rounds
        self.cvresult = None
        self.max_rounds = max_rounds
    
    def fit(self, X, y, **fit_params):    
        dtrain = xgb.DMatrix(X, label=y)  
        self.cvresult = xgb.cv(self.params, dtrain, 
                               num_boost_round=self.max_rounds,  
                               verbose_eval=True, maximize=True,
                               nfold=self.nfolds, metrics=['auc'], custom_metric = ks_stat,
                               early_stopping_rounds=self.early_stopping_rounds, stratified=True,  
                               seed=42)  
        self.bst = xgb.train(self.params, 
                             dtrain, 
                             num_boost_round=self.cvresult.shape[0], 
                             feval = ks_stat)  
        return self    
    
    def predict(self, X):    
        dtest = xgb.DMatrix(X)    
        return self.bst.predict(dtest) 