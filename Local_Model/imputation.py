import gc
from sklearn.base import TransformerMixin  

class NullImputa(TransformerMixin):   
    '''
    __Author__ = 'Firas Obeid'
    '''
    def __init__(self, min_count_na = 5, missing_indicator = True):      
        self.min_count_na = min_count_na   
        self.missing_indicator = missing_indicator
        self.missing_cols = None    
        self.additional_features = []  # Store names of additional features  
        self.column_names = None  # Store column names after transformation 
        
    def fit(self, X, y=None):      
        self.missing_cols = X.columns[X.isnull().any()].tolist()      
        return self      
        
    def transform(self, X, y=None):      
        X = X.copy()  # create a copy of the input DataFrame      
        for col in X.columns.tolist():      
            if col in X.columns[X.dtypes == object].tolist():      
                X[col] = X[col].fillna(X[col].mode()[0])      
            else:      
                if (col in self.missing_cols) & (self.missing_indicator):      
                    new_col_name = f'{col}-mi'  
                    X[new_col_name] = X[col].isna().astype(int)     
                    self.additional_features.append(new_col_name)  # Store the new column name  
                if X[col].isnull().sum() <= self.min_count_na:      
                   X[col] = X[col].fillna(X[col].median())      
                else:      
                    X[col] = X[col].fillna(-9999)      
        assert X.isna().sum().sum() == 0      
        _ = gc.collect()      
        print("Imputation complete.....") 
        self.column_names = X.columns.tolist()  # Store column names after transformation  
        return X  