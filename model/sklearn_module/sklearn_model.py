import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from ..base_model import BaseModel


class MLmodel(BaseModel):
    def __init__(self):
        super().__init__()

    def save_model(self, path):
        Xcols = self.Xcols
        Ycols = self.Ycols
        params = self.params
        model = self.model
        scaler = self.scaler
        saved_dict = {"params": params, "model": model, "scaler": scaler, "Xcols": Xcols, "Ycols": Ycols}
        joblib.dump(saved_dict, path)

    @classmethod
    def load_model(cls, path):
        saved_dict = joblib.load(path)
        params = saved_dict["params"]
        model = saved_dict["model"]
        scaler = saved_dict["scaler"]
        inst = cls(params, model, scaler)
        inst.set_cols(saved_dict["Xcols"], saved_dict["Ycols"])
        inst.is_fitted = True
        return inst

    def set_cols(self, Xcols, Ycols):
        self.Xcols = Xcols
        self.Ycols = Ycols


class LgbRegressor(MLmodel):
    def __init__(self, params, model=None, scaler=None):
        super().__init__()
        self.params = params

        if model is None:
            self.model = lgb.LGBMRegressor(**params)
        else:
            self.model = model

        if scaler is None:
            self.scaler = RobustScaler()
        else:
            self.scaler = scaler
            
        self.is_fitted = False
        
        
    def fit(self, train_X, train_Y, sample_weight=None):
        if self.scaler is None or self.model is None:
            raise Exception("Scaler or model is not set")
        self.scaler = self.scaler.fit(train_X)
        train_X_copy = self.scaler.transform(train_X.copy())
        self.model.fit(train_X_copy, train_Y, sample_weight = sample_weight)
        self.is_fitted = True

    def predict(self, test_X):
        if self.scaler is None:
            raise Exception("Scaler is not set")
        if not self.is_fitted:
            raise Exception("Model is not fitted")
        test_X_copy = self.scaler.transform(test_X.copy())
        res = self.model.predict(test_X_copy)
        return res

    def plot_feature_importance(self):
        if self.model is None:
            raise Exception("Model is not set")
        feature_importance = self.model.feature_importance_
        feature_imp = pd.DataFrame(sorted(zip(feature_importance, self.Xcols)), columns=['Value', 'Feature'])
        plt.figure(figsize=(5, 0.2 * len(feature_imp)))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.tight_layout()
        plt.show()


class SklearnRegressor(MLmodel):
    def __init__(self, params, model=None, scaler=None):
        self.params = params
        if model is None:
            self.model = LassoCV(**params)
        elif isinstance(model, str):
            if model == "Lasso":
                self.model = LassoCV(**params)
            elif model == "Ridge":
                self.model = RidgeCV(**params)
            elif model == "ElasticNet":
                self.model = ElasticNetCV(**params)
            else:
                raise Exception(f"{model} is not implemented")
        else:
            self.model = model
            
        if scaler is None:
            self.scaler = RobustScaler()
        else:
            self.scaler = scaler
            
        self.is_fitted = False

    def fit(self, train_X, train_Y):
        if self.scaler is None or self.model is None:
            raise Exception("Scaler or model is not set")
        self.scaler = self.scaler.fit(train_X)
        train_X_copy = self.scaler.transform(train_X.copy())
        self.model.fit(train_X_copy, train_Y)
        self.is_fitted = True
        
    def predict(self, test_X):
        if self.scaler is None:
            raise Exception("Scaler is not set")
        if not self.is_fitted:
            raise Exception("Model is not fitted")
        test_X_copy = self.scaler.transform(test_X.copy())
        res = self.model.predict(test_X_copy)
        return res

    def plot_feature_importance(self):
        if self.model is None:
            raise Exception("Model is not set")
        imp_map = pd.DataFrame({"feature": list(self.Xcols), "coef": list(self.model.coef_)})
        imp_map = imp_map.sort_values(by="coef", key=np.abs, ascending=False)
        plt.figure(figsize=(5, 0.2 * len(imp_map)))
        sns.barplot(x="coef", y="feature", data=imp_map)
        plt.tight_layout()
        plt.show()
