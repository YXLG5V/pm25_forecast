import numpy as np


class PM25Model:

    def __init__(self, artifacts):
        self.model = artifacts.model
        self.features = artifacts.features
        self.categories = artifacts.categories

    def predict(self, df):

        X = df[self.features]
        
        X = X.ffill().fillna(0)

        pred_log = self.model.predict(X)
        pred = np.maximum(0, np.expm1(pred_log))

        return pred