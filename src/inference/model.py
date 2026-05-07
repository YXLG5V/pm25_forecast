import numpy as np

USE_LOG_TARGET = False


def inverse_target(y):
    return np.maximum(0, np.expm1(y)) if USE_LOG_TARGET else y


class PM25Model:

    def __init__(self, artifacts):
        self.model = artifacts.model
        self.features = artifacts.features
        self.categories = artifacts.categories

    def predict(self, df):

        X = df[self.features]

        # ensemble
        if isinstance(self.model, dict):

            preds = []

            for m in self.model.values():

                pred_raw = m.predict(X)
                pred = inverse_target(pred_raw)

                preds.append(pred)

            return np.mean(preds, axis=0)

        # single model
        pred_raw = self.model.predict(X)

        return inverse_target(pred_raw)