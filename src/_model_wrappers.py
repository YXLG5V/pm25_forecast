class NNWrapper:
    def __init__(self, model, imputer, scaler):
        self.model = model
        self.imputer = imputer
        self.scaler = scaler

    def predict(self, X):
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return self.model.predict(X_scaled, verbose=0).flatten()