import joblib


class ModelArtifacts:

    def __init__(
        self,
        model_path,
        features_path,
        categories_path
    ):
        self.model = joblib.load(model_path)
        self.features = joblib.load(features_path)
        self.categories = joblib.load(categories_path)

    def info(self):
        return {
            "n_features": len(self.features),
            "n_locations": len(self.categories)
        }