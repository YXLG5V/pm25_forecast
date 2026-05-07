import joblib
import sys
import src._model_wrappers as model_wrappers

sys.modules["_model_wrappers"] = model_wrappers

class ModelArtifacts:

    def __init__(
        self,
        model_path,
        features_path,
        location_map
    ):
        self.model = joblib.load(model_path)
        self.features = joblib.load(features_path)
        self.categories = joblib.load(location_map)

    def info(self):
        return {
            "n_features": len(self.features),
            "n_locations": len(self.categories)
        }