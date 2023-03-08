from enum import Enum
import lightgbm as lgbm

class ModelType(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1

def get_lgbm_model(model_type: ModelType, params_lgbm):
    if model_type == ModelType.CLASSIFICATION:
        return lgbm.LGBMClassifier(**params_lgbm)
    elif model_type == ModelType.REGRESSION:
        return lgbm.LGBMRegressor(**params_lgbm)