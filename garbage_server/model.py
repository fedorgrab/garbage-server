from tensorflow import keras
import constants


def get_trained_model() -> keras.Model:
    classifier = keras.models.load_model(constants.MODEL_PATH)
    return classifier
