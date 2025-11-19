from cnn_model import CNNModel
from dataset import DatasetDownloader


class AgePrediction:
    def __init__(self):
        pass

    def train_age_prediction_model(self):
        downloader = DatasetDownloader()
        X_train, X_test, age_train, age_test, gen_train, gen_test, etn_train, etn_test = downloader.generate_dataset()
        cnn_model = CNNModel(X_train, age_train, X_test, age_test)
        cnn_model.build_cnn_model()
        cnn_model.evaluate_model_performance()

if __name__ == "__main__":
    age_predictor = AgePrediction()
    age_predictor.train_age_prediction_model()