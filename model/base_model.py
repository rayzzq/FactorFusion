from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_X, train_Y):
        pass

    @abstractmethod
    def predict(self, test_X):
        pass

    @abstractmethod
    def save_model(self, path):
        pass
    
    @classmethod
    @abstractmethod
    def load_model(cls, path):
        pass
