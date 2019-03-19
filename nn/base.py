from keras.models import load_model

class Prototype:

    def __init__(self, image_size):
        self.model = None
        self.image_shape = (image_size, image_size, 3)

    @property
    def model_name(self):
        raise NotImplementedError

    def create_model(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, path):
        self.model.save(path)

    def save_weights(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model = load_model(path)
        self.model._make_predict_function()

    def load_weights(self, path):
        assert self.model is not None
        self.model.load_weights(path)
