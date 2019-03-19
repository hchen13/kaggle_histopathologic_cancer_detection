from keras import regularizers, Model, Input
from keras.applications import VGG16, InceptionV3, VGG19, NASNetMobile, MobileNet, ResNet50
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Concatenate, Dropout, Dense, \
    BatchNormalization, Conv2D

from nn.base import Prototype


class SNet(Prototype):
    def __init__(self, base_model: str, dense_layers: list, pretrained=True, keep_prob=1, l2_lambda=0, image_size=96):
        super(SNet, self).__init__(image_size)
        self.base_model = base_model.lower()
        self.dense_layers = dense_layers
        self.keep_prob = keep_prob
        self.l2_lambda = l2_lambda
        self.pretrained = pretrained

    @property
    def model_name(self):
        return "trans_{base_model}+fc{dense}+{pretrain}".format(
            base_model=self.base_model,
            dense=len(self.dense_layers),
            pretrain='pretrain' if self.pretrained else 'random_init'
        )

    def get_configs(self):
        configs = {
            'base_model': self.base_model,
            'pretrained': self.pretrained,
            'dense_layers': self.dense_layers,
            'image_shape': self.image_shape,
            'keep_prob': self.keep_prob,
            'l2_lambda': self.l2_lambda
        }
        return configs

    def _get_base_model(self, model_name):
        valid_models = {
            'vgg16': VGG16,
            'vgg19': VGG19,
            'inception': InceptionV3,
            'nasnet': NASNetMobile,
            'mobilenet': MobileNet,
            'resnet': ResNet50,
        }
        net_params = {
            'weights': 'imagenet' if self.pretrained else None,
            'include_top': False,
            'input_shape': self.image_shape
        }
        if model_name not in valid_models.keys():
            raise ValueError("{} is not a valid base model".format(model_name))
        return valid_models[model_name](**net_params)

    def create_model(self):
        base_model = self._get_base_model(self.base_model)
        x = base_model.output

        ave_pool = GlobalAveragePooling2D()(x)
        max_pool = GlobalMaxPooling2D()(x)
        flatten = Flatten()(x)
        bottleneck = Concatenate()([ave_pool, max_pool, flatten])

        out = Dropout(1 - self.keep_prob)(bottleneck)

        for neurons in self.dense_layers:
            out = Dense(
                neurons,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_lambda),
            )(out)
            out = BatchNormalization()(out)
            out = Dropout(1 - self.keep_prob)(out)

        preds = Dense(1, activation='sigmoid')(out)

        retrain_layers = len(base_model.layers)

        for layer in base_model.layers[:len(base_model.layers) - retrain_layers]:
            layer.trainable = False

        self.model = Model(inputs=base_model.inputs, outputs=preds)


class ZPNet(SNet):
    def __init__(self, base_models: list, dense_layers: list, pretrained=True, keep_prob=1, l2_lambda=0, image_size=96):
        super(SNet, self).__init__(image_size)
        self.base_models = [bm.lower() for bm in base_models]
        self.dense_layers = dense_layers
        self.keep_prob = keep_prob
        self.l2_lambda = l2_lambda
        self.pretrained = pretrained

    def get_configs(self):
        configs = {
            'base_models': self.base_models,
            'pretrained': self.pretrained,
            'dense_layers': self.dense_layers,
            'image_shape': self.image_shape,
            'keep_prob': self.keep_prob,
            'l2_lambda': self.l2_lambda
        }
        return configs

    @property
    def model_name(self):
        return "zpnet_fc{dense}".format(
            dense=len(self.dense_layers),
        )

    def create_model(self):
        input_image = Input(shape=self.image_shape)
        base1 = self._get_base_model(self.base_models[0])
        base2 = self._get_base_model(self.base_models[1])
        out1 = base1(input_image)
        out2 = base2(input_image)
        bottleneck = Concatenate()([out1, out2])
        bottleneck = Flatten()(bottleneck)

        out = Dropout(1 - self.keep_prob)(bottleneck)

        for neurons in self.dense_layers:
            out = Dense(
                neurons,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_lambda),
            )(out)
            out = BatchNormalization()(out)
            out = Dropout(1 - self.keep_prob)(out)

        preds = Dense(1, activation='sigmoid')(out)
        self.model = Model(inputs=input_image, outputs=preds)


if __name__ == "__main__":
    # engine = SNet('vgg16', [4096], image_size=96)
    # engine.create_model()
    engine = ZPNet(['vgg16', 'nasnet'], [4096], image_size=96)
    engine.create_model()
    engine.model.summary()

    import numpy as np
    engine.model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])
    x, y = np.random.randn(10, 96, 96, 3), np.random.randint(0, 2, size=(10,))
    engine.model.fit(x, y)