from keras import regularizers, Model
from keras.applications import VGG16, InceptionV3, VGG19, NASNetMobile, MobileNet, ResNet50
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Concatenate, Dropout, Dense, \
    BatchNormalization

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
        return "transfer[{base_model}]+fc{dense}+{pretrain}".format(
            base_model=self.base_model,
            dense=len(self.dense_layers),
            pretrain='pretrain' if self.pretrained else 'random_init'
        )

    def _get_base_model(self):
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
        if self.base_model not in valid_models.keys():
            raise ValueError("{} is not a valid base model".format(self.base_model))
        return valid_models[self.base_model](**net_params)

    def create_model(self):
        base_model = self._get_base_model()
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


if __name__ == "__main__":
    snet = SNet('nasnet', [1024, 512])
    snet.create_model()
    snet.model.summary()
