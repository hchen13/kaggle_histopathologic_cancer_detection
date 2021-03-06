import os
from datetime import datetime

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import to_list

from nn.prototypes import ZPNet
from utils import ensure_dir, Chart, save_trial_configs

if "MACHINE_ROLE" in os.environ and os.environ['MACHINE_ROLE'] == 'trainer':
    IMAGE_ROOT = "/home/ethan/Pictures/cancer"
else:
    IMAGE_ROOT = "/Users/ethan/datasets/kaggle_pathology"

TRAIN_DIR = os.path.join(IMAGE_ROOT, 'training')
VALID_DIR = os.path.join(IMAGE_ROOT, 'validation')

def load(image_size, batch_size):
    print("[info] loading images...")
    idg = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_epsilon=1e-6,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=1/255.0,
        # preprocessing_function=crop,
        data_format=None
    )
    train_batches = idg.flow_from_directory(
        TRAIN_DIR,
        color_mode='rgb',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary'
    )
    valid_batches = idg.flow_from_directory(
        VALID_DIR,
        color_mode='rgb',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    return train_batches, valid_batches


def save_model(engine, model_name, weights_only=True):
    folder = 'weights' if weights_only else 'models'
    model_path = os.path.join(folder, model_name + '.h5')
    if weights_only:
        engine.save_weights(model_path)
    else:
        engine.save(model_path)


def train(experiment_name, train_batches, valid_batches, engine, learning_rate, epochs=10, eval_steps=50):
    print("[info] compiling optimizer...")
    opt = optimizers.Adam(lr=learning_rate)
    engine.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

    train_configs = {
        'learning_rate': learning_rate,
        'epochs': epochs
    }

    trial_name = "{}-{}".format(engine.model_name, experiment_name)
    print("[info] initializing training session, session name: {}".format(trial_name.upper()))

    logdir = os.path.join('./logs/', engine.model_name, experiment_name)
    train_chart = Chart(
        log_dir=os.path.join(logdir, 'train'),
        histogram_freq=0,
        write_graph=False,
        write_images=True,
    )
    valid_chart = Chart(
        log_dir=os.path.join(logdir, 'validation'),
        histogram_freq=0,
        write_graph=False,
        write_images=True,
    )
    train_chart.set_model(engine.model)
    valid_chart.set_model(engine.model)

    val_data_gen = iter(valid_batches)

    print("[info] training...")
    data_gen = iter(train_batches)
    steps_per_epoch = len(train_batches)
    global_steps = 0

    def display_on_chart(outputs, chart):
        logs = {}
        outputs = to_list(outputs)
        for l, o in zip(engine.model.metrics_names, outputs):
            logs[l] = o
        chart.draw(logs, global_steps // eval_steps)

    tick = datetime.now()
    for ep in range(epochs):
        print("Epoch {}/{}".format(ep + 1, epochs))
        for local_step in range(steps_per_epoch):
            global_steps += 1
            batch_x, batch_y = next(data_gen)

            outs = engine.model.train_on_batch(batch_x, batch_y)

            if global_steps % 10 == 0:
                tock = datetime.now()
                elapsed = (tock - tick) / 10
                print("steps: {}/{}, progress: {:.2f}%, {:.2f}ms/step, eta: {:.2f}min".format(
                    local_step + 1, steps_per_epoch,
                    (local_step + 1) / steps_per_epoch * 100,
                    elapsed.total_seconds() * 1000,
                    ((steps_per_epoch - local_step - 1) * elapsed).total_seconds() / 60
                ))
                tick = tock

            if global_steps % eval_steps == 0:
                display_on_chart(outs, train_chart)

                print("\tvalidating...")
                val_outs = engine.model.evaluate_generator(
                    val_data_gen,
                    len(valid_batches),
                    workers=0,
                    verbose=0
                )
                display_on_chart(val_outs, valid_chart)

                val_acc = val_outs[1] * 100

                # checkpointing towards the end of each epoch
                if val_acc > 92:
                    print('[info] saving intermediate model, accuracy: {:.2f}%'.format(val_acc))
                    save_model(
                        engine=engine,
                        model_name='{}_tmp{:.0f}'.format(trial_name, val_acc),
                        # weights_only=True
                    )
                    save_trial_configs(trial_name, engine.get_configs(), train_configs)

    print('[info] training complete, saving model...')
    ensure_dir('weights')
    save_model(
        engine=engine,
        model_name=trial_name
        # weights_only=True
    )
    save_trial_configs(trial_name, engine.get_configs(), train_configs)


if __name__ == "__main__":
    image_size = 96
    batch_size = 64
    train_batches, valid_batches = load(image_size=image_size, batch_size=batch_size)

    print("[info] constructing neural network for training...")
    engine = ZPNet(
        base_models=['nasnet', 'vgg16'],
        dense_layers=[2048],
        pretrained=True,
        keep_prob=.5,
        l2_lambda=.02,
        image_size=image_size
    )
    engine.create_model()

    train_params = {
        'learning_rate': 0.00007,
        'epochs': 1,
        'eval_steps': 100
    }
    train(
        experiment_name='trial_1',
        train_batches=train_batches,
        valid_batches=valid_batches,
        engine=engine,
        **train_params
    )
