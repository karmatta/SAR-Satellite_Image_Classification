import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = 10, 10

from keras.layers import Dropout
from keras.layers import GlobalMaxPooling2D

from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.applications.vgg16 import VGG16

from keras.layers import Dense, Input, concatenate

from keras.preprocessing.image import ImageDataGenerator


batch_size = 64


def gen_image_data(data):
    # Generate the training data
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2"]])
    X_band_1_kuan = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1_kuan"]])
    X_band_2_kuan = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2_kuan"]])
    #X_band_3 = (X_band_1 + X_band_2) / 2
    band_1_g = np.power((np.exp(X_band_1) / 64), (1 / 2.2)) * 64
    #band_2_g = np.power((np.exp(X_band_2) / 64), (1 / 2.2)) * 64
    # X_band_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in train["inc_angle"]])
    X_train = np.concatenate([band_1_g[:, :, :, np.newaxis]
                                 , X_band_1_kuan[:, :, :, np.newaxis]
                                 , X_band_2_kuan[:, :, :, np.newaxis]], axis=-1)

    return(X_train)


gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.,
                         height_shift_range = 0.,
                         channel_shift_range=0,
                         zoom_range = 0.2,
                         rotation_range = 10)


# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=55)
    genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=55)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]


# Finally create generator
def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=10, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def getVggAngleModel(X_train):
    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=X_train.shape[1:], classes=1)
    x = base_model.get_layer('block5_pool').output

    x = GlobalMaxPooling2D()(x)
    merge_one = concatenate([x, angle_layer])
    merge_one = Dense(512, activation='relu', name='fc2')(merge_one)
    merge_one = Dropout(0.3)(merge_one)
    merge_one = Dense(512, activation='relu', name='fc3')(merge_one)
    merge_one = Dropout(0.3)(merge_one)

    predictions = Dense(1, activation='sigmoid')(merge_one)

    model = Model(input=[base_model.input, input_2], output=predictions)

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


# Using K-fold Cross Validation with Data Augmentation.
def myAngleCV(X_train, X_angle, X_test, target_train, X_test_angle):
    K = 3
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))
    y_test_pred_log = 0
    y_train_pred_log = 0
    y_valid_pred_log = 0.0 * target_train
    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n===================FOLD=', j)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout = target_train[test_idx]

        # Angle
        X_angle_cv = X_angle[train_idx]
        X_angle_hold = X_angle[test_idx]

        # define file path and get callbacks
        file_path = "%s_aug_model_weights.hdf5" % j
        callbacks = get_callbacks(filepath=file_path, patience=5)
        gen_flow = gen_flow_for_two_inputs(X_train_cv, X_angle_cv, y_train_cv)
        galaxyModel = getVggAngleModel(X_train_cv)
        galaxyModel.fit_generator(
            gen_flow,
            steps_per_epoch=24,
            epochs=100,
            shuffle=True,
            verbose=1,
            validation_data=([X_holdout, X_angle_hold], Y_holdout),
            callbacks=callbacks)

        # Getting the Best Model
        galaxyModel.load_weights(filepath=file_path)
        # Getting Training Score
        score = galaxyModel.evaluate([X_train_cv, X_angle_cv], y_train_cv, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        # Getting Test Score
        score = galaxyModel.evaluate([X_holdout, X_angle_hold], Y_holdout, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # Getting validation Score.
        pred_valid = galaxyModel.predict([X_holdout, X_angle_hold])
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        # Getting Test Scores
        temp_test = galaxyModel.predict([X_test, X_test_angle])
        y_test_pred_log += temp_test.reshape(temp_test.shape[0])

        # Getting Train Scores
        temp_train = galaxyModel.predict([X_train, X_angle])
        y_train_pred_log += temp_train.reshape(temp_train.shape[0])

    y_test_pred_log = y_test_pred_log / K
    y_train_pred_log = y_train_pred_log / K

    print('\n Train Log Loss Validation= ', log_loss(target_train, y_train_pred_log))
    print(' Test Log Loss Validation= ', log_loss(target_train, y_valid_pred_log))
    return y_test_pred_log


def main():

    train = pd.read_json('~/Karthik/StatOil/train_trans_band_kuan.json')
    test = pd.read_json('~/Karthik/StatOil/test_trans_band_kuan.json')

    target_train = train['is_iceberg']
    test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
    train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')  # We have only 133 NAs.
    train['inc_angle'] = train['inc_angle'].fillna(method='pad')
    X_angle = train['inc_angle']
    test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
    X_test_angle = test['inc_angle']

    X_train = gen_image_data(train)
    X_test = gen_image_data(test)
    gen_imgs = {"X_train":X_train, 'X_test': X_test}

    X_train = gen_imgs["X_train"]
    X_test = gen_imgs["X_test"]

    preds = myAngleCV(X_train, X_angle, X_test, target_train, X_test_angle)

    # Create submission file
    submission = pd.DataFrame({'id': test["id"], 'is_iceberg': preds.reshape((preds.shape[0]))})
    print(submission.head(10))

    submission.to_csv('VGG_kuanb1b2_gammab1.csv', index=False)


if __name__ == '__main__':
    main()

