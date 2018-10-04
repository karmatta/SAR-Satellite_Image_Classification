import pandas as pd
import numpy as np
import cv2 # Used to manipulated the images
import h5py
np.random.seed(1337) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though

# Import Keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


def median_correction(imgs):
    ig = []
    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]
        d = imgs[i, :, :, 3]
        e = imgs[i, :, :, 4]
        f = imgs[i, :, :, 5]

        med_1 = np.median(a)
        med_2 = np.median(b)
        med_3 = np.median(c)
        med_4 = np.median(d)
        med_5 = np.median(e)
        med_6 = np.median(f)

        a = (a - med_1)
        a = np.square(a)
        b = (b - med_2)
        b = np.square(b)
        c = (c - med_3)
        c = np.square(c)
        d = (d - med_4)
        d = np.square(d)
        e = (e - med_5)
        e = np.square(e)
        f = (f - med_6)
        f = np.square(f)

        ig.append(np.dstack((a, b, c, d, e, f)))

    return np.array(ig)


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = variance(img)

    img_weights = img_variance ** 2 / (img_variance ** 2 + overall_variance ** 2)
    img_output = img_mean + img_weights * (img - img_mean)

    return img_output


# Scale images
def get_scaled_imgs(df):
    imgs = []
    for i, row in df.iterrows():
        # make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2  # plus since log(x*y) = log(x) + log(y)
        band_4 = band_1 - band_2
        band_5 = np.array([a*b for a, b in zip(band_1, band_2)])
        band_6 = (band_1 + band_2)/2
        band_1_e = np.exp(band_1)
        band_1_g = np.power((band_1_e / 40), (1 / 2.2)) * 40
        band_2_e = np.exp(band_2)
        band_2_g = np.power((band_2_e / 40), (1 / 2.2)) * 40

        # use a lee filter to help with speckling
        band_1 = lee_filter(band_1, 4)
        band_2 = lee_filter(band_2, 4)
        band_3 = lee_filter(band_3, 4)
        band_4 = lee_filter(band_4, 4)
        band_5 = lee_filter(band_5, 4)
        band_6 = lee_filter(band_6, 4)

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        d = (band_4 - band_4.mean()) / (band_4.max() - band_4.min())
        e = (band_5 - band_5.mean()) / (band_5.max() - band_5.min())
        f = (band_6 - band_6.mean()) / (band_6.max() - band_6.min())
        g = (band_1_g - band_1_g.mean()) / (band_1_g.max() - band_1_g.min())
        h = (band_2_g - band_2_g.mean()) / (band_2_g.max() - band_2_g.min())

        imgs.append(np.dstack((a, b, c, d, e, f, g, h)))

    return np.array(imgs)


def get_more_images(imgs):
    # more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]
        d = imgs[i, :, :, 3]
        e = imgs[i, :, :, 4]
        f = imgs[i, :, :, 5]
        g = imgs[i, :, :, 6]
        h = imgs[i, :, :, 7]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)
        dv = cv2.flip(d, 1)
        dh = cv2.flip(d, 0)
        ev = cv2.flip(e, 1)
        eh = cv2.flip(e, 0)
        fv = cv2.flip(f, 1)
        fh = cv2.flip(f, 0)
        gv = cv2.flip(g, 1)
        gh = cv2.flip(g, 0)
        hv = cv2.flip(h, 1)
        hh = cv2.flip(h, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv, dv, ev, fv, gv, hv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch, dh, eh, fh, gh, hh)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images


def getModel(drop_prob=0.5):
    # Build keras model

    model = Sequential()

    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 8)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(drop_prob))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(drop_prob))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(drop_prob))

    # CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(drop_prob))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    # Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(drop_prob))

    # Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drop_prob))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def main():

    train = pd.read_json('~/Karthik/StatOil/train.json')

    Xtrain = get_scaled_imgs(train)
    #Xtrain = median_correction(Xtrain)
    Ytrain = np.array(train['is_iceberg'])

    train.inc_angle = train.inc_angle.replace('na', 0)
    idx_tr = np.where(train.inc_angle > 0)

    Ytrain = Ytrain[idx_tr[0]]
    Xtrain = Xtrain[idx_tr[0], ...]

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)

    for train_index, cv_index in sss.split(Xtrain, Ytrain):
        X_train, X_cv = Xtrain[train_index], Xtrain[cv_index]
        y_train, y_cv = Ytrain[train_index], Ytrain[cv_index]
        Xtr_more = get_more_images(X_train)
        Xcv_more = get_more_images(X_cv)
        Ytr_more = np.concatenate((y_train, y_train, y_train))
        Ycv_more = np.concatenate((y_cv, y_cv, y_cv))

    model = getModel()
    model.summary()

    batch_size = 32
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=1,
              callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)

    model.load_weights(filepath='.mdl_wts.hdf5')

    # Evaluate model
    score = model.evaluate(Xcv_more, Ycv_more, verbose=2)
    print('CV loss:', score[0])
    print('CV accuracy:', score[1])

    pt = model.predict(Xcv_more)
    mse = (np.mean((pt - Ycv_more) ** 2))
    print('CV MSE: ', mse)

    # Evaluate model on test
    test = pd.read_json('~/Karthik/StatOil/test.json')
    test.inc_angle = test.inc_angle.replace('na', 0)
    Xtest = (get_scaled_imgs(test))
    predA_test = model.predict(Xtest)

    idx_pred_1 = (np.where(predA_test[:, 0] > 0.95))
    idx_pred_0 = (np.where(predA_test[:, 0] < 0.05))

    Xtrain_pl = np.concatenate((Xtrain,Xtest[idx_pred_1[0],...],Xtest[idx_pred_0[0],...]))
    Ytrain_pl = np.concatenate((Ytrain,np.ones(idx_pred_1[0].shape[0]),np.zeros(idx_pred_0[0].shape[0])))

    pl_kf = KFold(n_splits=5, shuffle=True)

    for train_pl_index, cv_pl_index in pl_kf.split(Xtrain_pl, Ytrain_pl):
        Xtrain_pl, Xpl_cv = Xtrain_pl[train_pl_index], Xtrain_pl[cv_pl_index]
        Ytrain_pl, Ypl_cv = Ytrain_pl[train_pl_index], Ytrain_pl[cv_pl_index]
        break  # you can remove this to add more folds - set to one for demo

    model = getModel()

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wtsPL.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=1, epsilon=1e-4, mode='min')

    model.fit(Xtrain_pl, Ytrain_pl, batch_size=batch_size, epochs=30, verbose=0,
                           callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_data=(Xpl_cv, Ypl_cv))

    model.load_weights(filepath='.mdl_wtsPL.hdf5')

    scorePLCV = model.evaluate(Xpl_cv, Ypl_cv, verbose=0)
    print('Train PL CV score:', scorePLCV[0])
    print('Train PL CV accuracy:', scorePLCV[1])

    score = model.evaluate(Xtrain_pl, Ytrain_pl, verbose=0)
    print('Train PL score:', score[0])
    print('Train PL accuracy:', score[1])

    score = model.evaluate(X_cv, y_cv, verbose=0)
    print('X_cv score:', score[0])
    print('X_cv accuracy:', score[1])

    score = model.evaluate(Xtrain, Ytrain, verbose=0)
    print('Train score:', score[0])
    print('Train accuracy:', score[1])

    predA_test = model.predict(Xtest)

    # Create submission file
    submission = pd.DataFrame({'id': test["id"], 'is_iceberg': predA_test.reshape((predA_test.shape[0]))})
    print(submission.head(10))

    submission.to_csv('submission_v10.csv', index=False)


if __name__ == '__main__':
    main()


