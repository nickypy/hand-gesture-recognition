import pickle
import cv2
import numpy as np

from os import listdir, makedirs, rename
from os.path import isdir, isfile, join
from data import Data
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

def process_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (64, 64))

    return img

def process_frame(img, text):
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return img


if __name__ == '__main__':
    data = pickle.load(open("dataset.pkl", "rb"))
    model = load_model('model.h5')
    cap = cv2.VideoCapture(0)

    encoder = LabelEncoder()
    encoder.fit(data.train_y)

    prediction = ""
    while(True):
        # find max prediction 10 frames at a time
        total_preds = np.zeros(6)

        for _ in range(5):
            # Capture frame-by-frame
            ret, frame = cap.read()
            new_frame = cv2.flip(frame, 1)

            # crop webcam image into a square
            img = new_frame[300:900, 0:600]

            x = process_image(img)
            nx, ny, nz = x.shape
            x = x.reshape((1, nx, ny, nz))
            x = x.astype('float32')
            mean = np.mean(x)
            std_dev = np.std(x)

            x -= mean
            x /= std_dev

            prediction_list = model.predict(np.array(x))
            total_preds += prediction_list[0]

            new_frame = process_frame(img, f'{prediction}')
            cv2.imshow('hands', new_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

        prediction = np.argmax(total_preds)
        prediction = encoder.inverse_transform(prediction)
