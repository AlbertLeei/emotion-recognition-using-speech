from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, Dense, Dropout
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
from utils import AVAILABLE_EMOTIONS

from deep_emotion_recognition import DeepEmotionRecognizer
from emotion_recognition import EmotionRecognizer


# from sklearn.svm import SVC
# # init a model, let's use SVC
# my_model = SVC()
# # pass my model to EmotionRecognizer instance
# # and balance the dataset
# rec = EmotionRecognizer(model=my_model, emotions=['angry', 'sad', 'neutral', 'fear', 'happy'], balance=True, verbose=1)
# # train the model
# rec.train()
# # check the test accuracy for that model
# print("Test score:", rec.test_score())
# # check the train accuracy for that model
# print("Train score:", rec.train_score())


# initialize instance
# inherited from emotion_recognition.EmotionRecognizer
# default parameters (LSTM: 128x2, Dense:128x2)
deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'happy', 'fear', 'disgust'], n_rnn_layers=3, n_dense_layers=2, rnn_units=128, dense_units=128, epochs = 300)
# train the model
deeprec.train()
# get the accuracy
print(deeprec.test_score())
# # predict angry audio sample
# # prediction = deeprec.predict('data/validation/Actor_10/03-02-05-02-02-02-10_angry.wav')
# # print(f"Prediction: {prediction}")

# deeprec.determine_best_model()
# print(deeprec.model.__class__.__name__, "is the best")