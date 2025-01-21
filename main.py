from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, Dense, Dropout
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
from utils import AVAILABLE_EMOTIONS

from deep_emotion_recognition import DeepEmotionRecognizer
# initialize instance
# inherited from emotion_recognition.EmotionRecognizer
# default parameters (LSTM: 128x2, Dense:128x2)
deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'happy', 'fear', 'disgust', 'boredom'], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
# train the model
deeprec.train()
# get the accuracy
print(deeprec.test_score())
# predict angry audio sample
prediction = deeprec.predict('data/validation/Actor_10/03-02-05-02-02-02-10_angry.wav')
print(f"Prediction: {prediction}")