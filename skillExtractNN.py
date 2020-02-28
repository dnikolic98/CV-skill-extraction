from tensorflow import keras
import numpy as np

'''
Neural network used for extracting skills from CVs.

Model consists of three input layers:
    First (lstm) takes variable length vector of arbitrary number of words (phrase).
    Second (lstm) takes context of a phrase. Variable length vector of a phrase and n-words to the right and left of a phrase.
    Third (dense) takes fixed size vector representing presence or absence of binary features
'''
class SkillsExtractorNN:

    def __init__(self, word_features_dim, dense_features_dim):
        lstm_input_phrase = keras.layers.Input(shape=(None, word_features_dim))
        lstm_input_cont = keras.layers.Input(shape=(None, word_features_dim))
        dense_input = keras.layers.Input(shape=(dense_features_dim,))

        lstm_emb_phrase = keras.layers.LSTM(256)(lstm_input_phrase)
        lstm_emb_phrase = keras.layers.Dense(128, activation='relu')(lstm_emb_phrase)

        lstm_emb_cont = keras.layers.LSTM(256)(lstm_input_cont)
        lstm_emb_cont = keras.layers.Dense(128, activation='relu')(lstm_emb_cont)

        dense_emb = keras.layers.Dense(512, activation='relu')(dense_input)
        dense_emb = keras.layers.Dense(256, activation='relu')(dense_emb)

        x = keras.layers.concatenate([lstm_emb_phrase, lstm_emb_cont, dense_emb])
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)

        main_output = keras.layers.Dense(2, activation='softplus')(x)

        self.model = keras.models.Model(inputs=[lstm_input_phrase, lstm_input_cont, dense_input],
                                        outputs=main_output)

        optimizer = keras.optimizers.Adam(lr=0.0001)

        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def onehot_transform(self, y):
        onehot_y = []
        for numb in y:
            onehot_arr = np.zeros(2)
            onehot_arr[numb] = 1
            onehot_y.append(np.array(onehot_arr))

        return np.array(onehot_y)

    def fit(self, x_lstm_phrase, x_lstm_context, x_dense, y,
            val_split=0.25, patience=5, max_epochs=1000, batch_size=32):

        x_lstm_phrase_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_phrase, dtype=np.float32)
        x_lstm_context_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_context, dtype=np.float32)

        y_onehot = self.onehot_transform(y)

        return self.model.fit([x_lstm_phrase_seq, x_lstm_context_seq, x_dense],
                       y_onehot,
                       batch_size=batch_size,
                       epochs=max_epochs,
                       validation_split=val_split,
                       callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)])

    def evaluate(self, x_lstm_phrase, x_lstm_context, x_dense, y):
        x_lstm_phrase_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_phrase, dtype=np.float32)
        x_lstm_context_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_context, dtype=np.float32)

        y_onehot = self.onehot_transform(y)

        return self.model.evaluate([x_lstm_phrase_seq, x_lstm_context_seq, x_dense], y_onehot, verbose=0)
        
    def predict(self, x_lstm_phrase, x_lstm_context, x_dense):
        x_lstm_phrase_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_phrase, dtype=np.float32)
        x_lstm_context_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_context, dtype=np.float32)
        
        y = self.model.predict([x_lstm_phrase_seq, x_lstm_context_seq, x_dense])
        return y

    def load(self, path):
        self.model.load_weights(path)
        #print("Loaded model from disk")

    def score(self, x_lstm_phrase, x_lstm_context, x_dense, Y):
        hit = 0
        
        x_lstm_phrase_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_phrase, dtype=np.float32)
        x_lstm_context_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_context, dtype=np.float32)
        
        y = self.model.predict([x_lstm_phrase_seq, x_lstm_context_seq, x_dense])
        
        for i in range(len(Y)):
            if(np.argmax(y[i]) == Y[i]):
                hit += 1
        return hit/len(Y)
