import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Embedding, LSTM

def preprocess_input(data):
        inputs = np.array([list(map(int, list(seq))) for seq in data['input_str']])
        return inputs

def build_cnn_rnn_model(input_shape):
        model = Sequential()
        

        model.add(Embedding(input_dim=10, output_dim=8, input_length=input_shape))
        model.add(Conv1D(filters=32, kernel_size=4, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(LSTM(32, return_sequences=False))
        
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

# these are dummy models
class MLModel():
    def __init__(self) -> None:
        pass
    
    def train(self, X, y):
        NotImplemented
    
    def predict(self, X):
        NotImplemented
    
class TextSeqModel(MLModel):
    train_data = pd.read_csv("datasets/train/train_text_seq.csv")

    

    X_train = preprocess_input(train_data)

    # Label encoding
    y_train = train_data['label'].values
    
    input_shape = X_train.shape[1] 
    total_train_size = X_train.shape[0]

    model = build_cnn_rnn_model(input_shape)
    
    # Aditional manual build before summary evaluation
    model.build(input_shape=(None, input_shape))
    model.summary()
    total_params = model.count_params()
    print(f"Total number of trainable parameters: {total_params}")
    
    model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)



    def __init__(self) -> None:
        pass

    def predict(self, X):# random predictions
        col_name = ["input_str"]
        X = pd.DataFrame(X,columns=col_name)
        X_test = preprocess_input(X)
        preds = (self.model.predict(X_test) > 0.5).astype(int)
        lst=[]
        for p in preds:
            lst.append(p[0])
        return lst
    
    
class EmoticonModel(MLModel):
    dat = pd.read_csv("datasets/train/train_emoticon.csv")
    df_split = dat['input_emoticon'].apply(list).apply(pd.Series)

    df_split.columns = [f'emoji_{i+1}' for i in range(df_split.shape[1])]
    emoji_columns = [f'emoji_{i}' for i in range(1, 14)]
    X = df_split # Features (all emoji columns)
    y = dat['label']

    encoder = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
    encoded_columns = encoder.fit_transform(X[emoji_columns])
    encoded_col_names = encoder.get_feature_names_out(emoji_columns)
    X_subset = pd.DataFrame(encoded_columns, columns=encoded_col_names)
    pipeline = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=500)

    pipeline.fit(X_subset,y)


    def __init__(self) -> None:
        pass

    def predict(self, X):
        for idx,str in enumerate(X):
            lst = []
            for emo in str:
                lst.append(emo)
            X[idx] = lst
        X_test = pd.DataFrame(X,columns=self.emoji_columns)
        test_encode = self.encoder.transform(X_test[self.emoji_columns])
        test_encode_cols = self.encoder.get_feature_names_out(self.emoji_columns)
        encoded_X_test = pd.DataFrame(test_encode,columns=test_encode_cols)
        res = self.pipeline.predict(encoded_X_test)
        return res
    
class FeatureModel(MLModel):
    train_feature = "datasets/train/train_feature.npz"

    tdata_feature = np.load(train_feature)

    Xd_feature = tdata_feature['features']
    Y_feature = tdata_feature['label']
    X_feature=[]
    for mat in Xd_feature:
        X_feature.append(mat.flatten())


    pca = PCA(n_components=100)

    X_feature = pca.fit_transform(X_feature)

    feature_columns = [f'feature_{i}' for i in range(1, 101)]
    X_feature = pd.DataFrame(X_feature,columns=feature_columns)
    logreg = LogisticRegression()
    logreg.fit(X_feature,Y_feature)
    def __init__(self) -> None:
        pass

    def predict(self, X): # random predictions
        X_test_feature=[]
        for mat in X:
            X_test_feature.append(mat.flatten())
        X_test_feature = self.pca.transform(X_test_feature)
        X_test_feature = pd.DataFrame(X_test_feature,columns=self.feature_columns)
        log_pred = self.logreg.predict(X_test_feature)
        return log_pred

    
class CombinedModel(MLModel):
    train_emoji = "datasets/train/train_emoticon.csv"

    train_feature = "datasets/train/train_feature.npz"

    train_seq = "datasets/train/train_text_seq.csv"

    ###########################################################33
    dat_emoji = pd.read_csv(train_emoji)
    df_split_emoji = dat_emoji['input_emoticon'].apply(list).apply(pd.Series)

    df_split_emoji.columns = [f'emoji_{i+1}' for i in range(df_split_emoji.shape[1])]
    emoji_columns = [f'emoji_{i}' for i in range(1, 14)]


    X_emoji = df_split_emoji # Features (single emoji per column)
    Y_emoji = dat_emoji['label']
    ##############################################################
    tdata_feature = np.load(train_feature)
    Xd_feature = tdata_feature['features']
    Y_feature = tdata_feature['label']
    X_feature=[]
    for mat in Xd_feature:
        X_feature.append(mat.flatten())


    pca = PCA(n_components=100)

    X_feature = pca.fit_transform(X_feature)

    feature_columns = [f'feature_{i}' for i in range(1, 101)]
    X_feature = pd.DataFrame(X_feature,columns=feature_columns)
    ###########################################################
    dat_seq = pd.read_csv(train_seq)
    df_split_seq = dat_seq['input_str'].apply(list).apply(pd.Series)
    df_split_seq.columns = [f'num_{i+1}' for i in range(df_split_seq.shape[1])]
    num_columns = [f'num_{i}' for i in range(1,51)]
    X_seq = df_split_seq
    Y_seq = dat_seq['label']
    ###########################################################
    encoder = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
    encoded_columns = encoder.fit_transform(X_emoji[emoji_columns])
    encoded_col_names = encoder.get_feature_names_out(emoji_columns)
    X_emoji = pd.DataFrame(encoded_columns, columns=encoded_col_names)

    X_combo = pd.concat([X_emoji,X_feature,X_seq],axis=1)
    logreg = LogisticRegression(max_iter=500)
    logreg.fit(X_combo,Y_emoji)

    def __init__(self) -> None:
        pass

    def predict(self, X1, X2, X3): # random predictions
        for idx,str in enumerate(X2):
            lst = []
            for emo in str:
                lst.append(emo)
            X2[idx] = lst
        X2_test = pd.DataFrame(X2,columns=self.emoji_columns)
        test_encode = self.encoder.transform(X2_test[self.emoji_columns])
        test_encode_cols = self.encoder.get_feature_names_out(self.emoji_columns)
        encoded_X2_test = pd.DataFrame(test_encode,columns=test_encode_cols)

        X1_test_feature=[]
        for mat in X1:
            X1_test_feature.append(mat.flatten())
        X1_test_feature = self.pca.transform(X1_test_feature)
        X1_test_feature = pd.DataFrame(X1_test_feature,columns=self.feature_columns)

        for idx,str in enumerate(X3):
            lst=[]
            for dig in str:
                lst.append(dig)
            X3[idx]=lst
        X3_test = pd.DataFrame(X3,columns=self.num_columns)

        X_test_combo = pd.concat([encoded_X2_test,X1_test_feature,X3_test],axis=1)
        log_pred = self.logreg.predict(X_test_combo)
        return log_pred
    
    
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == '__main__':
    # read datasets
    test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']
    test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
    test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()
    
    # your trained models 
    feature_model = FeatureModel()
    text_model = TextSeqModel()
    emoticon_model  = EmoticonModel()
    best_model = CombinedModel()
    
    # predictions from your trained models
    pred_feat = feature_model.predict(test_feat_X)
    pred_emoticons = emoticon_model.predict(test_emoticon_X)
    pred_text = text_model.predict(test_seq_X)
    pred_combined = best_model.predict(test_feat_X, test_emoticon_X, test_seq_X)
    
    # saving prediction to text files
    save_predictions_to_file(pred_feat, "pred_feat.txt")
    save_predictions_to_file(pred_emoticons, "pred_emoticon.txt")
    save_predictions_to_file(pred_text, "pred_text.txt")
    save_predictions_to_file(pred_combined, "pred_combined.txt")
    
    
