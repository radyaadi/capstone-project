import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('https://raw.githubusercontent.com/MurwanjaniTejoRiyono/capstone/master/dataset/Crop_recommendation.csv')

features = df[["N", "P", "K","temperature", "humidity", "ph", "rainfall"]]
targets = df["label"]
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = 0.20, random_state = 123, stratify = targets)

RF = RandomForestClassifier(n_estimators=150,max_depth=8,random_state=42)
RF.fit(x_train,y_train)

pickle.dump(RF, open("model.pkl", "wb"))
