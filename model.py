#Gerekli kütüphaneleri import ediyoruz

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn import model_selection

#CLASSIFIER
from lightgbm import LGBMClassifier
import pickle

from sklearn import preprocessing 
le = preprocessing.LabelEncoder()

data = pd.read_csv("diabetes.csv")
df = data.copy()
df = df.rename(columns = {"Pregnancies":"HamilelikDurumu",
                         "Glucose":"Glikoz",
                         "BloodPressure":"KanBasinci",
                         "SkinThickness":"CiltKalinligi",
                         "BMI":"VKIndeksi",
                         "DiabetesPedigreeFunction":"SoyagaciIslevi",
                         "Age":"Yas",
                         "Outcome":"Sonuc"})

df.Glikoz = df.Glikoz.replace(0, np.NaN)
df.KanBasinci = df.KanBasinci.replace(0, np.NaN)
df.CiltKalinligi = df.CiltKalinligi.replace(0, np.NaN)
df.VKIndeksi = df.VKIndeksi.replace(0, np.NaN)
df.SoyagaciIslevi = df.SoyagaciIslevi.replace(0, np.NaN)
df.Yas = df.Yas.replace(0, np.NaN)
df.Insulin = df.Insulin.replace(0, np.NaN)

df.Insulin = df.Insulin.fillna(df.Insulin.mean())
df.Yas = df.Yas.fillna(df.Yas.mean())
df.SoyagaciIslevi = df.SoyagaciIslevi.fillna(df.SoyagaciIslevi.mean())
df.VKIndeksi = df.VKIndeksi.fillna(df.VKIndeksi.mean())
df.CiltKalinligi = df.CiltKalinligi.fillna(df.CiltKalinligi.mean())
df.KanBasinci = df.KanBasinci.fillna(df.KanBasinci.mean())
df.Glikoz = df.Glikoz.fillna(df.Glikoz.mean())


x = df.drop("Sonuc", axis = 1)
y = df["Sonuc"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 100)

lgbm_model = LGBMClassifier()

lgbm_params = {"learning_rate" : [0.001, 0.1, 0.01],
              "n_estimators" : [200, 500, 100, 800],
              "max_depth" : [1, 2, 3, 5, 8],
              "random_state" : np.arange(1,50,5)}

#lgbm_cv = GridSearchCV(lgbm_model, lgbm_params, cv = 10, verbose = 2, n_jobs = -1).fit(x_train, y_train)
#lgbm_cv.best_params_

tuned = LGBMClassifier(learning_rate = 0.1, max_depth = 2, n_estimators = 200, random_state = 1).fit(x_train, y_train)

pickle.dump(tuned, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

print("Model.py calisti..")
