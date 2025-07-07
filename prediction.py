import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 

df=pd.read_csv("veriler.csv")

x=df[["yas","maas","deneyim","sehir"]]
y=df["ev_fiyati"]

categoricial=["sehir"]
numeric=["yas","maas","deneyim"]

ct=ColumnTransformer(transformers=[("pre",OneHotEncoder(handle_unknown="ignore"),categoricial)],remainder="passthrough")

model=Pipeline(steps=[("preprocessor",ct),("regressor",RandomForestRegressor(n_estimators=100,random_state=0))])

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.33,random_state=0)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("Mean Squared Error",mse)
print("R^2",r2)