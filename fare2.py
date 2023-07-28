import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

df_fare = pd.read_csv('UPDATEDitineries-airportCodes3.csv', low_memory=False)

#decision tree
X = df_fare[['isBasicEconomy','isRefundable','minutes','seatsRemaining','baseFare','destinationAirportDistance','startingAirportDistance', 'totalTravelDistance', "totalFare"]]
Y = df_fare["connections"] 
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)
model = tree.DecisionTreeClassifier(max_depth = 5)
model.fit(x_train, y_train)
predictions_test = model.predict(x_test)
predictions_train = model.predict(x_train)
accuracy_train = accuracy_score(y_train, predictions_train)
accuarcy_test = accuracy_score(y_test, predictions_test)
print("Decision Tree - Connections")
print("Train")
print(accuracy_train)
print("test")
print(accuarcy_test)



#decision tree
X = df_fare[['isBasicEconomy','isRefundable','minutes','seatsRemaining','baseFare','destinationAirportDistance','startingAirportDistance', 'totalTravelDistance', "totalFare"]]
Y = df_fare["connections"] 
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)
model = tree.DecisionTreeClassifier(max_depth = 5)
model.fit(x_train, y_train)
predictions_test = model.predict(x_test)
predictions_train = model.predict(x_train)
accuracy_train = accuracy_score(y_train, predictions_train)
accuarcy_test = accuracy_score(y_test, predictions_test)
print("Decision Tree - Connections")
print("Train")
print(accuracy_train)
print("test")
print(accuarcy_test)

#kmeans
model = LogisticRegression(max_iter=120)
model.fit(x_train, y_train)
predictions_test = model.predict(x_test)
predictions_train = model.predict(x_train)
accuracy_train = accuracy_score(y_train, predictions_train)
accuarcy_test = accuracy_score(y_test, predictions_test)
print("Logistic Tree - Connections")
print("Train")
print(accuracy_train)
print("test")
print(accuarcy_test)

model = KNeighborsClassifier()
model.fit(x_train, y_train)
predictions_test = model.predict(x_test)
predictions_train = model.predict(x_train)
accuracy_train = accuracy_score(y_train, predictions_train)
accuarcy_test = accuracy_score(y_test, predictions_test)
print("KNeighborsClassifier - Connections")
print("Train")
print(accuracy_train)
print("test")
print(accuarcy_test)

#decision tree - airline
#decision tree
X = df_fare[['segmentsEpochTimeInSeconds4','segmentsEpochTimeInSeconds3','segmentsEpochTimeInSeconds2','segmentsEpochTimeInSeconds1','isBasicEconomy',"segmentsDistance4",'segmentsDistance3','segmentsDistance2','segmentsDistance1','segmentsDurationInSeconds4','segmentsDurationInSeconds3','segmentsDurationInSeconds2','segmentsDurationInSeconds1',"connections",'isRefundable','minutes','seatsRemaining','baseFare','destinationAirportDistance','startingAirportDistance', 'totalTravelDistance', "totalFare"]]
Y = df_fare["segmentsAirlineName"] 
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)
model = tree.DecisionTreeClassifier(max_depth = 5)
model.fit(x_train, y_train)
predictions_test = model.predict(x_test)
predictions_train = model.predict(x_train)
accuracy_train = accuracy_score(y_train, predictions_train)
accuarcy_test = accuracy_score(y_test, predictions_test)
print("Decision Tree - segmentsAirlineName")
print("Train")
print(accuracy_train)
print("test")
print(accuarcy_test)

#kmeans
# model = LogisticRegression(max_iter=120)
# model.fit(x_train, y_train)
# predictions_test = model.predict(x_test)
# predictions_train = model.predict(x_train)
# accuracy_train = accuracy_score(y_train, predictions_train)
# accuarcy_test = accuracy_score(y_test, predictions_test)
# print("Logistic Tree - segmentsAirlineName")
# print("Train")
# print(accuracy_train)
# print("test")
# print(accuarcy_test)

model = KNeighborsClassifier()
model.fit(x_train, y_train)
predictions_test = model.predict(x_test)
predictions_train = model.predict(x_train)
accuracy_train = accuracy_score(y_train, predictions_train)
accuarcy_test = accuracy_score(y_test, predictions_test)
print("KNeighborsClassifier - segmentsAirlineName")
print("Train")
print(accuracy_train)
print("test")
print(accuarcy_test)

#decision tree - segment distance
#decision tree
X = df_fare[['segmentsEpochTimeInSeconds4','segmentsEpochTimeInSeconds3','segmentsEpochTimeInSeconds2','segmentsEpochTimeInSeconds1','isBasicEconomy',"segmentsDistance4",'segmentsDistance3','segmentsDistance2','segmentsDistance1','segmentsDurationInSeconds4','segmentsDurationInSeconds3','segmentsDurationInSeconds2','segmentsDurationInSeconds1',"connections",'isRefundable','minutes','seatsRemaining','baseFare','destinationAirportDistance','startingAirportDistance', 'totalTravelDistance', "totalFare"]]
Y = df_fare['segmentsEquipmentDescription'] 
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)
model = tree.DecisionTreeClassifier(max_depth = 5)
model.fit(x_train, y_train)
predictions_test = model.predict(x_test)
predictions_train = model.predict(x_train)
accuracy_train = accuracy_score(y_train, predictions_train)
accuarcy_test = accuracy_score(y_test, predictions_test)
print("Decision Tree - segmentsEquipmentDescription")
print("Train")
print(accuracy_train)
print("test")
print(accuarcy_test)

#kmeans
# model = LogisticRegression(max_iter=120)
# model.fit(x_train, y_train)
# predictions_test = model.predict(x_test)
# predictions_train = model.predict(x_train)
# accuracy_train = accuracy_score(y_train, predictions_train)
# accuarcy_test = accuracy_score(y_test, predictions_test)
# print("Logistic Regression - segmentsEquipmentDescription")
# print("Train")
# print(accuracy_train)
# print("test")
# print(accuarcy_test)

model = KNeighborsClassifier()
model.fit(x_train, y_train)
predictions_test = model.predict(x_test)
predictions_train = model.predict(x_train)
accuracy_train = accuracy_score(y_train, predictions_train)
accuarcy_test = accuracy_score(y_test, predictions_test)
print("KNeighborsClassifier - segmentsEquipmentDescription")
print("Train")
print(accuracy_train)
print("test")
print(accuarcy_test)

#LinearRegression - cost
X = df_fare[['segmentsEpochTimeInSeconds4','segmentsEpochTimeInSeconds3','segmentsEpochTimeInSeconds2','segmentsEpochTimeInSeconds1','isBasicEconomy',"segmentsDistance4",'segmentsDistance3','segmentsDistance2','segmentsDistance1','segmentsDurationInSeconds4','segmentsDurationInSeconds3','segmentsDurationInSeconds2','segmentsDurationInSeconds1',"connections",'isRefundable','minutes','seatsRemaining','destinationAirportDistance','startingAirportDistance', 'totalTravelDistance']]
Y = df_fare['totalFare'] 
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
predictions_test = model.predict(x_test)
mse = mean_squared_error(y_test, predictions_test)
score = model.score(x_test, y_test) 
print("LinearRegression - cost")
print("Prediction")
print(predictions_test)
print("Mean Squared")
print(mse)
print("Score")
print(score)




# X = df_fare[['destinationAirport','startingAirport']]
# Y = df_fare["totalTravelDistance"] 
# x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)
# model = LinearRegression()
# model.fit(x_train, y_train)
# predictions = model.predict(x_test)
# accuracy = accuracy_score(predictions, y_test)
# print(accuracy)