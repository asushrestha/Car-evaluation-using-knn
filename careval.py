import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("car.data")  # loading data set
# print(data.head())

preprocessor = preprocessing.LabelEncoder()
buying = preprocessor.fit_transform(list(data["buying"]))
maint = preprocessor.fit_transform(list(data["maint"]))
door = preprocessor.fit_transform(list(data["door"]))
persons = preprocessor.fit_transform(list(data["persons"]))
lug_boot = preprocessor.fit_transform(list(data["lug_boot"]))
safety = preprocessor.fit_transform(list(data["safety"]))
cls = preprocessor.fit_transform(list(data["class"]))
# convert to fit integer to 0,1,2
predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,test_size=0.1)
# creating test(10%) and  train model(90%)

model = KNeighborsClassifier(n_neighbors=9)  # value of k is defined here
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
predicted_data = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]  # to visualize as string
for x in range(len(x_test)):
    print("Predicted: ", names[predicted_data[x]], "Data:", x_test[x], "actual data :", names[y_test[x]])
    kneigh = model.kneighbors([x_test[x]], 9, True)  # finding the neighbors
    print("'kneighbors are: \t", kneigh);
