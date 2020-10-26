#knn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from disease import disease
def knn():
    df=pd.read_csv('parkinsons.data',index_col = 0)
    df.head()
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    scaler.fit(df.drop('status', axis=1))
    scaled_features = scaler.transform(df.drop('status', axis=1))

    df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    df_feat.head()

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split( scaled_features, df['status'], test_size=0.30,random_state=101)
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    from sklearn import metrics
    accuracy=metrics.accuracy_score(y_test, pred)

    print("Accuracy is:",accuracy*100)
    names = ["doesnot have pd", "has pd"]
    disease["total"] = len(pred)
    for x in range(len(pred)):
        print("Predicted: ", names[pred[x]], "Data: ", X_test[x], "Actual: ", names[y_test[x]])

        if(pred[x]==y_test[x]):
            disease["true"]=disease["true"]+1
        else:
            disease["false"] = disease["false"] + 1
    print(disease)
    return accuracy*100





