from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd

def createMLDataframe(df: pd.DataFrame, hour: int) -> pd.DataFrame:
    """
    Create a new database with the necessary columns required for machine learning.
    """
    type_one_hot = OneHotEncoder()
    scaler = StandardScaler()

    hours_encoded = type_one_hot.fit_transform(df['hour'].values.reshape(-1,1)).toarray()
    hours_df = pd.DataFrame(hours_encoded, columns = ["hour_"+str(int(i)) for i in range(hours_encoded.shape[1])])

    weekdays_encoded = type_one_hot.fit_transform(df['weekday'].values.reshape(-1,1)).toarray()
    weekdays_df = pd.DataFrame(weekdays_encoded, columns = ["weekday_"+str(int(i)) for i in range(weekdays_encoded.shape[1])])

    ml_df = pd.concat([df, hours_df, weekdays_df],axis=1)
    ml_df = ml_df.drop(['hour'], axis=1)
    ml_df[['']] = scaler.fit_transform(ml_df[['']])

    return ml_df

def linearSVC(ml_df: pd.DataFrame, new_df: pd.DataFrame, category: str) -> float:
    """
    Use the LinearSVC Classifier to get the probability of a shopping trip going past a shelf.
    """
    X = ml_df.drop([category],axis=1).values
    Y = ml_df[category].values
    new_X = new_df.drop([category],axis=1).values
    new_Y = new_df[category].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = Y, test_size = 0.25)

    linear_svc = LinearSVC(C=4.1, penalty="l2")
    linear_svc.fit(X_train, Y_train)
    clf = CalibratedClassifierCV(linear_svc, cv="prefit")
    clf.fit(X_test, Y_test,)
    y_proba = clf.predict_proba(new_X)

    return y_proba
