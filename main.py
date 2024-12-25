import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv('cars_dataset.csv')



encoder = LabelEncoder()

df['safety'] = encoder.fit_transform(df['safety'])

df['maint'] = encoder.fit_transform(df['maint']) 
df['buying'] = encoder.fit_transform(df['buying'])
df['lug_boot'] = encoder.fit_transform(df['lug_boot'])
df['doors'] = encoder.fit_transform(df['doors'])
df['persons'] = encoder.fit_transform(df['persons'])
df['car'] = encoder.fit_transform(df['car'])


x = df.drop(columns=['car'])
y = df['car']


# --------------------------------------------------------------------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# --------------------------------------------------------------------------------------------------------------------------------------------

model = SVC(kernel='poly',C=0.1, random_state = 42)  # kernal('linear=0.69', 'precomputed', 'sigmoid=0.62', 'poly=0.76', 'rbf=0.69)

model.fit(x_train, y_train)



# --------------------------------------------------------------------------------------------------------------------------------------------

predictions = model.predict(x_test)

# -------------------------------------------------------------------------------------------------------------------------------------------

accuracy_score = accuracy_score(y_test, predictions)

classification_results = classification_report(y_test, predictions)


print(f'Accuracy: {accuracy_score}')
print(f'Classification Report: {classification_results}')


