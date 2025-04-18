import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("heart_disease.csv")

print("\n✅ Data Loaded Successfully")
print(df.head())
print("\nData Info:")
print(df.info())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'heart_disease_detector.pkl')
print("\n✅ Model saved as 'heart_disease_detector.pkl'")

print("\n🔍 Enter new patient details to predict heart disease:")
age = int(input("Age: "))
sex = int(input("Sex (0 = female, 1 = male): "))
chest_pain_type = int(input("Chest Pain Type (1–4): "))
resting_bp_s = int(input("Resting Blood Pressure: "))
cholesterol = int(input("Cholesterol: "))
fasting_blood_sugar = int(input("Fasting Blood Sugar (0 or 1): "))
resting_ecg = int(input("Resting ECG (0, 1, 2): "))
max_heart_rate = int(input("Max Heart Rate: "))
exercise_angina = int(input("Exercise Angina (0 = No, 1 = Yes): "))
oldpeak = float(input("Oldpeak (e.g., 1.5): "))
st_slope = int(input("ST Slope (1–3): "))

sample = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'chest pain type': chest_pain_type,
    'resting bp s': resting_bp_s,
    'cholesterol': cholesterol,
    'fasting blood sugar': fasting_blood_sugar,
    'resting ecg': resting_ecg,
    'max heart rate': max_heart_rate,
    'exercise angina': exercise_angina,
    'oldpeak': oldpeak,
    'ST slope': st_slope
}])

model = joblib.load('heart_disease_detector.pkl')
prediction = model.predict(sample)

print("\n🩺 Prediction Result:")
if prediction[0] == 1:
    print("✅ Heart Disease Detected")
else:
    print("❌ No Heart Disease Detected")












