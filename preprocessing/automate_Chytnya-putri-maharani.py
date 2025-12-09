import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

def preprocess():
    print("=== Loading raw dataset ===")
    df = pd.read_csv("predictive_maintenance_raw.csv")

    # Hapus kolom tidak dipakai
    df = df.drop(columns=['UDI', 'Product ID'], errors='ignore')

    # Handle missing values (hapus saja)
    df = df.dropna()

    # Deteksi & clipping outlier
    numeric_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] > upper, upper, df[col])
        df[col] = np.where(df[col] < lower, lower, df[col])

    # Normalisasi numeric
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encoding kolom kategori
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Split X–y
    X = df.drop(columns=['Target', 'Failure Type'])
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("--- Saving output ---")
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/processed_data.csv", index=False)
    X_train.to_csv("output/X_train.csv", index=False)
    X_test.to_csv("output/X_test.csv", index=False)
    y_train.to_csv("output/y_train.csv", index=False)
    y_test.to_csv("output/y_test.csv", index=False)

    print("Preprocessing selesai. File tersedia di folder output/")

if __name__ == "__main__":
    preprocess()
