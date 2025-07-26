import pandas as pd
import numpy as np
from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# === Load and Prepare Data ===
dataset = pd.read_csv("C:\\Users\\soumo\\OneDrive\\Documents\\PYTHON\\ML-40\\Datasets\\National_Stock_Exchange_of_India_Ltd.csv")
dataset = dataset.drop(["LTP", "Chng", "% Chng", "52w H", "52w L", "365 d % chng", "30 d % chng"], axis=1)

encoder = OrdinalEncoder()
dataset['Symbol'] = encoder.fit_transform(dataset[['Symbol']])

features = dataset.drop("Symbol", axis=1)
labels = dataset["Symbol"]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

x_train, x_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.4, random_state=5)

model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)

model_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, model_predict)

# === GUI Setup ===
root = Tk()
root.title("Stock Company Predictor")
root.geometry("520x500")
root.configure(bg="#1e1e2f")

style_font = ("Segoe UI", 12)
label_font = ("Segoe UI", 11)
title_font = ("Segoe UI", 16, "bold")
entry_bg = "#2d2d44"
entry_fg = "white"

Label(root, text="Stock Company Predictor", font=title_font, bg="#1e1e2f", fg="white").pack(pady=15)

form_frame = Frame(root, bg="#1e1e2f")
form_frame.pack(pady=5)

def create_input_row(label_text):
    row = Frame(form_frame, bg="#1e1e2f")
    Label(row, text=label_text, font=label_font, width=14, anchor='w', bg="#1e1e2f", fg="white").pack(side=LEFT)
    entry = Entry(row, font=style_font, bg=entry_bg, fg=entry_fg, insertbackground="white", relief=FLAT, width=25)
    entry.pack(side=RIGHT, expand=True, fill=X, padx=10)
    row.pack(fill=X, pady=6, padx=20)
    return entry

open_entry = create_input_row("Open")
high_entry = create_input_row("High")
low_entry = create_input_row("Low")
qty_entry = create_input_row("Qty")
turnover_entry = create_input_row("Turnover")

result_label = Label(root, text="", font=label_font, bg="#1e1e2f", fg="lightgreen")
result_label.pack(pady=15)

accuracy_label = Label(root, text=f"Model Accuracy: {accuracy*100:.2f}%", font=label_font, bg="#1e1e2f", fg="#ffb703")
accuracy_label.pack()

# === Predict Company ===
def predict_company():
    try:
        open_val = float(open_entry.get())
        high_val = float(high_entry.get())
        low_val = float(low_entry.get())
        qty_val = float(qty_entry.get())
        turnover_val = float(turnover_entry.get())

        input_data = np.array([[open_val, high_val, low_val, qty_val, turnover_val]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        company = encoder.inverse_transform([prediction])[0][0]

        result_label.config(text=f"Predicted Company: {company}", fg="lightgreen")
    except Exception as e:
        result_label.config(text=f"Error: {e}", fg="red")

# === Predict Button ===
def on_enter(e): e.widget['background'] = "#00b894"
def on_leave(e): e.widget['background'] = "#0984e3"

predict_btn = Button(root, text="Predict Company", command=predict_company,
                     font=style_font, bg="#0984e3", fg="white", padx=10, pady=8, relief=FLAT, cursor="hand2")
predict_btn.pack(pady=20)
predict_btn.bind("<Enter>", on_enter)
predict_btn.bind("<Leave>", on_leave)

Label(root, text="Designed by Soumo", font=("Segoe UI", 10), bg="#1e1e2f", fg="gray").pack(side=BOTTOM, pady=10)

root.mainloop()
