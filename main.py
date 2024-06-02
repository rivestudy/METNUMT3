import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import sqrt

# Membaca data dari file CSV
data = pd.read_csv('D:\Database\CODES\PAITON\METNUM\METNUMT3\Student_Performance (1).csv')

# Mengambil kolom yang dibutuhkan
TB = data['Hours Studied'].values
NL = data['Sample Question Papers Practiced'].values
NT = data['Performance Index'].values

# Fungsi untuk regresi eksponensial
def exponential_model(x, a, b):
    return a * np.exp(b * x)

# Fungsi untuk regresi linear
def linear_model(x, m, c):
    return m * x + c

# Regresi Linear untuk TB terhadap NT
params_lin_tb, _ = curve_fit(linear_model, TB, NT)
m_tb, c_tb = params_lin_tb

# Regresi Eksponensial untuk TB terhadap NT
params_exp_tb, _ = curve_fit(exponential_model, TB, NT)

# Prediksi menggunakan model linear dan eksponensial
NT_pred_lin_tb = linear_model(TB, m_tb, c_tb)
NT_pred_exp_tb = exponential_model(TB, *params_exp_tb)

# Plot untuk TB terhadap NT
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(TB, NT, color='blue', label='Data Points')
plt.plot(TB, NT_pred_lin_tb, color='red', label='Linear Regression')
plt.plot(TB, NT_pred_exp_tb, color='green', label='Exponential Regression')
plt.xlabel('Hours Studied (TB)')
plt.ylabel('Performance Index (NT)')
plt.legend()
plt.title('TB vs NT')

# Menghitung galat RMS
rms_linear_tb = np.sqrt(np.mean((NT - NT_pred_lin_tb) ** 2))
rms_exponential_tb = np.sqrt(np.mean((NT - NT_pred_exp_tb) ** 2))

print(f'RMS Error for Linear Regression (TB vs NT): {rms_linear_tb}')
print(f'RMS Error for Exponential Regression (TB vs NT): {rms_exponential_tb}')

# Regresi Linear untuk NL terhadap NT
params_lin_nl, _ = curve_fit(linear_model, NL, NT)
m_nl, c_nl = params_lin_nl

# Regresi Eksponensial untuk NL terhadap NT
params_exp_nl, _ = curve_fit(exponential_model, NL, NT)

# Prediksi menggunakan model linear dan eksponensial
NT_pred_lin_nl = linear_model(NL, m_nl, c_nl)
NT_pred_exp_nl = exponential_model(NL, *params_exp_nl)

# Plot untuk NL terhadap NT
plt.subplot(1, 2, 2)
plt.scatter(NL, NT, color='blue', label='Data Points')
plt.plot(NL, NT_pred_lin_nl, color='red', label='Linear Regression')
plt.plot(NL, NT_pred_exp_nl, color='green', label='Exponential Regression')
plt.xlabel('Sample Question Papers Practiced (NL)')
plt.ylabel('Performance Index (NT)')
plt.legend()
plt.title('NL vs NT')

# Menghitung galat RMS
rms_linear_nl = np.sqrt(np.mean((NT - NT_pred_lin_nl) ** 2))
rms_exponential_nl = np.sqrt(np.mean((NT - NT_pred_exp_nl) ** 2))

print(f'RMS Error for Linear Regression (NL vs NT): {rms_linear_nl}')
print(f'RMS Error for Exponential Regression (NL vs NT): {rms_exponential_nl}')

plt.tight_layout()
plt.show()
