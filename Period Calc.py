import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


def calculate_fundamental_period_and_plot(csv_file_path):
    
    style = 'seaborn'
    # Load data from CSV
    data = pd.read_csv(csv_file_path)
    
    # Assuming the first column is 'time' and the second is 'acceleration'
    time = data.iloc[:, 0].values
    acceleration = data.iloc[:, -2].values

    # Calculate time step (dt) and sampling frequency (fs)
    dt = time[1] - time[0]  # Assuming uniform time intervals
    fs = 1 / dt

    # Perform the Fourier Transform
    n = len(acceleration)
    fft_result = np.fft.fft(acceleration)
    frequencies = np.fft.fftfreq(n, dt)  # Frequency bins

    # Magnitude of FFT results
    magnitude = np.abs(fft_result)

    # Focus on positive frequencies
    positive_frequencies = frequencies[:n // 2]
    positive_magnitude = magnitude[:n // 2]

    # Find the fundamental frequency (highest peak)
    peak_index = np.argmax(positive_magnitude)
    fundamental_frequency = positive_frequencies[peak_index]
    fundamental_period = 1 / fundamental_frequency
    peak_amplitude = positive_magnitude[peak_index]
    
    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.style.use(style)
    plt.plot(time, acceleration, label="Time series")
    plt.title("Acceleration mesurement")
    plt.xlabel("Time (s)")
    plt.ylabel("Acc (m/s2)")
    

    # Plot the frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.style.use(style)
    plt.plot(positive_frequencies, positive_magnitude, label="Frequency Spectrum")
    plt.scatter(fundamental_frequency, peak_amplitude, color='red', label="Fundamental Frequency")
    plt.axvline(fundamental_frequency, color='green', linestyle='--', label=f"f = {fundamental_frequency:.2f} Hz")
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()
    plt.annotate(f"f = {fundamental_frequency:.2f} Hz\nT = {fundamental_period:.2f} s", 
                 xy=(fundamental_frequency, peak_amplitude), 
                 xytext=(fundamental_frequency + 0.5, peak_amplitude),
                 arrowprops=dict(facecolor='black', arrowstyle="->"))
    plt.show()

    return fundamental_frequency, fundamental_period

def calculate_stiffness(effective_mass, fundamental_period):
    """
    Calculate the stiffness of the system based on effective mass and fundamental period.
    
    Parameters:
        effective_mass (float): Effective mass of the system (kg)
        fundamental_period (float): Fundamental period of the system (seconds)
    
    Returns:
        float: Stiffness of the system (N/m)
    """
    # Calculate angular frequency
    omega = 2 * np.pi / fundamental_period  # ω = 2π/T

    # Calculate stiffness
    stiffness = (omega ** 2) * effective_mass  # k = ω² * m_eff

    return stiffness


csv_file_path = Path(r'C:\Users\dsuas\Desktop\Fundamental Period\acc floor.csv') 
fundamental_frequency, fundamental_period = calculate_fundamental_period_and_plot(csv_file_path)
print(f"Fundamental Frequency: {fundamental_frequency:.4f} Hz")
print(f"Fundamental Period: {fundamental_period:.4f} seconds")

stiffness = calculate_stiffness(1000, fundamental_period)
print(f"Stiffness: {stiffness:.4f} kg/m")


