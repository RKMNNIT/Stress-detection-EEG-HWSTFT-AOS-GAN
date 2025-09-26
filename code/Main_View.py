import os, time, shutil, warnings
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import mne
import scipy.io
from plyer import notification

from Preprocessing import EEGSignal_Preprocessing
from EEGSegmentation import FA_FCN
import FeatureExtraction
from Feature_Selection import RLASSO
from Stress_Detection_and_Classification import AOS_GAN
from Metrics import PerformanceMetrics

warnings.filterwarnings("ignore")

def Dataset():
    print("\n=== DATASET SELECTION ===\n")
    notification.notify(message='Collect EEG signal from Wearable Sensors', app_name='EEG Stress Detection')
    file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat"), ("EDF files", "*.edf"), ("Text files", "*.txt")])
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return
    _, ext = os.path.splitext(file_path)
    if ext == '.edf':
        raw = mne.io.read_raw_edf(file_path)
        data = raw.get_data()
        scipy.io.savemat('Input_Signal.mat', {'o': data})
    elif ext == '.txt':
        data = np.loadtxt(file_path)
        scipy.io.savemat('Input_Signal.mat', {'o': data})
    elif ext == '.mat':
        shutil.copy(file_path, 'Input_Signal.mat')
    else:
        messagebox.showerror("Error", f"Unsupported file type: {ext}")
        return
    mat_data = scipy.io.loadmat('Input_Signal.mat')
    data = mat_data['o']
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(data[0, :500])
    ax.set_title("Raw EEG Signal (first 500 samples)")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()
    messagebox.showinfo('Dataset', 'EEG signal loaded successfully!')

def EEGPreprocessing():
    print("\n=== PREPROCESSING (Bandpass + Notch filter) ===\n")
    notification.notify(message='EEG Preprocessing', app_name='EEG Stress Detection')
    EEGSignal_Preprocessing()
    messagebox.showinfo('Preprocessing', 'Preprocessing completed!')

def Segmentation():
    print("\n=== SEGMENTATION (FA-FCN with Fuzzy Attention) ===\n")
    notification.notify(message='Segmentation with FA-FCN + Fuzzy Attention', app_name='EEG Stress Detection')
    FA_FCN()
    messagebox.showinfo('Segmentation', 'Segmentation with FA-FCN completed!')

def Feature_Extraction():
    print("\n=== FEATURE EXTRACTION (HW-STFT + Statistical + Fuzzy Features) ===\n")
    notification.notify(message='Feature Extraction (Hybrid)', app_name='EEG Stress Detection')
    FeatureExtraction.main()
    messagebox.showinfo('Feature Extraction', 'Hybrid Feature Extraction completed!')

def Select_the_Features():
    print("\n=== FEATURE SELECTION (RLASSO) ===\n")
    notification.notify(message='Recursive LASSO Feature Selection', app_name='EEG Stress Detection')
    RLASSO()
    messagebox.showinfo('Feature Selection', 'Recursive LASSO completed!')

def Stress_Detection_and_Classification():
    print("\n=== CLASSIFICATION (AOS-GAN) ===\n")
    notification.notify(message='Stress Classification with AOS-GAN', app_name='EEG Stress Detection')
    AOS_GAN()
    messagebox.showinfo('Classification', 'AOS-GAN classification completed!')

def Performancemetrics():
    print("\n=== PERFORMANCE METRICS (Real Computation) ===\n")
    notification.notify(message='Computing Performance Metrics', app_name='EEG Stress Detection')
    PerformanceMetrics()
    messagebox.showinfo('Performance Metrics', 'Metrics generated successfully!')

root = tk.Tk()
root.title("Wearable Sensor-Based EEG Stress Detection")
root.geometry("850x600")
root.configure(background="PeachPuff")

Label(root, text="Wearable Sensor-Based EEG Stress Detection", bg="purple", fg="white",
      width="80", height="2", font=('Georgia', 14, 'bold')).pack(pady=10)

buttons = [
    ("START: Load EEG Dataset", Dataset),
    ("PREPROCESSING (Bandpass + Notch)", EEGPreprocessing),
    ("SEGMENTATION (FA-FCN + Fuzzy Attention)", Segmentation),
    ("FEATURE EXTRACTION (HW-STFT + Stats + Fuzzy)", Feature_Extraction),
    ("FEATURE SELECTION (RLASSO)", Select_the_Features),
    ("STRESS DETECTION & CLASSIFICATION (AOS-GAN)", Stress_Detection_and_Classification),
    ("PERFORMANCE METRICS", Performancemetrics)
]

for text, cmd in buttons:
    Button(root, text=text, width=50, height=2, bg="MediumPurple", fg="white",
           font=('Times New Roman', 12), command=cmd).pack(pady=6)

root.mainloop()
