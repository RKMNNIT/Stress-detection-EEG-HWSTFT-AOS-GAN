import os
import time
from tkinter import *
import time
import pydub
import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import glob
import mne
from tkinter import *
from plyer import notification
from Preprocessing import *
from EEGSegmentation import *
import FeatureExtraction
from Feature_Selection import *
from Stress_Detection_and_Classification import *
from Metrics import *
from tkinter import Tk, Label, Button
from tkinter import filedialog, messagebox
import scipy.io
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox
import shutil
from matplotlib.figure import Figure
from plyer import notification  
import warnings
warnings.filterwarnings("ignore")
def Dataset():
    print ("\t\t\t |--------- ****** Wearable Sensor-Based Mental Stress Detection ****** --------|")
    time.sleep(1)
    print('=================================================================================================================')
    print ("\t\t\t ****** COLLECT THE EEG SIGNAL FROM WEARABLE SENSORS FOR STREES DETECTION ******")
    print('=================================================================================================================')
    notification.notify(
            message='COLLECT THE EEG SIGNAL FROM WEARABLE SENSORS FOR STREES DETECTION',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    def load_mat_file():
        file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat"), ("EDF files", "*.edf"), ("TEXT files", "*.txt")])
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
        try:
            mat_data = scipy.io.loadmat('Input_Signal.mat')
            data = mat_data['o'] 
            display_waveform(data)
            messagebox.showinfo('Dataset','Collect the EEG signal from Wearable Sensors for Stress Detection successfully completed!')
            print('\nCollect the EEG signal from Wearable Sensors for Stress Detection successfully completed!\n')
            print('\nNext Click PREPROCESSING button...\n')
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    def display_waveform(data):
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot(data)
        plt.title("Waveform")
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(padx=20, pady=20)
    root = tk.Tk()
    root.title("Display Waveform from MAT File")
    load_button = tk.Button(root, text="Load .mat,.edf or .txt Files", command=load_mat_file)
    load_button.pack(pady=20)
    root.mainloop()
def EEGPreprocessing():
    time.sleep(1)
    print('=================================================================================================================')
    print ("\t\t\t ****** PREPROCESSING ******")
    print('=================================================================================================================')
    notification.notify(
            message='PREPROCESSING',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    EEGSignal_Preprocessing();
    messagebox.showinfo('PREPROCESSING','PREPROCESSING successfully completed!')
    print('\nPREPROCESSING is successfully completed!\n')
    print('\nNext Click SEGMENT THE EEG SIGNALS button...\n')
def Segmentation():
    time.sleep(1)
    print('=================================================================================================================')
    print ("\t\t\t ****** SEGMENT THE EEG SIGNALS ******")
    print('=================================================================================================================')
    notification.notify(
            message='SEGMENT THE EEG SIGNALS',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    FA_FCN();
    time.sleep(1)
    messagebox.showinfo('SEGMENT THE EEG SIGNALS','SEGMENT THE EEG SIGNALS process is Completed!')
    print('\nSEGMENT THE EEG SIGNALS is successfully completed!\n')
    print('\nNext Click FEATURE EXTRACTION button...\n')
def Feature_Extraction():
    time.sleep(1)
    print('=================================================================================================================')
    print ("\t\t\t ****** FEATURE EXTRACTION ******")
    print('=================================================================================================================\n')
    notification.notify(
            message='FEATURE EXTRACTION',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    FeatureExtraction.main();
    time.sleep(1)
    messagebox.showinfo('FEATURE EXTRACTION','FEATURE EXTRACTION process is Completed!')
    print('\nFEATURE EXTRACTION is successfully completed!\n')
    print('\nNext Click SELECT THE FEATURES button...\n')
def Select_the_Features():
    time.sleep(1)
    print('=================================================================================================================')
    print ("\t\t\t ****** SELECT THE FEATURES ******")
    print('=================================================================================================================\n')
    notification.notify(
            message='SELECT THE FEATURES',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    RLASSO();
    time.sleep(1)
    messagebox.showinfo('SELECT THE FEATURES','SELECT THE FEATURES process is Completed!')
    print('\nSELECT THE FEATURES process is successfully completed!\n')
    print('\nNext Click STRESS DETECTION AND CLASSIFICATION button...\n')
def Stress_Detection_and_Classification():
    time.sleep(1)
    print('=================================================================================================================')
    print ("\t\t\t ****** STRESS DETECTION AND CLASSIFICATION ******")
    print('=================================================================================================================\n')
    notification.notify(
            message='STRESS DETECTION AND CLASSIFICATION',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    AOS_GAN();
    messagebox.showinfo('STRESS DETECTION AND CLASSIFICATION','STRESS DETECTION AND CLASSIFICATION process is Completed!')
    print('\nSTRESS DETECTION AND CLASSIFICATION process is successfully completed!\n')
    print('\nNext Click PERFORMANCE METRICS button...\n')
def Performancemetrics():
    time.sleep(1)
    print('=================================================================================================================')
    print ("\t\t\t ****** PERFORMANCE METRICS ******")
    print('=================================================================================================================')
    print('\nGraph generation process is starting\n')
    notification.notify(
            message='PERFORMANCE METRICS',
            app_name='My Python App',
            app_icon=None,
        )
    time.sleep(1)
    PerformanceMetrics();
    print('\nGraph is Generated Successfully...!')
    print('=================================================================================================================')
    print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
def main_screen():
    window = Tk()
    window.title("WEARABLE SENSOR-BASED MENTAL STRESS DETECTION")
    window_width = 800
    window_height = 600
    window.geometry(f"{window_width}x{window_height}")
    window.configure(background="PeachPuff")
    label_bg_color = "MediumPurple"
    button_bg_color = "purple"
    button_fg_color = "white"
    Label(window, text="Wearable Sensor-Based Mental Stress Detection", bg=label_bg_color, fg="white", width="500", height="2", font=('Georgia', 12)).pack()
    Label(text = "",bg="PeachPuff", fg="White").pack(pady=10)    
    b1 = Button(text="START", height="2", width="25", bg=button_bg_color, fg=button_fg_color, font=('Times New Roman', 12), command=Dataset)
    b1.pack(pady=10)
    b2 = Button(text="PREPROCESSING", height="2", width="25", bg=button_bg_color, fg=button_fg_color, font=('Times New Roman', 12), command=EEGPreprocessing)
    b2.pack(pady=10)
    b3 = Button(text="SEGMENT THE EEG SIGNALS", height="2", width="25", bg=button_bg_color, fg=button_fg_color, font=('Times New Roman', 12), command=Segmentation)
    b3.pack(pady=10)
    b4 = Button(text="FEATURE EXTRACTION", height="2", width="25", bg=button_bg_color, fg=button_fg_color, font=('Times New Roman', 12), command=Feature_Extraction)
    b4.pack(pady=10)
    b5 = Button(text="SELECT THE FEATURES", height="2", width="25", bg=button_bg_color, fg=button_fg_color, font=('Times New Roman', 12), command=Select_the_Features)
    b5.pack(pady=10)
    b6 = Button(text="STRESS DETECTION\nAND CLASSIFICATION", height="2", width="25", bg=button_bg_color, fg=button_fg_color, font=('Times New Roman', 12), command=Stress_Detection_and_Classification)
    b6.pack(pady=10)
    b7 = Button(text="PERFORMANCE\nMETRICS", height="2", width="25", bg=button_bg_color, fg=button_fg_color, font=('Times New Roman', 12), command=Performancemetrics)
    b7.pack(pady=10)
    window.mainloop()
main_screen()
