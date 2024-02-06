import numpy as np
import pyaudio
import joblib
import tkinter as tk
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import wave
import pandas as pd
from scipy.io import wavfile
import os
import librosa

# os.remove('recorded_audio.wav')
# Create a new tkinter window
window = tk.Tk()
window.title("Real-time Audio Judgement")
window.configure(background='white')
def NG(prediction):
    if prediction == 0:
        NG_button.configure(bg='red')
        
    else:
        NG_button.configure(bg='white')
        
def OK(prediction):
    if prediction == 1:
        OK_button.configure(bg='green')
        
    else:
        OK_button.configure(bg='white')
        
canvas2 = tk.Canvas(window, width=200, height=200,background='white')
canvas2.pack(side='top', padx=10, pady=10)

NG_button = tk.Button(canvas2, text="NG", command=NG, height=3, width=20, state='disabled', font=('TkDefaultFont', 10, 'bold'), disabledforeground='black')
NG_button.pack(side='right', padx=10, pady=10)
OK_button = tk.Button(canvas2, text="OK", command=OK, height=3, width=20, state='disabled', font=('TkDefaultFont', 10, 'bold'), disabledforeground='black')
OK_button.pack(side='left', padx=10, pady=10)

# Create a new figure and two subplots, one for the audio data and one for the FFT data
fig = Figure(figsize=(14, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
tit = plt

# Add the figure to the tkinter window
canvas = FigureCanvasTkAgg(fig, master=window)

canvas.draw()
canvas.get_tk_widget().pack()

# Define a function to read WAV files and convert to FFT
def read_wav_file(file_path, fft_length=1024):
    try:
        sample_rate, audio = wavfile.read(file_path)
        if len(audio) == 0:
            return None
        return np.fft.fft(audio)[:fft_length]
    except Exception as e:
        print(f"Error reading WAV file: {file_path}")
        print(f"Error message: {str(e)}")
        return None

def dframe(fmd, pm):
    # Create a DataFrame with the magnitude of the FFT outputs and numerical labels
    fft_magnitude_df = pd.DataFrame(fmd)
    mask_df = pd.DataFrame(pm)    

    return fft_magnitude_df, mask_df

    

# Process a single WAV file
def proses_file_audio(file_path, fft_length=1024):
    fft_magnitude_data = []
    masks = []
    
    # masuk fungsi baca file untuk mengembalikan dalam bentuk data fft 
    fft = read_wav_file(file_path, fft_length)
    if fft is not None:
        fft_magnitude = np.abs(fft)
        mask = np.ones_like(fft_magnitude)
        mask[len(fft_magnitude):] = 0
        fft_magnitude_padded = np.pad(fft_magnitude, (0, fft_length - len(fft_magnitude)))
        mask_padded = np.pad(mask, (0, fft_length - len(mask)))
        
        # Append the padded FFT magnitude and mask to their respective lists
        fft_magnitude_data.append(fft_magnitude_padded)
        masks.append(mask_padded)

    # Initialize an empty list to store the padded masks
    padded_masks = [np.pad(mask, (0, fft_length - len(mask))) for mask in masks]

    # Find the maximum length among all padded masks
    max_length = max(len(mask) for mask in padded_masks)

    # Pad each mask to the maximum length
    padded_masks = [np.pad(mask, (0, max_length - len(mask))) for mask in padded_masks]

    # Convert the lists to numpy arrays
    fft_magnitude_data = np.array(fft_magnitude_data)
    padded_masks = np.array(padded_masks)

    # Print the shapes of the resulting arrays
    print("FFT Magnitude Data Shape:", fft_magnitude_data.shape)
    print("Masks Shape:", padded_masks.shape)

    
    hasil_magnitude , hasil_padded = dframe(fft_magnitude_data, padded_masks)
    
    # belum terbentuk numpy array dan masih dalam bentuk dataframe
    print('===================================================================')
   

    # import file csv
    X_train = pd.read_csv('exported/X_train.csv') 
    y_train =pd.read_csv('exported/y_train.csv')
    X_coba = pd.read_csv('exported/X_coba.csv')
    y_coba = pd.read_csv('exported/y_coba.csv') # ubah disini
    X_train_resampled = pd.read_csv('exported/X_train_resampled.csv')
    y_train_resampled = pd.read_csv('exported/y_train_resampled.csv')

    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()

    X_train_resampled = sc_X.fit_transform(X_train_resampled)
    X_coba = sc_X.transform(X_coba)
    scaled_input = sc_X.transform(hasil_magnitude)



    # Load the model
    model = joblib.load('C:/Users/ITPKL/Desktop/production/model20240130.pkl')

    # Make predictions
    y_pred = model.predict(scaled_input) # ganti disini untuk melihat test


    audio_data, sample_rate = sf.read('recorded_audio.wav')
    
    # resample if necessary
    if sample_rate != 44100:
        audio_data = librosa.resample(audio_data, 44100)

    # convert to mono if necessary
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # normalize the audio data
    audio_data = audio_data / np.max(np.abs(audio_data))

    # split the audio into chunks
    chunk_size = 1024 # Increase chunk_size to match the expected number of features
    num_chunks = len(audio_data) // chunk_size
    chunks = np.array_split(audio_data[:num_chunks * chunk_size], num_chunks)

    for chunk in chunks:
        fft_data = np.abs(np.fft.fft(chunk))[:chunk_size //2 ]
    
    
    # Plot the audio data
    ax1.clear()
    ax1.plot(audio_data)
    ax1.set_title('Audio data')
    
    # Plot the FFT data
    ax2.clear()
    ax2.plot(fft_data)
    ax2.set_title('FFT data')
    
    # Update the plots
    canvas.draw()

    if y_pred == 0:
        print("Prediksi model: NG")
    else:
        print("Prediksi model: OK")
    
    NG(y_pred)
    OK(y_pred)  

# Function untuk memulai proses rekaman
def start_recording():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "recorded_audio.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)



    print("* Recording started *")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Recording finished *")

    print(stream)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')   
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(p.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Panggil fungsi proses_file_audio dengan data audio yang baru direkam
    proses_file_audio(audio_data)

    # audio yang di rekam masuk kedalam proses keputusan
    # proses_file_audio(WAVE_OUTPUT_FILENAME)  

# Create the UI
record_button = tk.Button(window, text="Start Recording", command=start_recording, height=3, width=20, bg="white")

record_button.pack()

window.mainloop()
