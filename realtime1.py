import numpy as np
import pyaudio
import joblib
import tkinter as tk
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import wave
import os
# import noisereduce as nr

# Load the trained model
model = joblib.load('C:/Users/ITPKL/Desktop/pydev/cek.pkl')

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Create a stream to capture audio from the microphone
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

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

# Define the labels
label_dict = {1: 'OK', 0: 'Not OK'}


# Function to read audio, transform to FFT, and process with the ML model
def analyze_audio():
    audio_data, sample_rate = sf.read('recorded_audio.wav')

    # Apply noise reduction
    # reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noise_data)
    
    if sample_rate != 44100:
        audio_data = sf.resample(audio_data, 44100)
    
    # Mengubah audio menjadi mono jika perlu
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalisasi data audio
    audio_data = audio_data / np.max(np.abs(audio_data))
    # ======================================================
    # audio_data = np.frombuffer(stream.read(RATE*5), dtype=np.int16)  # Record for 1 second
    # fft_data = np.abs(np.fft.fft(audio_data))[:RATE // 2]
    # ===========================================
    # Memisahkan audio menjadi chunk-chunk kecil
    chunk_size = 2048  # Increase chunk_size to match the expected number of features
    num_chunks = len(audio_data) // chunk_size
    chunks = np.array_split(audio_data[:num_chunks * chunk_size], num_chunks)
    
    # Memproses setiap chunk dan membuat prediksi
    for chunk in chunks:
        fft_data = np.abs(np.fft.fft(chunk))[:chunk_size // 2]
        
        # Mengubah format data FFT sesuai dengan model yang dilatih
        input_data = fft_data.reshape(1, -1)  # Karena model membutuhkan input 2D
        
        # Menggunakan model untuk membuat prediksi
        prediction = model.predict(input_data)

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
    
    # Reshape the FFT data for prediction (adjust the shape according to your model)
    # input_data = fft_data[:1024].reshape(1, -1)  # Use only the first 1024 features
    # print (input_data)
    # # Use the loaded model for prediction
    # prediction = model.predict(input_data)
    # print(prediction)
    # print("Model prediction:", label_dict[prediction[0]])
    NG(prediction)
    OK(prediction)
    

# Function untuk memulai proses rekaman
def start_recording():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 4
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

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Memproses audio yang direkam
    analyze_audio()

    # Menampilkan visualisasi gelombang audio
    # audio_data, sample_rate = sf.read('recorded_audio.wav')
    # plt.figure(figsize=(10, 4))
    # plt.plot(audio_data)
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.title('Waveform of Recorded Audio')
    # plt.show()
    os.remove('recorded_audio.wav')

# Create the UI
record_button = tk.Button(window, text="Start Recording", command=start_recording, height=3, width=20, bg="white")

record_button.pack()

window.mainloop()