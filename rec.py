import pyaudio
import wave
import time
import sys
import os
import tkinter as tk
import threading

# Audio parameters
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Global variables for recording
recording = False
cycle_number = 1
step = "CW"  # Start with CW
 
audio_frames = []  # List to store audio data
p = None  # PyAudio instance

# Function to record audio
def record_audio():
    global p, audio_frames, recording
    if p is None:
        p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    while recording: 
        data = stream.read(CHUNK)
        audio_frames.append(data)
        
    stream.stop_stream()
    stream.close()
    print("record_audio stopped")

# Function to toggle recording state
def toggle_recording():
    global recording, cycle_number, audio_frames, step, record_button, status_label
    print("button clicked")  
    if not recording:
        print(f"Recording {step} started for cycle {cycle_number}...")
        audio_frames = []
        recording = True
        print (recording)
        record_button.config(text="Stop Recording")
        status_label.config(text=f"Recording {step} for cycle {cycle_number}")
        t = threading.Thread(target=record_audio)
        t .start()
       # record_audio() # Call the record_audio function here
        
    else:
        if len(audio_frames) > 0:
            print(f"Recording {step} stopped for cycle {cycle_number} and saved.")
            base_folder = "data"
            if step == "CW":
                folder_path = os.path.join(base_folder, "CW_Recordings")
            else:
                folder_path = os.path.join(base_folder, "CCW_Recordings")

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            filename = os.path.join(folder_path, f"{step}_{cycle_number}.wav")
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            
            #calculate the number to skip from beginning
            skip_frames_start = int(0.25 * RATE / CHUNK)
            skip_frames_end = int(0.25 * RATE / CHUNK)

           #write the audio frames to the wave file except the first skip_frame_start and the last_frames
                   
            wf.writeframes(b''.join(audio_frames[skip_frames_start:-skip_frames_end]))
            wf.close()

            if step == "CW":
                step = "CCW"
            else:
                step = "CW"
                cycle_number += 1
                
            audio_frames = []
            recording = False
            print(recording)
            record_button.config(text="Start Recording")
            status_label.config(text=f"Ready to start {step} recording for cycle {cycle_number}")
            print(f"Press the record button to start {step} recording for cycle {cycle_number}...")

# Main function
def main():
    global p, record_button, status_label

    # Create folders if they don't exist
    base_folder = "data"
    cw_folder = os.path.join(base_folder, "CW_Recordings")
    ccw_folder = os.path.join(base_folder, "CCW_Recordings")

    if not os.path.exists(cw_folder):
        os.makedirs(cw_folder)
    if not os.path.exists(ccw_folder):
        os.makedirs(ccw_folder)

    # Create GUI window and button
    window = tk.Tk()
    window.title("Audio Recorder")
    window.geometry("300x200")
    record_button = tk.Button(window, text="Start Recording", command=toggle_recording)
    record_button.pack(pady=50)
    status_label = tk.Label(window, text="Ready to start CW recording for cycle 1")
    status_label.pack()

    try:
        print("Press the record button to start CW recording for cycle 1...")
        window.mainloop()
        
    except KeyboardInterrupt:
        # Terminate PyAudio and close the window
        if p is not None:
            p.terminate()
        window.destroy()

if __name__ == "__main__":
    main()
