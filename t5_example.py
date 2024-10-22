from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, set_seed
from datasets import load_dataset
import torch
import sounddevice as sd
import logging  # Add logging for debugging
import numpy as np
import time
import threading  # {{ Added for handling threading }}
import queue       # {{ Added for inter-thread communication }}
import collections # {{ Added for buffering audio }}
import os
from dotenv import load_dotenv

load_dotenv()
class ListeningManager:
    def __init__(self, threshold=0.05, buffer_duration=2, sample_rate=16000, channels=1, silence_duration=1.0):
        self.threshold = threshold  # {{ Parameterized threshold }}
        self.audio_queue = queue.Queue()  # {{ Queue to handle detected speech }}
        self.buffer = collections.deque()  # {{ Initialize a buffer to store audio chunks }}
        self.buffer_duration = buffer_duration  # {{ Parameterized buffer duration }}
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_buffering = False
        self.last_speech_time = None
        self.silence_duration = silence_duration  # {{ Parameterized silence duration }}
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logging.warning(f"Sounddevice status: {status}")
        audio = indata.flatten()
        rms = np.sqrt(np.mean(audio**2))
        
        current_time = time.time()  # {{ Get current time }}

        if rms > self.threshold:
            if not self.is_buffering:
                logging.debug("Speech started, initiating buffering.")
                self.is_buffering = True
            self.buffer.append(audio.copy())
            self.last_speech_time = current_time  # {{ Update last_speech_time }}
        else:
            if self.is_buffering:
                if self.last_speech_time and (current_time - self.last_speech_time) > self.silence_duration:
                    logging.debug("Speech ended after 1 second of silence, processing buffered audio.")
                    self.is_buffering = False
                    full_audio = np.concatenate(list(self.buffer))
                    self.buffer.clear()
                    self.audio_queue.put(full_audio)

    def start_listening(self, sample_rate=16000, channels=1):
        self.stream = sd.InputStream(callback=self.audio_callback, samplerate=sample_rate, channels=channels)
        self.stream.start()
        logging.debug("Started audio stream for listening.")

class SpeechManager:
    def __init__(self, processor_model="microsoft/speecht5_vc", vocoder_model="microsoft/speecht5_hifigan", seed=555):
        self.processor = SpeechT5Processor.from_pretrained(processor_model)  # {{ Initialize processor once }}
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained(processor_model)  # {{ Initialize model once }}
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model)  # {{ Initialize vocoder once }}
        set_seed(seed)  # {{ Set seed for reproducibility }}
    
    def generate_speech(self, audio_inputs, sampling_rate=16000):
        inputs = self.processor(audio=audio_inputs, sampling_rate=sampling_rate, return_tensors="pt")  # {{ Process audio inputs }}
        print(f"Inputs shape: {inputs['input_values'].shape}")  # Debug statement
        speaker_embeddings = torch.zeros((1, 512))  # {{ Placeholder for speaker embeddings }}
        speech = self.model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=self.vocoder)
        return speech.squeeze().cpu().numpy()  # {{ Return the generated speech as a NumPy array }}

class SpeechToSpeechApplication:
    def __init__(self):
        self.listening_manager = ListeningManager()
        self.speech_manager = SpeechManager()
        self.listening_thread = threading.Thread(target=self.listen_and_respond)
        self.listening_thread.daemon = True  # {{ Allows thread to exit when main program exits }}
    
    def listen_and_respond(self):
        while True:
            try:
                audio = self.listening_manager.audio_queue.get(timeout=1)  # {{ Retrieve audio from queue }}
                logging.debug("Received audio from queue, generating speech.")
                generated_speech = self.speech_manager.generate_speech(audio)
                if generated_speech is not None:
                    sd.play(generated_speech, self.listening_manager.sample_rate)  # {{ Play generated speech }}
                    sd.wait()
                    logging.debug("Played the generated speech.")
            except queue.Empty:
                continue  # {{ No audio received, continue listening }}
    
    def run(self):
        self.listening_manager.start_listening()  # {{ Start the listening manager }}
        self.listening_thread.start()  # {{ Start the listening thread }}
        print("Speech-to-Speech Application is running. Press Ctrl+C to exit.")
        self.listening_thread.join()

# Example usage
if __name__ == "__main__":
    sample_rate = 16000
    s2s_app = SpeechToSpeechApplication()
    s2s_app.run()
