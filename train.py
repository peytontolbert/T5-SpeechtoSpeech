# t5.py
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, set_seed
from datasets import load_dataset
import torch
import logging
import numpy as np
import os
import sounddevice as sd
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
import librosa
import queue       # {{ Added for inter-thread communication }}
import collections # {{ Added for buffering audio }}
import time
import torch.nn.functional as F
from main import SpeechToSpeechApplication
from t5_s2s import T5SpeechManager
from speech_manager import SpeechManager
from llama_manager import LlamaManager
from whisper_manager import WhisperManager
from pydub import AudioSegment
import io
from pathlib import Path
from speechbrain.pretrained import EncoderClassifier



spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name, 
    run_opts={"device": device}, 
    savedir=os.path.join("/tmp", spk_model_name)
)

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

    def stop_listening(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logging.debug("Stopped audio stream for listening.")


def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu()  # {{ Removed .numpy() to keep as tensor }}
    return speaker_embeddings

def save_audio(audio_data, output_path):
    """
    Saves audio data to the specified file path in WAV format.
    Handles both bytes-like objects and file paths.
    """
    try:
        if isinstance(audio_data, (bytes, bytearray)):
            # Convert audio_data (assumed to be in MP3 format) to WAV
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            audio.export(output_path, format="wav")
            print(f"Audio data converted to WAV and saved to {output_path}")
        elif isinstance(audio_data, (str, Path)):
            # If audio_data is a file path, ensure it's in WAV format
            audio = AudioSegment.from_file(audio_data, format="wav")
            audio.export(output_path, format="wav")
            print(f"Audio file at {audio_data} saved as WAV to {output_path}")
        else:
            print("Unsupported audio_data type. Must be bytes or file path.")
    except Exception as e:
        print(f"Failed to convert and save audio: {e}")


def train_speech_t5_model(whisper_manager, llama_manager, speech_manager, student, learning_rate=1e-4):
    device = torch.device('cpu')
    conversation_history = []
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    model.to(device)
    model.train()
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    global_step = 0
    num_epochs = 10
    save_steps = 2

    # {{ Initialize and start ListeningManager }}
    listening_manager = ListeningManager()
    listening_manager.start_listening()

    for epoch in range(num_epochs):
        # {{ Retrieve audio from ListeningManager's queue }}
        try:
            audio_input = listening_manager.audio_queue.get(timeout=10)  # Wait for audio
        except queue.Empty:
            print("No audio received.")
            continue
        input_features = processor(audio=audio_input, sampling_rate=16000, return_tensors="pt")

        transcription = whisper_manager.transcribe_audio({
            "array": audio_input,
            "sampling_rate": whisper_manager.sample_rate
        })  # Updated to pass a dict with 'array' and 'sampling_rate'
        if transcription and transcription != "No speech detected.":
            print(f"Transcription: {transcription}")
            conversation_history.append({"role": "user", "content": transcription})  # Add to history
        text_inputs = llama_manager.generate_response(conversation_history, system_prompt="You are a helpful assistant.")
        conversation_history.append({"role": "assistant", "content": text_inputs})  # Update history with AI response
        target_audio_path, response = speech_manager.text_to_speech(text_inputs)

        # Load and process the synthesized audio
        target_audio, _ = librosa.load("speech.wav", sr=16000)
        target_features = processor(audio_target=target_audio, sampling_rate=16000, return_tensors="pt")
        if audio_input is None or len(audio_input) == 0:
            print("Error: No audio input received.")
        speaker_embeddings = torch.zeros((1, 512))  # or load xvectors from a file
        # speaker_embeddings = create_speaker_embedding(target_audio).to(device)  # {{ Ensure tensor is on the correct device }}
        target_values = target_features['input_values']
        print(f"Original target_values shape: {target_values.shape}")  # Debug statement
        input_values = input_features['input_values']
        print(f"Original input_feature_values shape: {input_values.shape}")  # Debug statement
        outputs = model(
            input_values=input_values,  # {{ Changed to use keyword argument }}
            speaker_embeddings=speaker_embeddings,   # {{ Changed to use keyword argument }}
            labels=target_values,
            return_dict=True  # Ensure this is set
        )
        
        # {{ Updated loss computation to handle shape mismatch }}
        if outputs.loss is None:
            loss_fct = torch.nn.MSELoss()

            pred_spectrogram = outputs.spectrogram  # Model's predicted spectrogram
            target_spectrogram = target_values  # Ground truth spectrogram

            # Align the spectrogram dimensions by trimming the longer tensor
            pred_length = pred_spectrogram.size(1)
            target_length = target_spectrogram.size(1)

            if pred_length > target_length:
                pred_spectrogram = pred_spectrogram[:, :target_length, :]
                print(f"Trimmed pred_spectrogram to shape: {pred_spectrogram.shape}")
            elif pred_length < target_length:
                target_spectrogram = target_spectrogram[:, :pred_length, :]
                print(f"Trimmed target_spectrogram to shape: {target_spectrogram.shape}")

            loss = loss_fct(pred_spectrogram, target_spectrogram)
        else:
            loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1

        if global_step % 10 == 0:
            print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item()}")

        # Save model checkpoint periodically
        if global_step % save_steps == 0:
            output_dir = f'checkpoint-{global_step}'
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"Checkpoint saved at {output_dir}")

    # {{ Stop Listening after training }}
    listening_manager.stop_listening()

    # Save final model
    output_dir = 'final_model'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Training completed. Final model saved at {output_dir}")


# Example usage
if __name__ == "__main__":
    dataset_folder = "dataset"
    api_key = os.getenv("OPENAI_API_KEY")
    whisper_manager = WhisperManager()
    llama_manager = LlamaManager(api_key)
    speech_manager = SpeechManager(api_key)
    student = T5SpeechManager()
    train_speech_t5_model(whisper_manager, llama_manager, speech_manager, student)


