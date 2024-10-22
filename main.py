from llama_manager import LlamaManager
from whisper_manager import WhisperManager
from speech_manager import SpeechManager
import os
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import wave
import threading
import logging
import io
from pydub import AudioSegment
import pygame
from pathlib import Path

load_dotenv()


class SpeechToSpeechApplication:
    def __init__(self, api_key):
        self.llama_manager = LlamaManager(api_key)
        self.whisper_manager = WhisperManager()
        self.speech_manager = SpeechManager(api_key)
        self.listening_thread = threading.Thread(target=self.listen_and_respond)
        self.listening_thread.daemon = True  # Allows thread to exit when main program exits
        self.conversation_history = []  # Initialize conversation history
        self.system_prompt = "You are an intelligent assistant."  # Define system prompt
        
        pygame.mixer.init()  # Initialize pygame mixer

    def transcribe_audio(self, audio_file):
        """
        Transcribes an audio file using Whisper.
        """
        transcription = self.whisper_manager.transcribe(audio_file)
        return transcription

    def generate_response(self, transcription):
        """
        Generates a response based on the transcription using Llama.
        """
        response = self.llama_manager.generate_response(
            conversation_history=self.conversation_history,
            system_prompt=self.system_prompt
        )
        self.conversation_history.append({"role": "user", "content": transcription})  # Update history with user input
        self.conversation_history.append({"role": "assistant", "content": response})  # Update history with AI response
        return response

    def synthesize_speech(self, text_response):
        """
        Converts text response to speech using the SpeechManager.
        """
        speech_output = self.speech_manager.text_to_speech(text_response)
        if isinstance(speech_output, Path):
            # If a Path is returned, read the audio data
            with open(speech_output, 'rb') as f:
                speech_output = f.read()
        return speech_output

    def save_audio(self, audio_data, output_path):
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

    def process_audio_input(self, audio_input_path=None, output_audio_path="output_response.wav"):
        """
        Processes an audio input and produces an output audio response.
        """
        if audio_input_path is None:
            audio_input_path = self.record_audio(duration=3)

        # Step 1: Transcribe input audio
        transcription = self.transcribe_audio(audio_input_path)
        print(f"Transcription: {transcription}")
        
        # Step 2: Generate response using Llama
        response = self.generate_response(transcription)
        print(f"Generated Response: {response}")
        
        # Step 3: Synthesize speech from the response
        speech_output = self.synthesize_speech(response)
        
        # Step 4: Save the synthesized audio
        self.save_audio(speech_output, output_audio_path)
        print(f"Output audio saved to {output_audio_path}")

        # Step 5: Play the saved audio
        self.play_audio(output_audio_path)

    def play_audio(self, audio_path):
        """
        Plays the audio file located at audio_path using pygame mixer.
        """
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            print(f"Playing audio: {audio_path}")
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # Wait until playback is finished
            pygame.mixer.music.unload()
            print("Playback of the response audio completed.")
        except pygame.error as e:
            print(f"Failed to play audio: {e}")

    def record_audio(self, duration=3, sample_rate=44100, channels=1):
        """
        Records audio from the microphone for a specified duration.
        
        :param duration: Duration of the recording in seconds.
        :param sample_rate: Sampling rate of the recording.
        :param channels: Number of audio channels.
        :return: Path to the recorded audio file.
        """
        print("Recording audio...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        sd.wait()  # Wait until recording is finished
        audio_filename = "temp_input.wav"
        with wave.open(audio_filename, 'w') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 2 bytes for 'int16'
            wf.setframerate(sample_rate)
            wf.writeframes(recording.tobytes())
        print(f"Recorded audio saved to {audio_filename}")
        return audio_filename

    def listen_and_respond(self):
        self.whisper_manager.start_listening()
        try:
            while True:
                transcription = self.whisper_manager.get_transcription()
                if transcription and transcription != "No speech detected.":
                    print(f"Transcription: {transcription}")
                    self.conversation_history.append({"role": "user", "content": transcription})  # Add to history
                    response = self.generate_response(transcription)
                    print(f"Generated Response: {response}")
                    speech_output = self.synthesize_speech(response)
                    self.save_audio(speech_output, "output_response.wav")
                    print("Output audio saved to output_response.wav")
                    # Play the saved audio
                    self.play_audio("output_response.wav")
        except KeyboardInterrupt:
            self.whisper_manager.stop_listening()

    def run(self):
        self.listening_thread.start()
        print("Speech-to-Speech Application is running. Press Ctrl+C to exit.")
        self.listening_thread.join()


# Example usage
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    s2s_app = SpeechToSpeechApplication(api_key)
    s2s_app.run()
