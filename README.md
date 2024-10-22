# T5 Speech-to-Speech Model

## Overview

The **T5 Speech-to-Speech Model** leverages state-of-the-art pretrained models to convert spoken language in real-time. By utilizing a teacher-student training paradigm, this project enables the creation of efficient and accurate speech-to-speech systems suitable for various applications such as virtual assistants, transcription services, and more.

## Features

- **Real-Time Speech Conversion:** Convert spoken input into synthesized speech instantly.
- **Teacher-Student Training Pipeline:** Utilize pretrained models as teachers to train a lightweight student model for efficient deployment.
- **Modular Architecture:** Integrates multiple pretrained models including Whisper, ChatGPT, and a Text-to-Speech (TTS) model.
- **Super-Alignment Training:** Advanced training scripts to ensure high-quality alignment between input and synthesized speech.
- **Extensible Design:** Easily extendable to incorporate additional models or functionalities.

## Architecture

The teacher model is built upon a three-model pipeline:

1. **Whisper:** Handles transcription of input speech.
2. **ChatGPT:** Generates textual responses based on transcriptions.
3. **Text-to-Speech (TTS) Model:** Synthesizes speech from the generated text.

A student model is trained using the outputs from these pretrained models to perform end-to-end speech-to-speech conversion efficiently.

## Repository Structure
```bash
├── README.md
├── .gitignore
├── .env
├── llama_manager.py
├── main.py
├── t5_eval.py
├── t5_example.py
├── speech_manager.py
├── train.py
└── requirements.txt
```


- **llama_manager.py:** Manages interactions with the LLAMA model.
- **main.py:** Entry point for the Speech-to-Speech application.
- **t5_eval.py:** Evaluation scripts for the T5 model.
- **model.py:** Defines the student model architecture.
- **s2smodel.py:** Speech-to-Speech model utilities.
- **speech_manager.py:** Handles speech synthesis and processing.
- **superalignment_example.py:** Example scripts for super-alignment training.
- **train.py:** Training script for the student model.
- **requirements.txt:** List of dependencies.

## Installation

### Prerequisites

- Python 3.8 or higher
- [Git](https://git-scm.com/)
- [Pip](https://pip.pypa.io/en/stable/installation/)
- [FFmpeg](https://ffmpeg.org/) (for audio processing)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/peytontolbert/T5-SpeechtoSpeech.git
   cd T5-SpeechtoSpeech
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**

   Create a `.env` file in the root directory and add the following:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   # Add other necessary environment variables here
   ```

## Usage

### Running the Application
```bash
python main.py
```


This will start the Speech-to-Speech application, which listens for audio input, processes it, and provides a synthesized speech response in real-time.

### Training the Student Model

To train the student model using the pretrained teacher models:
```bash
python train.py
```


**Parameters:**

- `learning_rate`: Learning rate for the optimizer (default: `1e-4`)
- `num_epochs`: Number of training epochs (default: `10`)
- `save_steps`: Frequency of saving model checkpoints (default: `2`)
