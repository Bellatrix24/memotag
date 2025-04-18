# MemoTag – Voice-Based Cognitive Decline Detection

This is a simple proof-of-concept project that screens for potential signs of cognitive stress or early decline using short voice recordings. The system uses speech and linguistic features to flag unusual patterns that may indicate mental fatigue, hesitation, or confusion.

## What It Does

Given a .wav audio file of someone speaking, this app:

- Transcribes the speech using OpenAI's Whisper model
- Extracts features like:
  - Speech rate (words per second)
  - Hesitation count ("uh", "um", etc.)
  - Number of long pauses
  - Pitch variation
- Uses unsupervised learning to detect samples that deviate from typical speaking patterns
- Returns a simple output: whether a potential cognitive risk is detected or not

## Demo Setup

This project is built with [Streamlit](https://streamlit.io/). To run the web app:

bash
streamlit run deploy.py


It will launch a local browser app where you can upload .wav files and see the results instantly.

## Requirements

All required packages are listed in requirements.txt. Make sure ffmpeg is installed as Whisper relies on it for audio decoding.

### Example setup:

bash
pip install -r requirements.txt
python -m spacy download en_core_web_md


## Dataset

The voice samples used in this project were provided by my colleague, who kindly simulated various levels of cognitive strain for testing purposes. No real patient data has been used. This is purely experimental.

## How It Works

Once an audio file is uploaded:

1. The app transcribes it using Whisper
2. It calculates features such as pauses and hesitations using librosa and regex
3. An Isolation Forest model checks if the sample falls outside the “normal” speaking range
4. If it does, the user is flagged as “at risk” (for demonstration purposes only)

## Why This Matters

Early cognitive issues often show up subtly in speech — slower response times, hesitation, repetition, or loss of word recall. This tool doesn’t diagnose anything, but it shows how AI can help flag these patterns early, potentially aiding future assistive technologies.

## About

*Developed by:* Saumya Singh  
*Collaborator (Voice Simulations):* A fellow researcher

This project was built as part of the MemoTag initiative — exploring AI tools that can assist in accessible and non-invasive cognitive screening.

---

⚠ This is a research prototype and is *not* intended for medical use or diagnosis.