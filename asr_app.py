import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import streamlit as st

st.header("Trascribe Russian Audio")

filePath = st.text_input(label = "Please enter the path to the file" )

LANG_ID = "ru"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
#test_dataset = load_dataset("common_voice", LANG_ID, split=f"test[:{SAMPLES}]")

# Preprocessing the datasets.
# We need to read the audio files as arrays
@st.cache
def speech_file_to_array_fn(filepath):
    speech_array, sampling_rate = librosa.load(filepath, sr=16_000)
    return speech_array

#test_dataset = test_dataset.map(speech_file_to_array_fn)
if filePath:

    audio_file = open(filePath, 'rb')
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format='audio/wav')

    with st.spinner('Please wait...'):

        processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

        speech = speech_file_to_array_fn(filepath=filePath)
        inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = processor.batch_decode(predicted_ids)

    st.balloons()
    st.header("Transcribed Text")
    st.subheader(predicted_sentence)