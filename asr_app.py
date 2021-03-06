import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import streamlit as st

st.header("Trascribe Russian Audio")

uploaded_file = st.file_uploader(label = "Please upload your file" )

LANG_ID = "ru"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"

#load wav2vec2 tokenizer and model
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

@st.cache
def asr_transcript(audio_file):
    transcript = ""

    # Stream over 20 seconds chunks
    stream = librosa.stream(
        audio_file.name, block_length=20, frame_length=16000, hop_length=16000
    )

    for speech in stream:
        if len(speech.shape) > 1:
            speech = speech[:, 0] + speech[:, 1]

        inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        transcript += transcription.lower() + " "

    return transcript


if uploaded_file is not None:

    audio_file = open(uploaded_file, 'rb')
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format='audio/wav')

    with st.spinner('Please wait...'):
        transcript = asr_transcript(uploaded_file)

    st.balloons()
    st.header("Transcribed Text")
    st.subheader(transcript)
