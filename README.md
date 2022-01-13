# Russian-ASR-app
This repository contains all project files for application with automatic speech recognition in Russian language.

The model used is available at hugginface. It was developed by Jonatas Grossman with WER at 13.38% and CER at 2.86%. It presents a fine-tuned facebook/wav2vec2-large-xlsr-53 model on a Russian dataset using the Common Voice and CSS10. When using this model, you have to make sure that your speech input is sampled at 16kHz. The other second best model is anton-l/wav2vec2-large-xlsr-53-russian with WER at 19.49% and CER at 4.15%. Hence, the former model was selected for the app.

To use the application one must provide the path to the wav file sampled at 16 kHz.
