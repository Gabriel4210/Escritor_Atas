import streamlit as st
import os
import torch
import tempfile
from transformers import pipeline
from pydub import AudioSegment

# Configurações do Streamlit para arquivos grandes
st.set_page_config(page_title="Gerador de Atas", layout="wide")
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",  # Modelo menor para melhor performance
        torch_dtype=torch_dtype,
        device=device,
    )

def process_large_audio(audio_path, pipe):
    """Processa o áudio em chunks para evitar estouro de memória"""
    sound = AudioSegment.from_file(audio_path)
    chunk_length_ms = 600000  # 10 minutos por chunk
    chunks = sound[::chunk_length_ms]

    full_text = ""
    for i, chunk in enumerate(chunks):
        with tempfile.NamedTemporaryFile(suffix=".wav") as fp:
            chunk.export(fp.name, format="wav")
            result = pipe(
                fp.name,
                generate_kwargs={
                    "language": "portuguese",
                    "return_timestamps": False
                }
            )
            full_text += result["text"] + "\n"
            st.info(f"Processado chunk {i+1}/{len(chunks)}")
    
    return full_text

def main():
    st.title("📝 Gerador Automático de Atas")
    
    audio_file = st.file_uploader(
        "Carregue seu arquivo de áudio (MP3, WAV, OGG)",
        type=["mp3", "wav", "ogg"]
    )

    if audio_file:
        with st.spinner("Preparando processamento..."):
            # Converter para formato WAV 16kHz mono
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(audio_file.read())
                audio_path = tmp_file.name
                
            sound = AudioSegment.from_file(audio_path)
            sound = sound.set_frame_rate(16000).set_channels(1)
            converted_path = "converted_audio.wav"
            sound.export(converted_path, format="wav")

            pipe = load_model()
            
            try:
                with st.spinner("Processando áudio (pode levar vários minutos)..."):
                    if audio_file.size > 50 * 1024 * 1024:  # >50MB
                        text = process_large_audio(converted_path, pipe)
                    else:
                        result = pipe(
                            converted_path,
                            generate_kwargs={
                                "language": "portuguese",
                                "return_timestamps": False
                            }
                        )
                        text = result["text"]

                st.success("Processamento concluído!")
                st.subheader("Ata Gerada:")
                st.write(text)

            except Exception as e:
                st.error(f"Erro no processamento: {str(e)}")
            finally:
                os.unlink(audio_path)
                os.unlink(converted_path)

if __name__ == "__main__":
    main()
