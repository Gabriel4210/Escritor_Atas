import streamlit as st
import os
import torch
import tempfile
import subprocess
from transformers import pipeline
from pydub import AudioSegment

os.environ["PATH"] += os.pathsep + "/usr/bin"

@st.cache_resource
def load_model():
    # Forçar o uso de CPU para evitar problemas com Torch
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        device=torch_device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

def convert_audio(input_path, output_format="wav"):
    output_path = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False).name
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-loglevel", "quiet",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

def main():
    st.title("📝 Gerador Automático de Atas")
    
    audio_file = st.file_uploader(
        "Carregue seu arquivo de áudio (MP3, WAV, OGG)",
        type=["mp3", "wav", "ogg"]
    )

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_input:
            tmp_input.write(audio_file.read())
            input_path = tmp_input.name
        
        try:
            # Conversão para WAV
            converted_path = convert_audio(input_path)
            
            # Carregar modelo
            pipe = load_model()
            
            # Processamento
            with st.spinner("Processando áudio..."):
                result = pipe(
                    converted_path,
                    generate_kwargs={"language": "portuguese"}
                )
                
            st.success("Processamento concluído!")
            st.write(result["text"])

        except subprocess.CalledProcessError as e:
            st.error(f"Erro na conversão de áudio: {str(e)}")
        except Exception as e:
            st.error(f"Erro no processamento: {str(e)}")
        finally:
            # Limpeza garantida
            for path in [input_path, converted_path]:
                if os.path.exists(path):
                    os.remove(path)

if __name__ == "__main__":
    main()
