import streamlit as st
import os
import torch
from transformers import pipeline

# Carrega e cacheia o modelo
#@st.cache_resource
def load_model():
    # Seleciona o dispositivo e o tipo de dados do tensor
    if torch.cuda.is_available():
        device = 0  # GPU
        torch_dtype = torch.float16
    else:
        device = -1  # CPU
        torch_dtype = torch.float32

    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        torch_dtype=torch_dtype,
        device=device,
    )

def generate_structured_minutes(text: str) -> str:
    """
    Função para estruturar a ata com base na transcrição.
    Personalize conforme necessário.
    """
    # Estrutura básica da ata
    structure = (
        "**Ata de Reunião**\n\n"
        "**Data:** [INSERIR DATA]\n\n"
        "**Participantes:**\n- [LISTAR PARTICIPANTES]\n\n"
        "**Pontos Discutidos:**\n"
        f"{text}\n\n"
        "**Decisões Tomadas:**\n- [LISTAR DECISÕES]\n\n"
        "**Ações Futuras:**\n- [LISTAR AÇÕES]"
    )
    return structure

def main():
    st.set_page_config(page_title="Gerador de Atas", layout="wide")
    st.title("📝 Gerador Automático de Atas")
    st.markdown("### Converta áudios de reuniões em atas estruturadas")
    
    # Upload do arquivo de áudio
    audio_file = st.file_uploader(
        "Carregue seu arquivo de áudio (MP3, WAV, OGG)",
        type=["mp3", "wav", "ogg"]
    )
    
    if audio_file is not None:
        # Salva o arquivo de áudio em um diretório temporário
        os.makedirs("audios", exist_ok=True)
        audio_path = os.path.join("audios", audio_file.name)
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        # Carrega o modelo
        try:
            pipe = load_model()
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
            return
        
        # Processa o áudio
        with st.spinner("Processando áudio... (Isso pode levar alguns minutos)"):
            try:
                result = pipe(
                    audio_path,
                    generate_kwargs={
                        "language": "portuguese",
                        "return_timestamps": True
                    }
                )
            except Exception as e:
                st.error(f"Erro ao processar o áudio: {e}")
                return
        
        # Exibe os resultados
        st.subheader("Transcrição Completa:")
        st.write(result.get("text", "Transcrição não disponível."))
        
        st.subheader("Ata Estruturada:")
        structured_text = generate_structured_minutes(result.get("text", ""))
        st.write(structured_text)
        
        download_filename = f"ata_{os.path.splitext(audio_file.name)[0]}.txt"
        st.download_button(
            label="⬇️ Baixar Ata",
            data=structured_text,
            file_name=download_filename,
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
