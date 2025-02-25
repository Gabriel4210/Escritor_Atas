import streamlit as st
import os
import torch
from transformers import pipeline

# Configuração inicial do modelo
@st.cache_resource
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch_dtype,
        device=device,
    )

def main():
    st.set_page_config(page_title="Gerador de Atas", layout="wide")
    
    st.title("📝 Gerador Automático de Atas")
    st.markdown("### Converta áudios de reuniões em atas estruturadas")

    # Upload de arquivo de áudio
    audio_file = st.file_uploader(
        "Carregue seu arquivo de áudio (MP3, WAV, OGG)",
        type=["mp3", "wav", "ogg"]
    )

    if audio_file:
        # Criar diretório para armazenar áudios
        os.makedirs("audios", exist_ok=True)
        audio_path = os.path.join("audios", audio_file.name)
        
        # Salvar arquivo
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        # Carregar modelo
        pipe = load_model()
        
        # Processar áudio
        with st.spinner("Processando áudio... (Isso pode levar alguns minutos)"):
            result = pipe(
                audio_path,
                generate_kwargs={
                    "language": "portuguese",
                    "return_timestamps": True
                }
            )
        
        # Exibir resultados
        st.subheader("Transcrição Completa:")
        st.write(result["text"])

        # Gerar ata estruturada (exemplo básico)
        st.subheader("Ata Estruturada:")
        structured_text = generate_structured_minutes(result["text"])
        st.write(structured_text)

        # Botão de download
        download_filename = f"ata_{os.path.splitext(audio_file.name)[0]}.txt"
        st.download_button(
            label="⬇️ Baixar Ata",
            data=structured_text,
            file_name=download_filename,
            mime="text/plain"
        )

def generate_structured_minutes(text):
    """Função básica para estruturação da ata (personalize conforme necessidade)"""
    structure = [
        "**Ata de Reunião**\n\n",
        "**Data:** [INSERIR DATA]\n\n",
        "**Participantes:**\n- [LISTAR PARTICIPANTES]\n\n",
        "**Pontos Discutidos:**\n",
        text + "\n\n",
        "**Decisões Tomadas:**\n- [LISTAR DECISÕES]\n\n",
        "**Ações Futuras:**\n- [LISTAR AÇÕES]"
    ]
    return "\n".join(structure)

if __name__ == "__main__":
    main()
