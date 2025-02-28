import streamlit as st
import os
import torch
from transformers import pipeline
from pydub import AudioSegment

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
        model="openai/whisper-large-v3",  # Modelo atualizado
        torch_dtype=torch_dtype,
        device=device,
    )

def generate_structured_minutes(text: str) -> str:
    """
    Função para estruturar a ata com base na transcrição.
    Personalize conforme necessário.
    """
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

def split_audio(file_path, segment_length_ms=60000):
    """
    Divide o áudio em segmentos com duração definida (em ms).
    Retorna uma lista com os caminhos dos arquivos segmentados.
    """
    audio = AudioSegment.from_file(file_path)
    segments = []
    segment_dir = os.path.join(os.path.dirname(file_path), "segments")
    os.makedirs(segment_dir, exist_ok=True)
    
    for i in range(0, len(audio), segment_length_ms):
        segment = audio[i:i+segment_length_ms]
        segment_filename = os.path.join(segment_dir, f"segment_{i//segment_length_ms}.wav")
        segment.export(segment_filename, format="wav")
        segments.append(segment_filename)
    return segments

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
        
        full_transcription = ""
        # Se o arquivo for maior que 50 MB, realiza segmentação
        file_size = os.path.getsize(audio_path)
        threshold = 50 * 1024 * 1024  # 50 MB em bytes
        
        if file_size > threshold:
            st.info("Áudio grande detectado. Segmentando o áudio para processamento...")
            try:
                segments = split_audio(audio_path, segment_length_ms=60000)  # 1 minuto por segmento
            except Exception as e:
                st.error(f"Erro ao segmentar o áudio: {e}")
                return
            
            progress_bar = st.progress(0)
            total_segments = len(segments)
            for idx, segment_file in enumerate(segments):
                with st.spinner(f"Processando segmento {idx+1} de {total_segments}..."):
                    try:
                        result = pipe(
                            segment_file,
                            generate_kwargs={
                                "language": "portuguese",
                                "num_beams": 5,
                                "temperature": 0.0,
                                "return_timestamps": True
                            }
                        )
                        text_segment = result.get("text", "")
                        full_transcription += text_segment + " "
                    except Exception as e:
                        st.error(f"Erro ao processar o segmento {idx+1}: {e}")
                progress_bar.progress((idx + 1) / total_segments)
        else:
            with st.spinner("Processando áudio... (Isso pode levar alguns minutos)"):
                try:
                    result = pipe(
                        audio_path,
                        generate_kwargs={
                            "language": "portuguese",
                            "num_beams": 5,
                            "temperature": 0.0,
                            "return_timestamps": True
                        }
                    )
                    full_transcription = result.get("text", "")
                except Exception as e:
                    st.error(f"Erro ao processar o áudio: {e}")
                    return
        
        # Exibe os resultados
        st.subheader("Transcrição Completa:")
        st.write(full_transcription)
        
        st.subheader("Ata Estruturada:")
        structured_text = generate_structured_minutes(full_transcription)
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
