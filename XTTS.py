import os
import string
import torch
import torchaudio
from datetime import datetime
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from underthesea import sent_tokenize
from unidecode import unidecode
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from vinorm import TTSnorm

class XTTS:
    LANGUAGE_CODE_MAP = {
        "Tiếng Việt": "vi",
        "Tiếng Anh": "en",
        "Tiếng Trung (giản thể)": "zh-cn",
        "Tiếng Nhật": "ja"
    }

    def __init__(
        self,
        model_path: str = "model/model.pth",
        config_path: str = "model/config.json",
        vocab_path: str = "model/vocab.json",
        output_dir: str = "./output"
    ):
        """
        Initialize the Vietnamese XTTS model.
        
        Args:
            model_path: Path to the model checkpoint
            config_path: Path to the model config file
            vocab_path: Path to the vocabulary file
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model = self._load_model(model_path, config_path, vocab_path)

    def _clear_gpu_cache(self):
        """Clear GPU cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, model_path: str, config_path: str, vocab_path: str) -> Xtts:
        """
        Load the XTTS model.
        
        Args:
            model_path: Path to model checkpoint
            config_path: Path to model config
            vocab_path: Path to vocabulary file
            
        Returns:
            Loaded XTTS model
        """
        self._clear_gpu_cache()
        
        config = XttsConfig()
        config.load_json(config_path)
        model = Xtts.init_from_config(config)
        
        print("Loading XTTS model...")
        model.load_checkpoint(
            config,
            checkpoint_path=model_path,
            vocab_path=vocab_path,
            use_deepspeed=True
        )
        
        if torch.cuda.is_available():
            model.cuda()
        
        print("Model Loaded!")
        return model

    @staticmethod
    def _get_file_name(text: str, max_char: int = 50) -> str:
        """Generate a filename from text."""
        filename = text[:max_char].lower()
        filename = filename.replace(" ", "_")
        filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
        filename = unidecode(filename)
        current_datetime = datetime.now().strftime("%m%d%H%M%S")
        return f"{current_datetime}_{filename}"

    @staticmethod
    def _calculate_keep_len(text: str, lang: str) -> int:
        """Calculate length to keep for the audio output."""
        if lang in ["ja", "zh-cn"]:
            return -1

        word_count = len(text.split())
        num_punct = sum(text.count(p) for p in ".,!?")

        if word_count < 5:
            return 15000 * word_count + 2000 * num_punct
        elif word_count < 10:
            return 13000 * word_count + 2000 * num_punct
        return -1

    @staticmethod
    def normalize_vietnamese_text(text: str) -> str:
        """Normalize Vietnamese text for TTS."""
        text = (
            TTSnorm(text, unknown=False, lower=False, rule=True)
            .replace("..", ".")
            .replace("!.", "!")
            .replace("?.", "?")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace('"', "")
            .replace("'", "")
            .replace("AI", "Ây Ai")
            .replace("A.I", "Ây Ai")
        )
        return text

    def split_text(self, text: str, lang: str = 'vi', max_tokens: int = 250) -> List[str]:
        """Split text into manageable chunks."""
        if lang in ["ja", "zh-cn"]:
            sentences = text.split("。")
        else:
            ss = sent_tokenize(text)
            g = []
            for s in ss:
                g.extend(s.split(";"))
            sentences = []
            for s in g:
                sentences.extend(s.split(","))

        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence)
            
            if current_length + sentence_tokens > max_tokens:
                if len(current_chunk) > 0:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                else:
                    chunks.append(sentence)
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def generate_speech(
        self,
        text: str,
        speaker_audio_file: str,
        language: str = "Tiếng Việt",
        normalize_text: bool = True,
        verbose: bool = False,
        output_chunks: bool = False
    ) -> str:
        """
        Generate speech from text.
        
        Args:
            text: Input text to convert to speech
            speaker_audio_file: Path to reference speaker audio
            language: Language of the input text
            normalize_text: Whether to normalize the text
            verbose: Whether to print detailed information
            output_chunks: Whether to save individual chunks
            
        Returns:
            Path to the generated audio file
        """
        lang_code = self.LANGUAGE_CODE_MAP.get(language, "vi")
        
        # Get speaker conditioning
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=speaker_audio_file,
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )

        # Normalize text if needed
        if normalize_text and lang_code == "vi":
            text = self.normalize_vietnamese_text(text)

        # Split text into chunks
        text_chunks = self.split_text(text, lang_code)
        if verbose:
            print(f"Processing {len(text_chunks)} chunks:")
            print(text_chunks)

        # Process each chunk
        wav_chunks = []
        for chunk in tqdm(text_chunks):
            print("ASJHKAHSKJA", len(chunk))
            if not chunk.strip():
                continue

            wav_chunk = self.model.inference(
                text=chunk,
                language=lang_code,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.3,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=30,
                top_p=0.85,
            )

            # Adjust length for short sentences
            keep_len = self._calculate_keep_len(chunk, lang_code)
            wav_chunk["wav"] = torch.tensor(wav_chunk["wav"][:keep_len])

            if output_chunks:
                chunk_path = os.path.join(self.output_dir, f"{self._get_file_name(chunk)}.wav")
                torchaudio.save(chunk_path, wav_chunk["wav"].unsqueeze(0), 24000)
                if verbose:
                    print(f"Saved chunk to {chunk_path}")

            wav_chunks.append(wav_chunk["wav"])

        # Combine all chunks and save
        final_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
        output_path = os.path.join(self.output_dir, f"{self._get_file_name(text)}.wav")
        torchaudio.save(output_path, final_wav, 24000)

        if verbose:
            print(f"Saved final file to {output_path}")

        return output_path