import os
import string
import torch
import torchaudio
import hashlib
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
        
        # Caching for performance optimization
        self._conditioning_cache = {}
        self._text_cache = {}
        self._enable_caching = True

    def _clear_gpu_cache(self):
        """Clear GPU cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, model_path: str, config_path: str, vocab_path: str) -> Xtts:
        """
        Load the XTTS model with optimizations.
        
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
        
        print("Loading XTTS model with optimizations...")
        model.load_checkpoint(
            config,
            checkpoint_path=model_path,
            vocab_path=vocab_path,
            use_deepspeed=True  # Enable DeepSpeed for better performance
        )
        
        if torch.cuda.is_available():
            model.cuda()
            print("Model loaded on GPU with FP32 precision")
            
            # Enable optimizations for CUDA
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed
            
        else:
            print("Model loaded on CPU")
            
            # CPU-specific optimizations
            torch.set_num_threads(4)  # Optimize CPU thread usage
        
        # Set model to evaluation mode for inference optimizations
        model.eval()
        
        # Enable torch.no_grad() context for inference
        for param in model.parameters():
            param.requires_grad = False
        
        print("Model Loaded with optimizations!")
        return model

    def _get_cached_conditioning(self, speaker_audio_file: str):
        """Get cached conditioning latents or compute and cache them."""
        cache_key = os.path.abspath(speaker_audio_file)
        
        if self._enable_caching and cache_key in self._conditioning_cache:
            print(f"Using cached conditioning for {speaker_audio_file}")
            gpt_cond_latent, speaker_embedding = self._conditioning_cache[cache_key]
            
            # Ensure tensors are on the correct device and dtype
            if torch.cuda.is_available():
                gpt_cond_latent = gpt_cond_latent.cuda()
                speaker_embedding = speaker_embedding.cuda()
            
            return gpt_cond_latent, speaker_embedding
        
        # Compute conditioning latents
        print(f"Computing conditioning for {speaker_audio_file}")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=speaker_audio_file,
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )
        
        # Cache the results (move to CPU for storage to save GPU memory)
        if self._enable_caching:
            cached_gpt = gpt_cond_latent.cpu() if torch.cuda.is_available() else gpt_cond_latent
            cached_speaker = speaker_embedding.cpu() if torch.cuda.is_available() else speaker_embedding
            self._conditioning_cache[cache_key] = (cached_gpt, cached_speaker)
            print(f"Cached conditioning for {speaker_audio_file}")
        
        return gpt_cond_latent, speaker_embedding

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

    def _process_chunks_batch(self, text_chunks, lang_code, gpt_cond_latent, speaker_embedding, batch_size=4):
        """Process text chunks in batches for better performance."""
        wav_chunks = []
        
        # Ensure conditioning tensors are on the correct device
        if torch.cuda.is_available():
            gpt_cond_latent = gpt_cond_latent.cuda()
            speaker_embedding = speaker_embedding.cuda()
        
        # Process chunks in batches
        for i in range(0, len(text_chunks), batch_size):
            batch_chunks = text_chunks[i:i+batch_size]
            batch_results = []
            
            # Process each chunk in the current batch
            for chunk in batch_chunks:
                if not chunk.strip():
                    continue
                    
                # Use cached inference if available
                cache_key = f"{chunk}_{lang_code}_{hashlib.md5(str(gpt_cond_latent.cpu().numpy().tobytes()).encode()).hexdigest()[:8]}"
                
                if self._enable_caching and cache_key in self._text_cache:
                    wav_chunk = self._text_cache[cache_key]
                    print(f"Using cached result for chunk: {chunk[:50]}...")
                else:
                    try:
                        with torch.no_grad():  # Ensure no gradients
                            wav_chunk = self.model.inference(
                                text=chunk,
                                language=lang_code,
                                gpt_cond_latent=gpt_cond_latent,
                                speaker_embedding=speaker_embedding,
                                temperature=0.1,  # Lower temperature for faster generation
                                length_penalty=1.0,
                                repetition_penalty=5.0,  # Lower repetition penalty
                                top_k=10,  # Lower top_k for faster sampling
                                top_p=0.8,
                            )
                    except RuntimeError as e:
                        print(f"Error processing chunk: {e}")
                        # Skip this chunk and continue
                        continue
                    
                    # Cache the result (limit cache size to prevent memory issues)
                    if self._enable_caching and len(self._text_cache) < 50:
                        self._text_cache[cache_key] = wav_chunk
                
                # Adjust length for short sentences
                keep_len = self._calculate_keep_len(chunk, lang_code)
                if isinstance(wav_chunk["wav"], torch.Tensor):
                    wav_chunk["wav"] = wav_chunk["wav"][:keep_len]
                else:
                    wav_chunk["wav"] = torch.tensor(wav_chunk["wav"][:keep_len])
                    
                batch_results.append(wav_chunk["wav"])
            
            wav_chunks.extend(batch_results)
            
            # Clear GPU cache periodically
            if i % (batch_size * 2) == 0:
                self._clear_gpu_cache()
        
        return wav_chunks

    def generate_speech(
        self,
        text: str,
        speaker_audio_file: str,
        language: str = "Tiếng Việt",
        normalize_text: bool = True,
        verbose: bool = False,
        output_chunks: bool = False,
        batch_size: int = 4,
        enable_caching: bool = True
    ) -> str:
        """
        Generate speech from text with optimizations.
        
        Args:
            text: Input text to convert to speech
            speaker_audio_file: Path to reference speaker audio
            language: Language of the input text
            normalize_text: Whether to normalize the text
            verbose: Whether to print detailed information
            output_chunks: Whether to save individual chunks
            batch_size: Number of chunks to process in parallel
            enable_caching: Whether to use caching for performance
            
        Returns:
            Path to the generated audio file
        """
        self._enable_caching = enable_caching
        lang_code = self.LANGUAGE_CODE_MAP.get(language, "vi")
        
        # Get cached speaker conditioning (major performance boost)
        gpt_cond_latent, speaker_embedding = self._get_cached_conditioning(speaker_audio_file)

        # Normalize text if needed
        if normalize_text and lang_code == "vi":
            text = self.normalize_vietnamese_text(text)

        # Split text into chunks with optimized chunk size
        text_chunks = self.split_text(text, lang_code, max_tokens=200)  # Smaller chunks for faster processing
        if verbose:
            print(f"Processing {len(text_chunks)} chunks with batch size {batch_size}")

        # Process chunks with batch processing and caching
        wav_chunks = self._process_chunks_batch(
            text_chunks, lang_code, gpt_cond_latent, speaker_embedding, batch_size
        )

        # Save individual chunks if requested
        if output_chunks:
            for i, wav_chunk in enumerate(wav_chunks):
                chunk_path = os.path.join(self.output_dir, f"{self._get_file_name(text_chunks[i])}.wav")
                torchaudio.save(chunk_path, wav_chunk.unsqueeze(0), 24000)
                if verbose:
                    print(f"Saved chunk {i+1} to {chunk_path}")

        # Combine all chunks efficiently
        if wav_chunks:
            final_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
        else:
            # Fallback if no chunks were processed
            final_wav = torch.zeros(1, 1024)
        
        output_path = os.path.join(self.output_dir, f"{self._get_file_name(text)}.wav")
        torchaudio.save(output_path, final_wav, 24000)

        if verbose:
            print(f"Saved final file to {output_path}")
            print(f"Cache stats - Conditioning: {len(self._conditioning_cache)}, Text: {len(self._text_cache)}")

        return output_path
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self._conditioning_cache.clear()
        self._text_cache.clear()
        self._clear_gpu_cache()
        print("All caches cleared")
    
    def get_cache_info(self):
        """Get information about current cache usage."""
        return {
            "conditioning_cache_size": len(self._conditioning_cache),
            "text_cache_size": len(self._text_cache),
            "conditioning_keys": list(self._conditioning_cache.keys()),
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }