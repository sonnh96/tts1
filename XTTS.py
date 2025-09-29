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
        
        # VRAM optimization settings
        self._max_cache_size = 200  # Increased cache size
        self._prefetch_tensors = True
        self._large_batch_mode = True

    def _clear_gpu_cache(self):
        """Clear GPU cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _get_gpu_memory_info(self):
        """Get current GPU memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            free = total - reserved
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": free,
                "utilization_percent": (reserved / total) * 100
            }
        return None

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
            
            # Enable optimizations for CUDA and maximize VRAM usage
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed
            
            # Pre-allocate GPU memory to maximize utilization
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available VRAM
            
            # Enable memory mapping for better memory management
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matmul
            torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for convolutions
            
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
            
            # Keep tensors in VRAM for maximum performance
            if torch.cuda.is_available():
                if not gpt_cond_latent.is_cuda:
                    gpt_cond_latent = gpt_cond_latent.cuda()
                if not speaker_embedding.is_cuda:
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
        
        # Cache the results in VRAM for faster access (use more VRAM)
        if self._enable_caching:
            # Keep tensors in VRAM instead of moving to CPU
            if torch.cuda.is_available():
                cached_gpt = gpt_cond_latent.cuda()
                cached_speaker = speaker_embedding.cuda()
            else:
                cached_gpt = gpt_cond_latent
                cached_speaker = speaker_embedding
                
            self._conditioning_cache[cache_key] = (cached_gpt, cached_speaker)
            print(f"Cached conditioning in VRAM for {speaker_audio_file}")
        
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
        # Input validation
        if not text or not text.strip():
            return [""]  # Return empty string chunk instead of empty list
        
        if lang in ["ja", "zh-cn"]:
            sentences = text.split("。")
        else:
            try:
                ss = sent_tokenize(text)
                g = []
                for s in ss:
                    g.extend(s.split(";"))
                sentences = []
                for s in g:
                    sentences.extend(s.split(","))
            except Exception as e:
                print(f"⚠️  Text tokenization failed: {e}, using simple split")
                sentences = text.split(".")  # Fallback to simple split

        # Ensure we have at least one sentence
        if not sentences:
            sentences = [text]

        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = len(sentence)
            
            if current_length + sentence_tokens > max_tokens:
                if len(current_chunk) > 0:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                else:
                    # If single sentence is too long, add it anyway
                    chunks.append(sentence)
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Ensure we always return at least one chunk
        if not chunks:
            chunks = [text[:max_tokens]]  # Take first max_tokens characters
            
        return chunks
    def _calculate_optimal_batch_size(self, text_chunks):
        """Calculate optimal batch size based on available VRAM and text complexity."""
        if not torch.cuda.is_available():
            return 2
        
        # Handle edge cases
        if not text_chunks or len(text_chunks) == 0:
            return 4
        
        gpu_info = self._get_gpu_memory_info()
        if not gpu_info:
            return 4
        
        # Calculate batch size based on available memory and text complexity
        try:
            avg_chunk_length = sum(len(chunk) for chunk in text_chunks) / len(text_chunks)
        except (TypeError, ZeroDivisionError):
            avg_chunk_length = 50  # Default fallback
        
        # More aggressive batch sizing to use more VRAM
        if gpu_info["free_gb"] > 6:  # Lots of free VRAM
            base_batch = 12 if avg_chunk_length < 100 else 8
        elif gpu_info["free_gb"] > 4:  # Moderate VRAM
            base_batch = 8 if avg_chunk_length < 100 else 6
        elif gpu_info["free_gb"] > 2:  # Limited VRAM
            base_batch = 6 if avg_chunk_length < 100 else 4
        else:
            base_batch = 4
        
        # Apply large batch mode multiplier
        if self._large_batch_mode:
            base_batch = int(base_batch * 1.5)
        
        # Ensure batch size doesn't exceed number of chunks
        base_batch = min(base_batch, len(text_chunks))
        
        print(f"🎯 Optimal batch size: {base_batch} (Free VRAM: {gpu_info['free_gb']:.1f}GB, Avg chunk length: {avg_chunk_length:.0f}, Total chunks: {len(text_chunks)})")
        return base_batch

    def _process_chunks_batch(self, text_chunks, lang_code, gpt_cond_latent, speaker_embedding, batch_size=4):
        """Process text chunks in large batches for maximum VRAM utilization."""
        # Input validation
        if not text_chunks or len(text_chunks) == 0:
            print("⚠️  No text chunks to process")
            return []
            
        wav_chunks = []
        
        # Ensure conditioning tensors are on the correct device
        if torch.cuda.is_available():
            gpt_cond_latent = gpt_cond_latent.cuda()
            speaker_embedding = speaker_embedding.cuda()
        
        # Pre-allocate tensor lists to keep in VRAM
        tensor_cache = []
        
        # Use larger batch sizes to maximize VRAM usage
        effective_batch_size = max(1, batch_size * 2 if self._large_batch_mode else batch_size)
        
        # Process chunks in larger batches
        for i in range(0, len(text_chunks), effective_batch_size):
            batch_chunks = text_chunks[i:i+effective_batch_size]
            batch_results = []
            
            print(f"Processing batch {i//effective_batch_size + 1} with {len(batch_chunks)} chunks")
            
            # Pre-process all chunks in parallel for this batch
            for j, chunk in enumerate(batch_chunks):
                if not chunk or not chunk.strip():
                    print(f"⚠️  Skipping empty chunk {j+1}")
                    continue
                    
                # Use cached inference if available
                try:
                    cache_key = f"{chunk}_{lang_code}_{hashlib.md5(str(gpt_cond_latent.cpu().numpy().tobytes()).encode()).hexdigest()[:8]}"
                except Exception as e:
                    print(f"⚠️  Cache key generation failed: {e}")
                    cache_key = f"{chunk}_{lang_code}_{j}"
                
                if self._enable_caching and cache_key in self._text_cache:
                    wav_chunk = self._text_cache[cache_key]
                    print(f"Using cached result for chunk {j+1}: {chunk[:50]}...")
                else:
                    try:
                        with torch.no_grad():  # Ensure no gradients
                            # Process with optimized parameters for speed
                            wav_chunk = self.model.inference(
                                text=chunk,
                                language=lang_code,
                                gpt_cond_latent=gpt_cond_latent,
                                speaker_embedding=speaker_embedding,
                                temperature=0.05,  # Even lower temperature for max speed
                                length_penalty=1.0,
                                repetition_penalty=3.0,  # Lower repetition penalty for speed
                                top_k=5,   # Much lower top_k for fastest sampling
                                top_p=0.7, # Lower top_p for faster generation
                                num_beams=1,  # Single beam for speed
                            )
                    except RuntimeError as e:
                        print(f"❌ Error processing chunk {j+1}: {e}")
                        continue
                    except Exception as e:
                        print(f"❌ Unexpected error processing chunk {j+1}: {e}")
                        continue
                    
                    # Cache the result in VRAM with larger cache size
                    if self._enable_caching and len(self._text_cache) < self._max_cache_size:
                        try:
                            # Keep cached results in VRAM for faster access
                            if torch.cuda.is_available() and isinstance(wav_chunk["wav"], torch.Tensor):
                                cached_wav = wav_chunk.copy()
                                cached_wav["wav"] = wav_chunk["wav"].cuda() if not wav_chunk["wav"].is_cuda else wav_chunk["wav"]
                                self._text_cache[cache_key] = cached_wav
                            else:
                                self._text_cache[cache_key] = wav_chunk
                        except Exception as e:
                            print(f"⚠️  Failed to cache result: {e}")
                
                # Adjust length for short sentences
                try:
                    keep_len = self._calculate_keep_len(chunk, lang_code)
                    if isinstance(wav_chunk["wav"], torch.Tensor):
                        wav_tensor = wav_chunk["wav"][:keep_len]
                    else:
                        wav_tensor = torch.tensor(wav_chunk["wav"][:keep_len])
                    
                    # Keep tensor in VRAM
                    if torch.cuda.is_available():
                        wav_tensor = wav_tensor.cuda()
                        
                    batch_results.append(wav_tensor)
                    tensor_cache.append(wav_tensor)  # Keep reference to prevent cleanup
                except Exception as e:
                    print(f"⚠️  Failed to process wav chunk {j+1}: {e}")
                    continue
            
            wav_chunks.extend(batch_results)
            
            # Less frequent cache clearing to maintain VRAM usage
            if i % (effective_batch_size * 4) == 0:  # Clear less frequently
                # Only clear if we're running low on memory
                try:
                    gpu_info = self._get_gpu_memory_info()
                    if gpu_info and gpu_info["utilization_percent"] > 90:
                        print(f"High VRAM usage detected: {gpu_info['utilization_percent']:.1f}%, clearing cache")
                        self._clear_gpu_cache()
                except Exception as e:
                    print(f"⚠️  VRAM monitoring failed: {e}")
        
        print(f"✅ Processed {len(text_chunks)} chunks, generated {len(wav_chunks)} audio segments")
        return wav_chunks

    def generate_speech(
        self,
        text: str,
        speaker_audio_file: str,
        language: str = "Tiếng Việt",
        normalize_text: bool = True,
        verbose: bool = False,
        output_chunks: bool = False,
        batch_size: int = None,  # Auto-calculate if None
        enable_caching: bool = True,
        max_vram_usage: bool = True  # New parameter for aggressive VRAM usage
    ) -> str:
        """
        Generate speech from text with aggressive VRAM utilization.
        
        Args:
            text: Input text to convert to speech
            speaker_audio_file: Path to reference speaker audio
            language: Language of the input text
            normalize_text: Whether to normalize the text
            verbose: Whether to print detailed information
            output_chunks: Whether to save individual chunks
            batch_size: Number of chunks to process in parallel (auto-calculated if None)
            enable_caching: Whether to use caching for performance
            max_vram_usage: Use maximum VRAM for best performance
            
        Returns:
            Path to the generated audio file
        """
        self._enable_caching = enable_caching
        self._large_batch_mode = max_vram_usage
        lang_code = self.LANGUAGE_CODE_MAP.get(language, "vi")
        
        # Print initial VRAM usage
        if verbose:
            gpu_info = self._get_gpu_memory_info()
            if gpu_info:
                print(f"🖥️  Initial VRAM: {gpu_info['reserved_gb']:.2f}GB/{gpu_info['total_gb']:.2f}GB ({gpu_info['utilization_percent']:.1f}%)")
        
        # Get cached speaker conditioning (major performance boost)
        gpt_cond_latent, speaker_embedding = self._get_cached_conditioning(speaker_audio_file)

        # Normalize text if needed
        if normalize_text and lang_code == "vi":
            text = self.normalize_vietnamese_text(text)

        # Split text into chunks with optimized chunk size for VRAM usage
        text_chunks = self.split_text(text, lang_code, max_tokens=150)  # Smaller chunks for more batches
        
        # Validate text chunks
        if not text_chunks or len(text_chunks) == 0:
            print("⚠️  No text chunks generated, creating fallback")
            text_chunks = [text] if text.strip() else ["Hello"]
        
        # Calculate optimal batch size for VRAM utilization
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(text_chunks)
        else:
            # Ensure batch size doesn't exceed number of chunks
            batch_size = min(batch_size, len(text_chunks))
        
        if verbose:
            print(f"📊 Processing {len(text_chunks)} chunks with batch size {batch_size}")
            print(f"💾 Cache sizes - Conditioning: {len(self._conditioning_cache)}, Text: {len(self._text_cache)}")

        # Process chunks with aggressive VRAM utilization
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

        # Combine all chunks efficiently in VRAM
        if wav_chunks:
            # Keep final concatenation in VRAM for speed
            final_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
            if torch.cuda.is_available():
                final_wav = final_wav.cuda()
        else:
            # Fallback if no chunks were processed
            final_wav = torch.zeros(1, 1024)
            if torch.cuda.is_available():
                final_wav = final_wav.cuda()
        
        output_path = os.path.join(self.output_dir, f"{self._get_file_name(text)}.wav")
        
        # Move to CPU only for saving
        final_wav_cpu = final_wav.cpu()
        torchaudio.save(output_path, final_wav_cpu, 24000)

        if verbose:
            print(f"✅ Saved final file to {output_path}")
            gpu_info = self._get_gpu_memory_info()
            if gpu_info:
                print(f"🖥️  Final VRAM: {gpu_info['reserved_gb']:.2f}GB/{gpu_info['total_gb']:.2f}GB ({gpu_info['utilization_percent']:.1f}%)")
            print(f"💾 Cache stats - Conditioning: {len(self._conditioning_cache)}, Text: {len(self._text_cache)}")

        return output_path
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self._conditioning_cache.clear()
        self._text_cache.clear()
        self._clear_gpu_cache()
        print("All caches cleared")
    
    def get_cache_info(self):
        """Get information about current cache usage and VRAM utilization."""
        gpu_info = self._get_gpu_memory_info()
        
        cache_info = {
            "conditioning_cache_size": len(self._conditioning_cache),
            "text_cache_size": len(self._text_cache),
            "max_cache_size": self._max_cache_size,
            "conditioning_keys": list(self._conditioning_cache.keys()),
            "large_batch_mode": self._large_batch_mode,
            "gpu_available": torch.cuda.is_available(),
        }
        
        if gpu_info:
            cache_info.update({
                "gpu_memory_allocated_gb": gpu_info["allocated_gb"],
                "gpu_memory_reserved_gb": gpu_info["reserved_gb"],
                "gpu_memory_total_gb": gpu_info["total_gb"],
                "gpu_memory_free_gb": gpu_info["free_gb"],
                "gpu_utilization_percent": gpu_info["utilization_percent"]
            })
        else:
            cache_info["gpu_memory_allocated"] = 0
            
        return cache_info