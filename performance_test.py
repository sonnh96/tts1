#!/usr/bin/env python3
"""
Performance testing script for optimized XTTS model.
Tests the improvements made to reduce text-to-speech processing time.
"""

import time
import torch
from XTTS import XTTS

def test_performance():
    """Test the performance improvements of the optimized XTTS model."""
    
    print("🚀 Starting XTTS Performance Test")
    print("=" * 50)
    
    # Test texts of different lengths
    test_texts = [
        "Xin chào, đây là một câu ngắn để kiểm tra.",
        "Đây là một đoạn văn bản dài hơn để kiểm tra hiệu suất của mô hình text-to-speech. Chúng ta sẽ xem mô hình xử lý như thế nào với văn bản có độ dài trung bình.",
        """Đây là một đoạn văn bản rất dài để kiểm tra khả năng xử lý của mô hình với các văn bản phức tạp. 
        Mô hình cần phải xử lý nhiều câu khác nhau, với các dấu câu và cấu trúc ngữ pháp phức tạp. 
        Chúng ta sẽ đo thời gian xử lý để đánh giá hiệu suất của các tối ưu hóa đã được áp dụng. 
        Các tối ưu hóa bao gồm việc sử dụng cache, xử lý theo batch, và giảm thiểu các phép tính không cần thiết."""
    ]
    
    # Initialize model
    print("📋 Initializing XTTS model...")
    start_init = time.time()
    model = XTTS()
    init_time = time.time() - start_init
    print(f"✅ Model initialized in {init_time:.2f} seconds")
    
    # Test with caching enabled
    print("\n🔄 Testing with caching ENABLED:")
    print("-" * 30)
    
    cache_times = []
    for i, text in enumerate(test_texts):
        print(f"Test {i+1}: {len(text)} characters")
        start_time = time.time()
        
        output_file = model.generate_speech(
            text=text,
            speaker_audio_file="2_20231227_1742037149.wav",
            language="Tiếng Việt",
            batch_size=6,
            enable_caching=True,
            verbose=False
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        cache_times.append(processing_time)
        
        print(f"  ⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"  📄 Output: {output_file}")
        
        # Test repeat generation (should be much faster with cache)
        if i == 0:  # Test caching on first text
            start_repeat = time.time()
            repeat_output = model.generate_speech(
                text=text,
                speaker_audio_file="2_20231227_1742037149.wav",
                language="Tiếng Việt",
                batch_size=6,
                enable_caching=True,
                verbose=False
            )
            repeat_time = time.time() - start_repeat
            print(f"  🔄 Repeat time (cached): {repeat_time:.2f} seconds")
            print(f"  📈 Speedup: {processing_time/repeat_time:.2f}x faster")
    
    # Clear cache and test without caching
    print(f"\n🚫 Testing with caching DISABLED:")
    print("-" * 30)
    model.clear_cache()
    
    no_cache_times = []
    for i, text in enumerate(test_texts[:2]):  # Test fewer samples to save time
        print(f"Test {i+1}: {len(text)} characters")
        start_time = time.time()
        
        output_file = model.generate_speech(
            text=text,
            speaker_audio_file="2_20231227_1742037149.wav",
            language="Tiếng Việt",
            batch_size=1,  # Single batch processing
            enable_caching=False,
            verbose=False
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        no_cache_times.append(processing_time)
        
        print(f"  ⏱️  Processing time: {processing_time:.2f} seconds")
    
    # Performance summary
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Model initialization time: {init_time:.2f}s")
    print(f"Average time with caching: {sum(cache_times)/len(cache_times):.2f}s")
    print(f"Average time without caching: {sum(no_cache_times)/len(no_cache_times):.2f}s")
    if no_cache_times:
        improvement = (sum(no_cache_times)/len(no_cache_times)) / (sum(cache_times[:len(no_cache_times)])/len(no_cache_times[:len(cache_times)]))
        print(f"Performance improvement: {improvement:.2f}x faster with optimizations")
    
    # Cache info
    cache_info = model.get_cache_info()
    print(f"\n💾 Cache Information:")
    print(f"  Conditioning cache entries: {cache_info['conditioning_cache_size']}")
    print(f"  Text cache entries: {cache_info['text_cache_size']}")
    if torch.cuda.is_available():
        print(f"  GPU memory allocated: {cache_info['gpu_memory_allocated']/1024**2:.1f} MB")
    
    print(f"\n✅ Performance test completed!")

if __name__ == "__main__":
    test_performance()