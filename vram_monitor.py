#!/usr/bin/env python3
"""
VRAM monitoring and optimization tool for XTTS model.
Helps track and maximize GPU memory utilization for better performance.
"""

import torch
import time
import requests
import json
from XTTS import XTTS

def monitor_vram_usage():
    """Monitor VRAM usage in real-time."""
    print("🖥️  VRAM Monitoring Tool")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This tool requires a GPU.")
        return
    
    # Initialize model to see base memory usage
    print("📋 Initializing XTTS model...")
    model = XTTS()
    
    def print_vram_info():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB  
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free = total - reserved
        utilization = (reserved / total) * 100
        
        print(f"💾 VRAM Status:")
        print(f"   Allocated: {allocated:.2f}GB")
        print(f"   Reserved:  {reserved:.2f}GB") 
        print(f"   Total:     {total:.2f}GB")
        print(f"   Free:      {free:.2f}GB")
        print(f"   Usage:     {utilization:.1f}%")
        print("-" * 30)
        
        return {
            "allocated": allocated,
            "reserved": reserved, 
            "total": total,
            "free": free,
            "utilization": utilization
        }
    
    # Base memory usage
    print("\n🏠 Base Model Memory Usage:")
    base_stats = print_vram_info()
    
    # Test with different text lengths to see memory scaling
    test_texts = [
        "Câu ngắn để test.",
        "Đây là một đoạn văn bản trung bình để kiểm tra việc sử dụng VRAM khi xử lý text-to-speech với nhiều từ hơn.",
        """Đây là một đoạn văn bản rất dài để kiểm tra khả năng sử dụng VRAM tối đa của mô hình. 
        Chúng ta sẽ xem mô hình sử dụng bao nhiêu VRAM khi xử lý các đoạn văn phức tạp với nhiều câu.
        Việc tối ưu hóa VRAM rất quan trọng để tăng tốc độ xử lý và cho phép xử lý batch size lớn hơn.
        Với việc sử dụng VRAM hiệu quả, chúng ta có thể giảm thời gian xử lý từ vài giây xuống còn milliseconds."""
    ]
    
    print("\n🚀 Testing VRAM Usage with Different Text Lengths:")
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {len(text)} characters")
        print(f"Text preview: {text[:80]}...")
        
        start_time = time.time()
        
        # Run inference
        output_file = model.generate_speech(
            text=text,
            speaker_audio_file="2_20231227_1742037149.wav",
            language="Tiếng Việt", 
            verbose=False,
            max_vram_usage=True
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Check memory usage after processing
        stats = print_vram_info()
        
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📈 VRAM increase: {stats['reserved'] - base_stats['reserved']:.2f}GB")
        print(f"📄 Output: {output_file}")
        
        # Get cache info
        cache_info = model.get_cache_info()
        print(f"💾 Cache entries: {cache_info['conditioning_cache_size']} conditioning, {cache_info['text_cache_size']} text")
        print(f"🔧 Batch mode: {'ON' if cache_info.get('large_batch_mode', False) else 'OFF'}")
        
    # Test maximum VRAM utilization
    print(f"\n🔥 MAXIMUM VRAM UTILIZATION TEST")
    print("=" * 40)
    
    # Try to push VRAM usage higher with large text
    large_text = test_texts[2] * 3  # Triple the longest text
    
    print(f"Processing extra large text: {len(large_text)} characters")
    
    start_time = time.time()
    output_file = model.generate_speech(
        text=large_text,
        speaker_audio_file="2_20231227_1742037149.wav",
        language="Tiếng Việt",
        verbose=True,  # Show detailed info
        max_vram_usage=True
    )
    end_time = time.time()
    
    print(f"\n🏁 FINAL VRAM STATISTICS:")
    final_stats = print_vram_info()
    
    print(f"📊 Performance Summary:")
    print(f"   Max VRAM used: {final_stats['reserved']:.2f}GB ({final_stats['utilization']:.1f}%)")
    print(f"   Processing time: {end_time - start_time:.2f}s")
    print(f"   VRAM efficiency: {final_stats['utilization']:.1f}% utilization")
    
    # Recommendations
    if final_stats['utilization'] < 70:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   • VRAM utilization is {final_stats['utilization']:.1f}% - you can use larger batch sizes")
        print(f"   • Available free VRAM: {final_stats['free']:.2f}GB")
        print(f"   • Consider increasing max_cache_size in XTTS.py")
        print(f"   • Enable large_batch_mode for better performance")
    elif final_stats['utilization'] > 90:
        print(f"\n⚠️  WARNINGS:")
        print(f"   • High VRAM usage: {final_stats['utilization']:.1f}%")
        print(f"   • Risk of out-of-memory errors")
        print(f"   • Consider reducing batch size or cache size")
    else:
        print(f"\n✅ OPTIMAL:")
        print(f"   • Good VRAM utilization: {final_stats['utilization']:.1f}%")
        print(f"   • Balanced performance and stability")
    
    # Clean up
    model.clear_cache()
    torch.cuda.empty_cache()
    print(f"\n🧹 Cache cleared, memory freed")

def test_api_vram_monitoring():
    """Test VRAM monitoring through the API."""
    try:
        response = requests.get("http://localhost:8009/cache-info")
        if response.status_code == 200:
            info = response.json()
            print("\n🌐 API VRAM Information:")
            print(json.dumps(info, indent=2))
        else:
            print("❌ Could not connect to API. Make sure main.py is running.")
    except Exception as e:
        print(f"❌ API connection error: {e}")

if __name__ == "__main__":
    print("🎯 Choose monitoring option:")
    print("1. Full VRAM monitoring test")
    print("2. API VRAM info only")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        monitor_vram_usage()
    elif choice == "2":
        test_api_vram_monitoring()
    else:
        print("❌ Invalid choice. Running full test...")
        monitor_vram_usage()