"""
Test script to verify that microphone is not re-initialized when calling init_avatar multiple times
"""

import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_microphone_reinitialization():
    """Test that microphone is not re-initialized on multiple init_avatar calls"""
    
    print("🧪 Testing Microphone Re-initialization Prevention")
    print("=" * 60)
    
    try:
        from apis.virtual_avatar_service import VirtualAvatarService
        
        # Create service instance
        print("⏳ Creating VirtualAvatarService instance...")
        avatar_service = VirtualAvatarService()
        print("✅ Service instance created")
        
        # Prepare test data
        test_image_path = "wav2lip_sample/tom.jpg"
        test_audio_path = "identify_sample/ta.wav"
        
        if not os.path.exists(test_image_path):
            print(f"❌ Test image not found: {test_image_path}")
            return
        
        if not os.path.exists(test_audio_path):
            print(f"❌ Test audio not found: {test_audio_path}")
            return
        
        # Read test data
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        with open(test_audio_path, "rb") as f:
            audio_data = f.read()
        
        print(f"✅ Test data loaded: {len(image_data)} bytes image, {len(audio_data)} bytes audio")
        
        # Test 1: First initialization
        print("\n🎯 Test 1: First init_avatar call...")
        success1 = avatar_service.init_avatar(image_data, audio_data)
        mic_initialized_first = avatar_service.microphone_initialized
        print(f"   Result: {'✅ Success' if success1 else '❌ Failed'}")
        print(f"   Microphone initialized: {'✅ Yes' if mic_initialized_first else '❌ No'}")
        
        if avatar_service.virtual_mic:
            stream_active_first = avatar_service.virtual_mic.is_streaming
            print(f"   Microphone streaming: {'✅ Yes' if stream_active_first else '❌ No'}")
        
        time.sleep(2)  # Wait a bit
        
        # Test 2: Second initialization (should not re-initialize microphone)
        print("\n🎯 Test 2: Second init_avatar call (should skip mic re-init)...")
        success2 = avatar_service.init_avatar(image_data, audio_data)
        mic_initialized_second = avatar_service.microphone_initialized
        print(f"   Result: {'✅ Success' if success2 else '❌ Failed'}")
        print(f"   Microphone initialized: {'✅ Yes' if mic_initialized_second else '❌ No'}")
        
        if avatar_service.virtual_mic:
            stream_active_second = avatar_service.virtual_mic.is_streaming
            print(f"   Microphone streaming: {'✅ Yes' if stream_active_second else '❌ No'}")
        
        time.sleep(2)  # Wait a bit
        
        # Test 3: Third initialization (should still skip mic re-init)
        print("\n🎯 Test 3: Third init_avatar call (should still skip mic re-init)...")
        success3 = avatar_service.init_avatar(image_data, audio_data)
        mic_initialized_third = avatar_service.microphone_initialized
        print(f"   Result: {'✅ Success' if success3 else '❌ Failed'}")
        print(f"   Microphone initialized: {'✅ Yes' if mic_initialized_third else '❌ No'}")
        
        if avatar_service.virtual_mic:
            stream_active_third = avatar_service.virtual_mic.is_streaming
            print(f"   Microphone streaming: {'✅ Yes' if stream_active_third else '❌ No'}")
        
        # Summary
        print("\n📊 Test Summary:")
        print("=" * 40)
        
        if success1 and success2 and success3:
            print("✅ All init_avatar calls succeeded")
        else:
            print("❌ Some init_avatar calls failed")
        
        if mic_initialized_first and mic_initialized_second and mic_initialized_third:
            print("✅ Microphone remained initialized throughout all calls")
        else:
            print("❌ Microphone initialization state was inconsistent")
        
        if (avatar_service.virtual_mic and 
            avatar_service.virtual_mic.is_streaming):
            print("✅ Microphone is still streaming after multiple calls")
        else:
            print("❌ Microphone streaming was disrupted")
        
        print("\n💡 Expected behavior:")
        print("   - First call should initialize the microphone")
        print("   - Subsequent calls should skip microphone re-initialization")
        print("   - Microphone should remain streaming throughout")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        avatar_service.cleanup()
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(f"📋 Details: {traceback.format_exc()}")

if __name__ == "__main__":
    test_microphone_reinitialization()
