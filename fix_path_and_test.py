import os
import subprocess
import sys
from pathlib import Path

def fix_ffmpeg_path():
    """修復 FFmpeg PATH 設定"""
    print("🔧 修復 FFmpeg PATH 設定")
    print("=" * 30)
    
    ffmpeg_bin_path = "C:\\ffmpeg\\bin"
    ffmpeg_exe_path = "C:\\ffmpeg\\bin\\ffmpeg.exe"
    
    # 1. 檢查檔案是否存在
    if not os.path.exists(ffmpeg_exe_path):
        print(f"❌ FFmpeg 執行檔不存在: {ffmpeg_exe_path}")
        return False
    
    print(f"✅ FFmpeg 執行檔存在: {ffmpeg_exe_path}")
    
    # 2. 檢查當前 PATH
    current_path = os.environ.get('PATH', '')
    
    if ffmpeg_bin_path in current_path:
        print("✅ FFmpeg 已在 PATH 中")
    else:
        print("❌ FFmpeg 不在 PATH 中，正在添加...")
        
        # 臨時添加到當前會話
        new_path = f"{ffmpeg_bin_path};{current_path}"
        os.environ['PATH'] = new_path
        print(f"✅ 已添加到當前會話 PATH")
    
    # 3. 測試 FFmpeg
    try:
        result = subprocess.run([ffmpeg_exe_path, '-version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg 測試成功: {version_line}")
            return True
        else:
            print(f"❌ FFmpeg 測試失敗: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ FFmpeg 測試錯誤: {e}")
        return False

def permanently_add_to_path():
    """永久添加到系統 PATH"""
    print("\n🔧 永久添加到系統 PATH")
    print("=" * 30)
    
    try:
        import winreg
        
        ffmpeg_bin_path = "C:\\ffmpeg\\bin"
        
        # 讀取當前使用者的 PATH
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS) as key:
            try:
                current_path, _ = winreg.QueryValueEx(key, "PATH")
            except FileNotFoundError:
                current_path = ""
            
            # 檢查是否已存在
            if ffmpeg_bin_path in current_path:
                print("✅ FFmpeg 已在系統 PATH 中")
                return True
            
            # 添加新路徑
            new_path = f"{current_path};{ffmpeg_bin_path}" if current_path else ffmpeg_bin_path
            winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
            
            print("✅ 已永久添加到系統 PATH")
            print("⚠️ 需要重新啟動 PowerShell 才能生效")
            return True
            
    except Exception as e:
        print(f"❌ 無法修改系統 PATH: {e}")
        print("💡 請手動添加 C:\\ffmpeg\\bin 到系統 PATH")
        return False

def test_ffmpeg_directly():
    """直接測試 FFmpeg"""
    print("\n🔍 直接測試 FFmpeg")
    print("=" * 25)
    
    ffmpeg_exe = "C:\\ffmpeg\\bin\\ffmpeg.exe"
    
    try:
        # 測試版本
        result = subprocess.run([ffmpeg_exe, '-version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ FFmpeg 直接測試成功")
            version_line = result.stdout.split('\n')[0]
            print(f"   版本: {version_line}")
            
            # 測試簡單轉換
            print("\n🔄 測試音訊轉換功能...")
            test_cmd = [
                ffmpeg_exe, '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=1',
                '-f', 'wav', '-y', 'test_output.wav'
            ]
            
            result2 = subprocess.run(test_cmd, capture_output=True, text=True, timeout=15)
            
            if result2.returncode == 0:
                print("✅ 音訊轉換功能正常")
                # 清理測試檔案
                if os.path.exists('test_output.wav'):
                    os.remove('test_output.wav')
                return True
            else:
                print(f"❌ 音訊轉換失敗: {result2.stderr}")
                return False
        else:
            print(f"❌ FFmpeg 直接測試失敗: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 直接測試錯誤: {e}")
        return False

def update_identify_py():
    """更新 identify.py 以強制使用 FFmpeg"""
    print("\n🔧 更新 identify.py 以強制使用 FFmpeg")
    print("=" * 45)
    
    identify_file = "apis/identify.py"
    
    if not os.path.exists(identify_file):
        print(f"❌ 找不到 {identify_file}")
        return False
    
    # 讀取現有檔案
    with open(identify_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修改 _check_ffmpeg 方法
    new_check_ffmpeg = '''    def _check_ffmpeg(self) -> bool:
        """檢查系統是否安裝了 ffmpeg (強化版)"""
        
        # 方法 1: 檢查固定路徑
        ffmpeg_paths = [
            "C:\\\\ffmpeg\\\\bin\\\\ffmpeg.exe",
            "ffmpeg",  # 系統 PATH 中的 ffmpeg
            "C:\\\\Program Files\\\\ffmpeg\\\\bin\\\\ffmpeg.exe",
            "C:\\\\Program Files (x86)\\\\ffmpeg\\\\bin\\\\ffmpeg.exe"
        ]
        
        for ffmpeg_path in ffmpeg_paths:
            try:
                result = subprocess.run([ffmpeg_path, '-version'], 
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    self.ffmpeg_path = ffmpeg_path
                    logger.info(f"找到 FFmpeg: {ffmpeg_path}")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                continue
        
        logger.warning("FFmpeg 未找到")
        return False'''
    
    # 替換方法
    import re
    pattern = r'def _check_ffmpeg\(self\) -> bool:.*?return False'
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_check_ffmpeg.strip(), content, flags=re.DOTALL)
        
        # 在 __init__ 方法中添加 ffmpeg_path 屬性
        if 'self.ffmpeg_path = None' not in content:
            init_pattern = r'(self\.ffmpeg_available = self\._check_ffmpeg\(\))'
            replacement = r'self.ffmpeg_path = None\n        \1'
            content = re.sub(init_pattern, replacement, content)
        
        # 修改 _convert_with_ffmpeg 方法使用 self.ffmpeg_path
        convert_pattern = r"cmd = \[\s*'ffmpeg'"
        replacement = "cmd = [getattr(self, 'ffmpeg_path', 'ffmpeg')"
        content = re.sub(convert_pattern, replacement, content)
        
        # 保存修改後的檔案
        with open(identify_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 已更新 {identify_file}")
        return True
    else:
        print(f"❌ 無法找到 _check_ffmpeg 方法")
        return False

if __name__ == "__main__":
    print("🎬 FFmpeg 修復工具")
    print("=" * 25)
    
    # 1. 修復 PATH
    if fix_ffmpeg_path():
        print("\n✅ FFmpeg PATH 修復成功")
        
        # 2. 永久添加到 PATH
        permanently_add_to_path()
        
        # 3. 直接測試
        if test_ffmpeg_directly():
            print("\n✅ FFmpeg 功能正常")
            
            # 4. 更新 identify.py
            if update_identify_py():
                print("\n✅ identify.py 已更新")
                print("\n🎉 修復完成！現在可以測試語者辨識了")
                print("📋 執行: python test_speaker_identification.py")
            else:
                print("\n⚠️ identify.py 更新失敗，但 FFmpeg 應該可用")
        else:
            print("\n❌ FFmpeg 功能測試失敗")
    else:
        print("\n❌ FFmpeg PATH 修復失敗")