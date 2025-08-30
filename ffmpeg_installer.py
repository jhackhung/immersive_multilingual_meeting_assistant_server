import subprocess
import sys
import os
import requests
import zipfile
import shutil
from pathlib import Path

def install_ffmpeg_automatic():
    """自動下載並安裝 FFmpeg"""
    print("自動安裝 FFmpeg for Windows")
    print("=" * 40)
    
    # 方法 1: 嘗試 winget
    if install_with_winget():
        return True
    
    # 方法 2: 嘗試 chocolatey
    if install_with_chocolatey():
        return True
    
    # 方法 3: 手動下載安裝
    if install_manually():
        return True
    
    print("自動安裝失敗，請使用手動安裝")
    return False

def install_with_winget():
    """使用 Windows Package Manager 安裝"""
    print("\n嘗試使用 Winget 安裝...")
    
    try:
        # 檢查 winget 是否可用
        result = subprocess.run(['winget', '--version'], 
                              capture_output=True, timeout=10)
        
        if result.returncode != 0:
            print("Winget 不可用")
            return False
        
        print("找到 Winget，開始安裝...")
        
        # 安裝 FFmpeg
        result = subprocess.run(['winget', 'install', 'Gyan.FFmpeg'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("Winget 安裝成功！")
            return check_ffmpeg_installation()
        else:
            print(f"Winget 安裝失敗: {result.stderr}")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Winget 安裝錯誤: {e}")
        return False

def install_with_chocolatey():
    """使用 Chocolatey 安裝"""
    print("\n嘗試使用 Chocolatey 安裝...")
    
    try:
        # 檢查 chocolatey 是否可用
        result = subprocess.run(['choco', '--version'], 
                              capture_output=True, timeout=10)
        
        if result.returncode != 0:
            print("Chocolatey 不可用")
            return False
        
        print("找到 Chocolatey，開始安裝...")
        
        # 安裝 FFmpeg
        result = subprocess.run(['choco', 'install', 'ffmpeg', '-y'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("Chocolatey 安裝成功！")
            return check_ffmpeg_installation()
        else:
            print(f"Chocolatey 安裝失敗: {result.stderr}")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Chocolatey 安裝錯誤: {e}")
        return False

def install_manually():
    """手動下載並安裝 FFmpeg"""
    print("\n手動下載安裝 FFmpeg...")
    
    try:
        # FFmpeg 下載 URL (Gyan 編譯版本)
        download_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        
        # 創建下載目錄
        download_dir = Path.home() / "Downloads" / "ffmpeg"
        download_dir.mkdir(parents=True, exist_ok=True)
        
        zip_file = download_dir / "ffmpeg.zip"
        extract_dir = download_dir / "extracted"
        
        print(f"下載到: {download_dir}")
        print("開始下載 FFmpeg (約 50MB)...")
        
        # 下載檔案
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r下載進度: {percent:.1f}%", end="", flush=True)
        
        print(f"\n下載完成: {zip_file}")
        
        # 解壓縮
        print("解壓縮檔案...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # 找到 FFmpeg 執行檔
        ffmpeg_dirs = list(extract_dir.glob("ffmpeg-*"))
        if not ffmpeg_dirs:
            print("找不到 FFmpeg 目錄")
            return False
        
        ffmpeg_dir = ffmpeg_dirs[0]
        ffmpeg_bin = ffmpeg_dir / "bin"
        
        if not ffmpeg_bin.exists():
            print("找不到 bin 目錄")
            return False
        
        # 安裝到系統
        install_dir = Path("C:/ffmpeg")
        
        print(f"安裝到: {install_dir}")
        
        # 複製到系統目錄
        if install_dir.exists():
            shutil.rmtree(install_dir)
        
        shutil.copytree(ffmpeg_dir, install_dir)
        
        # 添加到 PATH
        add_to_path(install_dir / "bin")
        
        print("手動安裝完成！")
        
        # 清理下載檔案
        cleanup_download(download_dir)
        
        return check_ffmpeg_installation()
        
    except Exception as e:
        print(f"手動安裝失敗: {e}")
        return False

def add_to_path(bin_path):
    """將 FFmpeg 添加到系統 PATH"""
    print("添加到系統 PATH...")
    
    try:
        import winreg
        
        # 讀取當前 PATH
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS) as key:
            try:
                current_path, _ = winreg.QueryValueEx(key, "PATH")
            except FileNotFoundError:
                current_path = ""
            
            # 檢查是否已存在
            if str(bin_path) in current_path:
                print("PATH 已包含 FFmpeg")
                return True
            
            # 添加新路徑
            new_path = f"{current_path};{bin_path}" if current_path else str(bin_path)
            winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
            
            print("已添加到 PATH")
            print("請重新啟動命令提示字元以生效")
            return True
            
    except Exception as e:
        print(f"添加 PATH 失敗: {e}")
        print("請手動添加以下路徑到系統 PATH:")
        print(f"   {bin_path}")
        return False

def cleanup_download(download_dir):
    """清理下載檔案"""
    try:
        if download_dir.exists():
            shutil.rmtree(download_dir)
            print("🧹 已清理下載檔案")
    except Exception as e:
        print(f"清理失敗: {e}")

def check_ffmpeg_installation():
    """檢查 FFmpeg 安裝"""
    print("\n檢查 FFmpeg 安裝...")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"FFmpeg 安裝成功！")
            print(f"   版本: {version_line}")
            return True
        else:
            print("FFmpeg 無法執行")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print("FFmpeg 未找到")
        print("可能需要重新啟動命令提示字元")
        return False

if __name__ == "__main__":
    print("🎬 FFmpeg Windows 安裝器")
    print("=" * 30)
    
    # 先檢查是否已安裝
    if check_ffmpeg_installation():
        print("FFmpeg 已安裝，無需重新安裝")
        sys.exit(0)
    
    # 開始安裝
    success = install_ffmpeg_automatic()
    
    if success:
        print("\nFFmpeg 安裝完成！")
        print("下一步:")
        print("1. 重新啟動命令提示字元")
        print("2. 執行: python test_speaker_identification.py")
    else:
        print("\n自動安裝失敗")
        print("請參考手動安裝指南")