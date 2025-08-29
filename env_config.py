"""
環境配置模組 - 解決常見的 Python 套件衝突問題
"""

import os
import warnings

# 解決 OpenMP 重複載入問題
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 解決其他常見的環境問題
os.environ['OMP_NUM_THREADS'] = '1'  # 限制 OpenMP 線程數，避免過度使用資源
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免 tokenizers 的並行警告

# 抑制一些常見的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

print("✅ 環境配置已載入，解決了以下問題:")
print("   - OpenMP 重複載入衝突")
print("   - 線程數限制優化")
print("   - 抑制非關鍵警告")
