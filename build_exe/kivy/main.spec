# -*- mode: python ; coding: utf-8 -*-

block_cipher = None



a = Analysis(['main.py'],
             pathex=[],
             binaries=[],
             datas=[('./NanumGothic.ttf', './'),('./2021_06_02_09_47_46.h5', './'), ('./testmedia.py', './'), ('./main.kv', './'), ('./libpng16-16.dll', './'),('C:\\Users\\kkeee\\anaconda3\\envs\\blinkeye\\Lib\\site-packages\\mediapipe\\modules','mediapipe\\modules'),],
             hiddenimports=['plyer.platforms.win.notification'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe, 
               a.binaries,
               a.zipfiles,
               a.datas, 
               
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main')
