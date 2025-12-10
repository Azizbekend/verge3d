#!/usr/bin/env python

import ctypes, os, pathlib, random, shutil, re, sys, winreg

SUPPORTED_MAX_VERSIONS = ['22.0', '23.0', '24.0', '25.0', '26.0']
SUPPORTED_LOCALES = ['en-US', 'fr-FR', 'de-DE', 'zh-CN', 'ko-KR', 'ja-JP', 'pt-BR']

baseDir = os.path.dirname(os.path.abspath(__file__))

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def traverseMaxDirs(doInstall=True):
    for maxVer in SUPPORTED_MAX_VERSIONS:

        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\\Autodesk\\3dsMax\\' + maxVer)
        except OSError:
            continue

        try:
            maxDir, _ =  winreg.QueryValueEx(key, 'Installdir')
        except OSError:
            continue

        for maxLoc in SUPPORTED_LOCALES:

            srcDir = os.path.join(baseDir, '..', 'ribbon')
            dstDir = os.path.join(maxDir, maxLoc, 'UI', 'Ribbon', 'Extensions')

            if os.path.exists(dstDir):
                if doInstall:
                    print('Copy ribbon files to {0}'.format(dstDir))

                    for d, subdirs, files in os.walk(srcDir):
                        for f in files:
                            srcFile = os.path.join(d, f)
                            dstFile = os.path.join(dstDir, os.path.relpath(srcFile, srcDir))
                            shutil.copyfile(srcFile, dstFile)

                else:
                    print('Remove ribbon files from {0}'.format(dstDir))

                    for d, subdirs, files in os.walk(srcDir):
                        for f in files:
                            srcFile = os.path.join(d, f)
                            dstFile = os.path.join(dstDir, os.path.relpath(srcFile, srcDir))
                            try:
                                os.remove(dstFile)
                            except FileNotFoundError:
                                pass


if __name__ == '__main__':
    if is_admin():
        if len(sys.argv) <= 1 or sys.argv[1].lower() == 'install':
            traverseMaxDirs(True)
        elif len(sys.argv) > 1 and sys.argv[1].lower() == 'uninstall':
            traverseMaxDirs(False)
        else:
            print('Wrong script arguments!')
    else:
        print('Insufficient priviledges! Please run this script as admin!')

