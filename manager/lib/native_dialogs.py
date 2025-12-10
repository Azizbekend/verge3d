#!/usr/bin/env python3

import os, platform, subprocess, sys
from ctypes import *

def selectDir(title):
    system = platform.system().lower()

    if system == 'windows':

        BIF_RETURNONLYFSDIRS   = 0x0001
        BIF_DONTGOBELOWDOMAIN  = 0x0002
        BIF_STATUSTEXT         = 0x0004
        BIF_RETURNFSANCESTORS  = 0x0008
        BIF_EDITBOX            = 0x0010
        BIF_VALIDATE           = 0x0020
        BIF_NEWDIALOGSTYLE     = 0x0040
        BIF_BROWSEFORCOMPUTER  = 0x1000
        BIF_BROWSEFORPRINTER   = 0x2000
        BIF_BROWSEINCLUDEFILES = 0x4000

        LPOFNHOOKPROC = c_voidp
        LPCTSTR = LPTSTR = c_wchar_p
        MAX_PATH = 1024

        class BROWSEINFO(Structure):
            _fields_ = [('hwndOwner', wintypes.HWND),
                        ('pidlRoot', c_int),
                        ('pszDisplayName', LPTSTR),
                        ('lpszTitle', LPCTSTR),
                        ('ulFlags', wintypes.UINT),
                        ('lpfn', c_int),
                        ('lParam', wintypes.LPARAM),
                        ('iImage', c_int)]

        oledll.ole32.CoInitialize(None)

        bi = BROWSEINFO()

        buffer = create_unicode_buffer('', MAX_PATH)
        bi.hwndOwner = windll.user32.GetForegroundWindow()
        bi.pszDisplayName = cast(buffer, LPTSTR)
        bi.lpszTitle = title
        bi.ulFlags = BIF_EDITBOX | BIF_NEWDIALOGSTYLE

        windll.shell32.SHBrowseForFolderW.restype = wintypes.HANDLE
        windll.shell32.SHGetPathFromIDListW.argtypes = [wintypes.HANDLE, LPTSTR]

        pidl = windll.shell32.SHBrowseForFolderW(byref(bi))
        if pidl:
            windll.shell32.SHGetPathFromIDListW(pidl, cast(buffer, LPTSTR))
            return buffer.value
        else:
            return ''

    elif system == 'linux':
        try:
            ret = subprocess.run(['zenity', '--file-selection', '--directory', '--title="{}"'.format(title)], capture_output=True)
            if ret.returncode == 0:
                return ret.stdout.decode('utf-8')
            else:
                return ''
        except FileNotFoundError:
            return ''

    elif system == 'darwin':

        # NOTE: Cocoa must be executed from the main thread

        #from pycocoa import NSOpenPanel, NSPrintPanel, NSSavePanel, \
        #                    NSStr, nsString2str, nsLog, py2NS

        #panel = NSOpenPanel.openPanel()

        #panel.setPrompt_(NSStr('Set prompt...'))
        #panel.setCanChooseDirectories_(True)
        #panel.setCanChooseFiles_(False)

        #if panel.runModal():
        #    return nsString2str(panel.URL().path())
        #return ''

        # Using AppleScript instead

        script = 'POSIX path of (choose folder with prompt "{}")'.format(title)

        ret = subprocess.run(['osascript', '-e', script], capture_output=True)
        if ret.returncode == 0:
            print(ret.stdout.decode('utf-8').strip())
            return ret.stdout.decode('utf-8').strip()
        else:
            return ''

    else:
        return ''
