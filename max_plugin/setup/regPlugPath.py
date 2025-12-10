#!/usr/bin/env python

import os, pathlib, random, shutil, re, sys

SUPPORTED_MAX_VERSIONS = [2020, 2021, 2022, 2023, 2024]
SUPPORTED_LOCALES = ['ENU', 'FRA', 'DEU', 'CHS', 'KOR', 'JPN', 'PTB']

baseDir = os.path.dirname(os.path.abspath(__file__))
lib = os.path.join(baseDir, '..', '..', 'python')
sys.path.append(lib)

import iniparse

def registerPlugin(ini):
    pathChanged = False
    cfg = None

    with open(ini, 'r', encoding='utf_16') as f:
        pluginDir = os.path.normpath(os.path.join(baseDir, '..'))

        cfg = iniparse.INIConfig(f)
        dirSect = cfg['Directories']

        # manually assigned
        for d in dirSect:
            if '\\max_plugin\\' in dirSect[d]:
                dirSect[d] = pluginDir + '\\'
                pathChanged = True

        if not pathChanged:
            dirSect['Verge3D for 3ds Max plug-in'] = pluginDir + '\\'
            pathChanged = True

    if pathChanged:
        with open(ini, 'w', encoding='utf_16') as f:
            f.write(str(cfg))


def unregisterPlugin(ini):
    pathChanged = False
    cfg = None

    with open(ini, 'r', encoding='utf_16') as f:
        pluginDir = os.path.normpath(os.path.join(baseDir, '..'))

        cfg = iniparse.INIConfig(f)
        dirSect = cfg['Directories']

        for d in dirSect:

            if 'Verge3D' in d:
                del dirSect[d]
                pathChanged = True

            # manually assigned
            if not pathChanged and '\\max_plugin\\' in dirSect[d]:
                del dirSect[d]
                pathChanged = True

    if pathChanged:
        with open(ini, 'w', encoding='utf_16') as f:
            f.write(str(cfg))


def traverseMaxDirs(doInstall=True):
    maxDir = os.path.expandvars(r'%LOCALAPPDATA%\Autodesk\3dsMax')
    if os.path.exists(maxDir):
        for maxVer in SUPPORTED_MAX_VERSIONS:
            maxVerDir = os.path.join(maxDir, str(maxVer) + ' - 64bit')
            for maxLoc in SUPPORTED_LOCALES:
                plugsIni = os.path.join(maxVerDir, maxLoc, 'Plugin.UserSettings.ini')
                if os.path.exists(plugsIni):

                    if doInstall:
                        print('Registering plugin path for Max {0} {1}'.format(maxVer, maxLoc))
                        registerPlugin(plugsIni)
                    else:
                        print('Unregistering plugin path for Max {0} {1}'.format(maxVer, maxLoc))
                        unregisterPlugin(plugsIni)

                if doInstall:
                    # cleanup possible remove unregister script
                    removeMenuScript = os.path.join(maxVerDir, maxLoc, 'scripts', 'startup', 'Verge3D-removeMenu.ms')
                    if os.path.exists(removeMenuScript):
                        os.remove(removeMenuScript)

                    # generate new Verge3D menu context
                    v3dMain = os.path.join(baseDir, '..', 'verge3d.ms')
                    content = ''

                    with open(v3dMain, 'r', newline='', encoding='utf-8') as fin:
                        content = fin.read()

                    content = re.sub('V3D_MENU_CONTEXT\s*=\s*0x\w+',
                                     'V3D_MENU_CONTEXT = ' + hex(random.randrange(2**32)), content)

                    with open(v3dMain, 'w', newline='', encoding='utf-8') as fout:
                        fout.write(content)

                else:
                    settings = os.path.join(maxVerDir, maxLoc, 'scripts', 'Verge3D')
                    if os.path.exists(settings):
                        print('Removing {}'.format(settings))
                        shutil.rmtree(settings)

                    macros = os.path.join(maxVerDir, maxLoc, 'usermacros')
                    for p in pathlib.Path(macros).glob('Verge3D-*'):
                        print('Removing {}'.format(str(p)))
                        os.remove(p)

                    scripts = os.path.join(maxVerDir, maxLoc, 'scripts', 'startup')
                    if os.path.exists(scripts):
                        print('Placing menu unregister script')
                        shutil.copy(os.path.join(baseDir, 'Verge3D-removeMenu.ms.template'),
                                    os.path.join(scripts, 'Verge3D-removeMenu.ms'))


if __name__ == '__main__':
    if len(sys.argv) <= 1 or sys.argv[1].lower() == 'install':
        traverseMaxDirs(True)
    elif len(sys.argv) > 1 and sys.argv[1].lower() == 'uninstall':
        traverseMaxDirs(False)
    else:
        print('Wrong script arguments')
