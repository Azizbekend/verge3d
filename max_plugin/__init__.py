from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import fnmatch, os, sys, tempfile, threading, time, shutil, webbrowser

join = os.path.join
norm = os.path.normpath

sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

from pymxs import runtime as rt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# append the path to the maxcpp module
ver = rt.maxVersion()[0]
if ver <= 22000: libDir = '2020'
elif ver == 23000: libDir = '2021'
elif ver == 24000: libDir = '2022'
elif ver == 25000: libDir = '2023'
else: libDir = '2024'
sys.path.append(join(os.path.dirname(os.path.abspath(__file__)), libDir))

if sys.version_info[0] > 2:
    import importlib
else:
    # COMPAT: < 2021
    import imp as importlib

if 'log' in locals():
    importlib.reload(log)
if 'utils' in locals():
    importlib.reload(utils)
if 'extract' in locals():
    importlib.reload(extract)
if 'collect' in locals():
    importlib.reload(collect)
if 'generate' in locals():
    importlib.reload(generate)
if 'maxUtils' in locals():
    importlib.reload(maxUtils)
if 'progressDialog' in locals():
    importlib.reload(progressDialog)

from pluginUtils.log import printLog
from pluginUtils.manager import AppManagerConn
from pluginUtils.path import getAppManagerHost, getRoot, findExportedAssetPath

import collect
import extract
import generate
import progressDialog
import maxUtils
import utils

PING_DELAY_FIRST = 5
PING_DELAY = 2

extractCustomProp = extract.extractCustomProp

_trackBarExportState = {
    'playbackRange': rt.interval(0, 100),
    'playbackTime': 0,
    'setKeyMode': False
}

# prevent export path reset during importlib.reload()
try:
    _currentMaxPath
    _defaultExportPath
except:
    _currentMaxPath = ''
    _defaultExportPath = ''

def setDefaultExportPath(path):
    global _defaultExportPath

    _defaultExportPath = os.path.splitext(path)[0]

def getDefaultExportPath():
    global _currentMaxPath, _defaultExportPath

    maxPath = rt.maxFilePath + rt.maxFileName

    # return cached version
    if _defaultExportPath and maxPath == _currentMaxPath:
        return _defaultExportPath

    _currentMaxPath = maxPath

    return os.path.splitext(maxPath)[0]

def exportGLTF():

    filename = getDefaultExportPath().replace('\\', '\\\\')

    filepath = rt.getSaveFileName(caption='Export glTF:', filename='{}'.format(filename),
            types='glTF file (*.gltf)|*.gltf|glTF binary file (*.glb)|*.glb|')

    if filepath is not None:
        exportGLTFPath(filepath)
        setDefaultExportPath(filepath)

def exportGLTFPath(filepath, overrideFormat='', sneakPeek=False):

    progressDialog.ProgressDialog.open(title='Exporting GLTF...')

    name, ext = os.path.splitext(os.path.basename(filepath))

    if overrideFormat:
        gltfFormat = overrideFormat
    elif ext == '.glb':
        gltfFormat = 'BINARY'
    else:
        gltfFormat = 'ASCII'

    filepathBin = name + '.bin'
    filedir = os.path.dirname(filepath) + '/'

    printLog('INFO', 'Exporting glTF 2.0 asset (' + gltfFormat + ')')

    tmpDir = tempfile.mkdtemp(suffix='verge3d')

    exportPrepare()

    exportSettings = {
        'anim_playback_original_from': _trackBarExportState['playbackRange'].start,
        'anim_playback_original_to': _trackBarExportState['playbackRange'].end,
        'format' : gltfFormat,
        'binary' : bytearray(),
        'filepath' : filepath,
        'binaryfilename' : filepathBin,
        'filedirectory': filedir,
        'strip' : True,
        'uri_cache' : { 'uri': [], 'obj': [] },
        'tmp_dir': tmpDir,
        'sneak_peek': sneakPeek,
        'skinned_mesh_use_aux_bone': True
    }

    try:
        gltf = generate.GLTF(exportSettings)
        collector = collect.Collector(exportSettings)
        collector.collect()

        gltf.generate(collector)
        gltf.save(collector)

        shutil.rmtree(tmpDir)
    except:
        exportCleanup(collector)
        raise
    else:
        exportCleanup(collector)

def exportPrepare():

    animExport = extractCustomProp(rt.rootNode, 'V3DExportSettingsData', 'animExport', True)
    animUsePlaybackRange = extractCustomProp(rt.rootNode, 'V3DExportSettingsData', 'animUsePlaybackRange', False)

    animRange = rt.animationRange

    # create new, do not use reference
    _trackBarExportState['playbackRange'] = rt.interval(animRange.start, animRange.end)

    # convert MXSWrapperBase to float
    _trackBarExportState['playbackTime'] = float(rt.sliderTime)

    _trackBarExportState['setKeyMode'] = rt.animButtonState

    if _trackBarExportState['setKeyMode']:
        rt.animButtonState = False

    if animExport and not animUsePlaybackRange:
        maxUtils.disableViewportRedraw()
        # use the 0 frame for retreiving objects' state, otherwise the current
        # frame will be used
        rt.animationRange = rt.interval(0, 100)
        rt.sliderTime = 0

def exportCleanup(collector=None):

    animExport = extractCustomProp(rt.rootNode, 'V3DExportSettingsData', 'animExport', True)
    animUsePlaybackRange = extractCustomProp(rt.rootNode, 'V3DExportSettingsData', 'animUsePlaybackRange', False)

    if animExport and not animUsePlaybackRange:
        rt.animationRange = _trackBarExportState['playbackRange']
        rt.sliderTime = _trackBarExportState['playbackTime']
        maxUtils.enableViewportRedraw()

    if _trackBarExportState['setKeyMode']:
         rt.animButtonState = True

    if collector:
        for mTex in collector.textures:
            if hasattr(mTex, 'compressionErrorStatus'):
                del mTex.compressionErrorStatus

    progressDialog.ProgressDialog.close()

def reexportAll():
    apps = join(getRoot(), 'applications')

    for root, dirs, files in os.walk(apps):
        for name in files:
            if fnmatch.fnmatch(name, '*.max'):
                maxpath = norm(join(root, name))

                gltfpath = findExportedAssetPath(maxpath)
                if gltfpath:
                    printLog('INFO', 'Reexporting ' + maxpath)
                    rt.loadMaxFile(maxpath, useFileUnits=True, quiet=True)
                    exportGLTFPath(gltfpath)

def execBrowser(url):
    try:
        webbrowser.open(url)
    except BaseException:
        printLog('ERROR', 'Failed to open URL: ' + url)

def runAppManager():
    AppManagerConn.start(getRoot(), 'MAX', False)
    execBrowser(getAppManagerHost())

def runUserManual():
    execBrowser(AppManagerConn.getManualURL())

def sneakPeek():
    # always try to run server before sneak peek
    AppManagerConn.start(getRoot(), 'MAX', False)

    prevDir = AppManagerConn.getPreviewDir(True)

    exportGLTFPath(join(prevDir, 'sneak_peek.gltf'), sneakPeek=True)

    execBrowser(getAppManagerHost() +
            'player/player.html?load=/sneak_peek/sneak_peek.gltf')

def pingAppManager():
    while True:
        if AppManagerConn.ping():
            time.sleep(PING_DELAY)
        else:
            AppManagerConn.start(getRoot(), 'MAX', False)
            time.sleep(PING_DELAY_FIRST)

def init():
    printLog('INFO', 'Initialize Verge3D plugin')

    AppManagerConn.start(getRoot(), 'MAX', False)

    # COMPAT: threads are buggy in Python 2
    if sys.version_info[0] >= 3:
        pingTimer = threading.Timer(PING_DELAY_FIRST, pingAppManager)
        pingTimer.daemon = True
        pingTimer.start()
