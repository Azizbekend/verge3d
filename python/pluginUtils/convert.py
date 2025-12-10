from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

import os, platform, subprocess, sys, tempfile

from .path import getRoot, getPlatformBinDirName
from .log import printLog

COMPRESSION_THRESHOLD = 3

try:
    from subprocess import CompletedProcess
except ImportError:
    # COMPAT: Python 2
    class CompletedProcess:

        def __init__(self, args, returncode, stdout=None, stderr=None):
            self.args = args
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

        def check_returncode(self):
            if self.returncode != 0:
                err = subprocess.CalledProcessError(self.returncode, self.args, output=self.stdout)
                raise err
            return self.returncode

    def sp_run(*popenargs, **kwargs):
        input = kwargs.pop('input', None)
        check = kwargs.pop('handle', False)

        capture_output = kwargs.pop('capture_output', False)
        if capture_output:
            kwargs['stdout'] = subprocess.PIPE
            kwargs['stderr'] = subprocess.PIPE

        if input is not None:
            if 'stdin' in kwargs:
                raise ValueError('stdin and input arguments may not both be used.')
            kwargs['stdin'] = subprocess.PIPE
        process = subprocess.Popen(*popenargs, **kwargs)
        try:
            outs, errs = process.communicate(input)
        except:
            process.kill()
            process.wait()
            raise
        returncode = process.poll()
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, popenargs, output=outs)
        return CompletedProcess(popenargs, returncode, stdout=outs, stderr=errs)

    subprocess.run = sp_run

def runCMD(params):
    if platform.system().lower() == 'windows':
        # disable popup console window
        si = subprocess.STARTUPINFO()
        si.dwFlags = subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        app = subprocess.run(params, capture_output=True, startupinfo=si)
    else:
        app = subprocess.run(params, capture_output=True)

    return app


class CompressionFailed(Exception):
    pass

def removeICCChunk(srcPath):
    import pypng.png

    def removeChunksGen(chunks, delete):
        for type, v in chunks:
            if type.decode('ascii') in delete:
                continue
            yield type, v

    try:
        tmpImg = tempfile.NamedTemporaryFile(delete=False)

        reader = pypng.png.Reader(srcPath)
        chunks = removeChunksGen(reader.chunks(), ['iCCP'])
        pypng.png.write_chunks(tmpImg, chunks)

        tmpImg.close()
        dstPath = tmpImg.name

        return dstPath

    except Exception as e:
        printLog('WARNING', 'ICC chunk removal failed\n' + str(e))
        return None

def compressKTX2(srcPath='', srcData=None, dstPath='-', method='AUTO'):
    """
    srcPath/srcData are mutually exclusive
    """

    if srcData:
        # NOTE: toktx does not support stdin at the moment
        tmpImg = tempfile.NamedTemporaryFile(delete=False)
        tmpImg.write(srcData)
        tmpImg.close()
        srcPath = tmpImg.name

    params = [os.path.join(getRoot(), 'ktx', getPlatformBinDirName(), 'toktx')]

    params.append('--encode')
    if method == 'UASTC' or method == 'AUTO':
        params.append('uastc')
        params.append('--zcmp')
    else:
        params.append('etc1s')
        params.append('--clevel')
        params.append('2')
        params.append('--qlevel')
        params.append('255')

    params.append('--genmipmap')
    params.append(dstPath)
    params.append(srcPath)

    printLog('INFO', 'Compressing {0} to {1}'.format(os.path.basename(srcPath), params[2].upper()))

    app = runCMD(params)

    if app.stderr:
        msg = app.stderr.decode('utf-8').strip()

        if 'PNG file has an ICC profile chunk' in msg:
            printLog('WARNING', 'PNG with ICC profile chunk detected, stripping the chunk')

            srcPathRemICC = removeICCChunk(srcPath)

            if srcPathRemICC is not None:
                # replace src path and run compression again
                params[-1] = srcPathRemICC
                app = runCMD(params)

                if app.stderr:
                    msg = app.stderr.decode('utf-8').strip()
                else:
                    msg = 'Successfully compressed PNG with ICC profile chunk removed'

                os.unlink(srcPathRemICC)

        printLog('WARNING', msg)

        # allow non-critical warnings
        if app.returncode > 0:
            if srcData:
                os.unlink(srcPath)
            raise CompressionFailed

    if srcData:
        os.unlink(srcPath)

    if method == 'AUTO':
        if srcData:
            srcSize = len(srcData)
        else:
            srcSize = os.path.getsize(srcPath)

        if dstPath == '-':
            dstSize = len(app.stdout)
        else:
            dstSize = os.path.getsize(dstPath)

        if dstSize > COMPRESSION_THRESHOLD * srcSize:
            printLog('WARNING', 'Compressed image is too large, keeping original file as is')

            if dstPath != '-':
                os.unlink(dstPath)

            raise CompressionFailed

    return app.stdout
