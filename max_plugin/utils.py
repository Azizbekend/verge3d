from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

import math, os

from pluginUtils.log import printLog
from pluginUtils.gltf import appendExtension

import maxUtils

ORM_PREFIX = '__ORM__'
BASE_ALPHA_PREFIX = '__BASE_ALPHA__'

"""
Various glTF and debugging utility functions
"""

def defaultMatName(node):
    return '__DEFAULT__' + str(abs(maxUtils.getPtr(node)))

def baseAlphaTexName(matName):
    return BASE_ALPHA_PREFIX + matName

def ormTexName(matName):
    return ORM_PREFIX + matName

def nodeIsLamp(node):
    return ('extensions' in node and 'S8S_v3d_lights' in node['extensions']
            and 'light' in node['extensions']['S8S_v3d_lights'])

def nodeIsCurve(node):
    return ('extensions' in node and 'S8S_v3d_curves' in node['extensions']
            and 'curve' in node['extensions']['S8S_v3d_curves'])

def nodeIsReflProbe(node):
    return ('extensions' in node and 'S8S_v3d_light_probes' in node['extensions']
            and 'lightProbe' in node['extensions']['S8S_v3d_light_probes'])


def integerToMaxSuffix(val):

    suf = str(val)

    for i in range(0, 3 - len(suf)):
        suf = '0' + suf

    return suf

def maxShininessToHardness(shininess):
    """
    see maxsdk/samples/materials/stdshaders.cpp
    """
    return 4.0 * (2.0 ** (shininess * 10.0))


def recalcFOV(fov, aspect):
    """
    fov in radians
    aspect = width/height
    """
    return 2 * math.atan(math.tan( fov / 2 ) * aspect)

def calcOrthoScales(fov, dist, aspect):
    """
    fov in radians
    aspect = width/height
    """
    xmag = math.tan(fov/2) * dist
    ymag = xmag / aspect

    return (xmag, ymag)


def getByNameID(list, idname):
    """
    Return element by the given name or ID
    """

    if list is None or idname is None:
        return None

    for element in list:
        if ((isinstance(idname, int) and element.get('id') == idname) or
                element.get('name') == idname):
            return element

    return None

def getParentNode(gltf, nodeIndex):
    if not gltf.get('nodes'):
        return -1

    index = 0

    for node in gltf['nodes']:
        if node.get('children'):
            for childIndex in node['children']:
                if childIndex == nodeIndex:
                    return index
        index += 1

    return -1

