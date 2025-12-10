from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

import math

from pymxs import runtime as rt

import pluginUtils as pu
from pluginUtils.log import printLog

import extract
import maxcpp

DEFAULT_DIST_EMPTY_SCENE = 10
TICKS_PER_SEC = 4800

ILLUMINANT_TO_KELVIN = [
    6500, 5200, 7000, 6000, 3200, 4000,
    # CIE A - https://en.wikipedia.org/wiki/Standard_illuminant
    2856,
    # CIE D50, D55, D65, D75
    5003, 5503, 6504, 7504,
    # CIE F1-F12
    6430, 4230, 3450, 2940, 6350, 4150, 6500, 5000, 4150, 5000, 4000, 3000,
    # Halogen
    2800, 3200, 4000,
    # HID
    3000, 4200, 3200, 4000, 6000, 3900, 4000, 6000,
    # Sodium
    2100, 1800
]

# supported standard lights
STANDARD_LIGHTS = [
    rt.Directionallight,
    rt.freeSpot,
    rt.Omnilight,

    rt.TargetDirectionallight,
    rt.targetSpot
]

# supported photometric lights
PHOTOMETRIC_LIGHTS = [
    rt.Free_Light,
    rt.Free_Linear,
    rt.Free_Area,
    rt.Free_Rectangle,
    rt.Free_Disc,
    rt.Free_Sphere,
    rt.Free_Cylinder,

    rt.Target_Light,
    rt.Target_Linear,
    rt.Target_Area,
    rt.Target_Rectangle,
    rt.Target_Disc,
    rt.Target_Sphere,
    rt.Target_Cylinder,
]

def getPtr(mEntity):
    rt.execute('fn getProperPtr obj = (return trimRight (refs.getAddr(obj) as string) "P")')
    return int(rt.getProperPtr(mEntity))

def decomposeMatrix3(mat):
    trans = mat.translationpart
    rot = mat.rotationpart
    if mat.determinantsign > 0:
        scale = mat.scalepart
    else:
        # NOTE: seams like F<0 means negative scale
        scale = -mat.scalepart
    # the rotation component is left-handed
    return trans, rot, scale

def calcSceneSize(mSceneBox):
    boxWidth = mSceneBox[1] - mSceneBox[0]
    maxSize = max(boxWidth.x, boxWidth.y, boxWidth.z)

    # NOTE: avoid unreasonable infinity values that happen for empty scenes
    if maxSize == float('inf'):
        return 2e+30
    elif maxSize == -float('inf'):
        return -2e+30
    else:
        return maxSize

def calcFarDistance(mNode, mSceneBox):
    """Calculate maximum possible distance from the node to any geometric point within the scene"""

    dist = calcBoundBoxFarthestDistanceFromPoint(mSceneBox, mNode.transform.pos)

    # NOTE: avoid unreasonable infinity values that happen for empty scenes
    if dist == float('inf'):
        dist = DEFAULT_DIST_EMPTY_SCENE

    return dist


def mNodeIsLight(mNode):
    return (isStandardLight(mNode) or isPhotometricLight(mNode))

def mNodeIsLightProbe(mNode, ofType='ANY'):

    isCube = (rt.classOf(mNode) == getattr(rt, 'V3DReflectionCubemap', None)
              if (ofType == 'ANY' or ofType == 'CUBEMAP') else False)

    isPlane = (rt.classOf(mNode) == getattr(rt, 'V3DReflectionPlane', None)
               if (ofType == 'ANY' or ofType == 'PLANAR') else False)

    return (isCube or isPlane)

def mNodeIsClippingPlane(mNode):
    return rt.classOf(mNode) == getattr(rt, 'V3DClippingPlane', None)

def isStandardLight(node):
    return (rt.classOf(node) in STANDARD_LIGHTS)

def isPhotometricLight(node):
    return (rt.classOf(node) in PHOTOMETRIC_LIGHTS)

def mNodeIsCamera(mNode):
    return (rt.superClassOf(mNode) == rt.camera)

def isPhysicalCamera(mNode):
    return (rt.classOf(mNode) == rt.Physical_Camera)

def isOrthoCamera(mNode):
    if not isPhysicalCamera(mNode) and getattr(mNode, 'orthoProjection', False):
        return True
    else:
        return False

def mNodeIsTextPlus(mNode):
    return (rt.classOf(mNode) == rt.TextPlus)

def mNodeIsBipedFootsteps(mNode):
    ctl = rt.getTMController(mNode)
    return bool(ctl and rt.classOf(ctl) == rt.Footsteps)

def mNodeHasFixOrthoZoom(mNode):

    baseObj = mNode.baseObject

    if (mNode.parent and mNodeIsCamera(mNode.parent) and
            isOrthoCamera(mNode.parent) and
            extract.extractCustomProp(baseObj, 'V3DAdvRenderData', 'fixOrthoZoom')):
        return True
    else:
        return False

def mNodeHasCanvasFitParams(mNode):

    baseObj = mNode.baseObject

    fitX = extract.extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasFitX', 'None').upper()
    fitY = extract.extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasFitY', 'None').upper()

    if (mNode.parent and mNodeIsCamera(mNode.parent) and
            (fitX != 'NONE' or fitY != 'NONE')):
        return True
    else:
        return False

def mNodeUsesLineRendering(mNode):
    if hasattr(mNode, 'V3DLineData'):
        return extract.extractCustomProp(mNode, 'V3DLineData', 'enableLineRendering')
    return False

def isMultiMaterial(mMat):
    return (rt.classOf(mMat) == rt.Multimaterial)

def isStandardMaterial(mMat):
    return (rt.classOf(mMat) == rt.Standardmaterial)

def isPhysicalMaterial(mMat):
    return (rt.classOf(mMat) == rt.PhysicalMaterial)

def isUsdPreviewSurfaceMaterial(mMat):
    return rt.classOf(mMat) == getattr(rt, 'MaxUsdPreviewSurface', None)

def isGLTFMaterial(mMat):
    return (rt.classOf(mMat) == getattr(rt, 'glTFMaterial', None))

def isStandardSurfaceMaterial(mMat):
    return rt.classOf(mMat) == getattr(rt, 'ai_standard_surface', None)

def isLambertMaterial(mMat):
    return rt.classOf(mMat) == getattr(rt, 'ai_lambert', None)

def isVrayMaterial(mMat):
    return (str(rt.classOf(mMat)) == 'VRayMtl')

def hasUVParams(mTex):
    """Check if the texture has standard 2D coord params"""
    return (hasattr(mTex, 'coords') and rt.classOf(mTex.coords) == rt.StandardUVGen)

def imgNeedsCompression(mTex):

    if getattr(mTex, 'compressionErrorStatus', False) == True:
        return False

    compressTextures = extract.extractCustomProp(rt.rootNode, 'V3DExportSettingsData', 'compressTextures', False)
    method = extract.extractCustomProp(mTex, 'V3DTextureData', 'compressionMethod', 'AUTO')

    # only JPEG/PNG (ktx2) or HDR (xz/ktx2) compression supported
    if (compressTextures and
            extract.isBitmapTex(mTex) and mTex.bitmap and method != 'DISABLE' and
            pu.gltf.isCompatibleImagePath(extract.extractTexFileName(mTex)) and
            pu.isPowerOfTwo(mTex.bitmap.width) and pu.isPowerOfTwo(mTex.bitmap.height)):
        return True
    else:
        return False

def createDefaultMaterial(name, diffuseColor):

    mat = rt.PhysicalMaterial()

    mat.name = name
    # close to Arnold
    mat.Base_Color = diffuseColor
    mat.roughness = 0.2

    return mat

def createMColor(vec):
    return rt.color(vec[0], vec[1], vec[2])

def createMPoint3(vec):
    return rt.point3(vec[0], vec[1], vec[2])

def createMQuat(quat):
    # left-handed
    return rt.quat(-quat[0], -quat[1], -quat[2], quat[3])


def getFPS():
    return rt.frameRate;

def framesToTicks(frames):
    return frames / getFPS() * TICKS_PER_SEC

def disableViewportRedraw():
    rt.disableSceneRedraw()

def enableViewportRedraw():
    rt.enableSceneRedraw()

def getMaxVersion():
    return rt.maxVersion()[0]

def getToneMappingParams(cameraNode):

    exposureControl = rt.SceneExposureControl.exposureControl

    # A scene with exposure controlled by a 3d party plugin (e.g. VRay Exposure
    # Control) will have a dummy "missing" exposure control upon opening if the
    # plugin is missing and no compatibility conversion was done for the scene.
    if rt.classOf(exposureControl) == rt.Missing_Exposure_Control:
        return None

    if not exposureControl or not exposureControl.active:
        return None

    if rt.classOf(exposureControl) == rt.Logarithmic_Exposure_Control:

        whiteColorMS = rt.SceneExposureControl.exposureControl.whiteColor

        return {
            'type' : 'logarithmicMax',

            'processBG' : exposureControl.processBG,

            'brightness' : exposureControl.brightness,
            'contrast' : exposureControl.contrast,
            'midTones' : exposureControl.midTones,
            'physicalScale' : exposureControl.physicalScale,

            # "Color Correction"
            'chromaticAdaptation' : exposureControl.chromaticAdaptation,
            'whiteColor' : [whiteColorMS.r, whiteColorMS.g, whiteColorMS.b],
            # "Desaturate Low Levels"
            'colorDifferentiation' : exposureControl.colorDifferentiation,
            'exteriorDaylight' : exposureControl.exteriorDaylight
        }

    elif rt.classOf(exposureControl) == rt.Physical_Camera_Exposure_Control:

        params = {
            'type' : 'physicalMax',

            'processBG' : exposureControl.processBG,

            'highlights' : exposureControl.highlights,
            'midTones' : exposureControl.midtones,
            'shadows' : exposureControl.shadows,
            'saturation' : exposureControl.saturation,

            'physicalScaleMode' : True if exposureControl.physical_scale_mode else False,
            'physicalScale' : exposureControl.physical_scale,

            'aperture': 8,
            'shutter': 0.001,
            'iso': exposureValueToISO(exposureControl.global_ev, 8, 0.001),

            'whiteBalance': calcExposureWhiteBalance(exposureControl),
            'vignetting': exposureControl.vignetting_amount if exposureControl.vignetting_enabled else 0
        }

        if cameraNode:
            phyCamera = rt.getNodeByName(cameraNode.name)
            if phyCamera and rt.classOf(phyCamera) == rt.Physical_Camera:

                aperture = phyCamera.f_number
                shutter = phyCamera.shutter_length_seconds

                params['aperture'] = aperture
                params['shutter'] = shutter

                if exposureControl.use_physical_camera_controls:
                    totalEV = phyCamera.exposure_value - exposureControl.ev_compensation

                    params['whiteBalance'] = calcExposureWhiteBalance(phyCamera)
                    params['vignetting'] = phyCamera.vignetting_amount if phyCamera.vignetting_enabled else 0
                else:
                    totalEV = exposureControl.global_ev

                params['iso'] = exposureValueToISO(totalEV, aperture, shutter)

        return params

    else:
        return None

def exposureValueToISO(ev, aperture, shutter):
    return 100 * (aperture ** 2) / shutter / (2.0 ** ev)

def calcExposureWhiteBalance(obj):
    if obj.white_balance_type == 0:
        color = rt.ConvertKelvinToRGB(ILLUMINANT_TO_KELVIN[obj.white_balance_illuminant], 1)
    elif obj.white_balance_type == 1:
        color = rt.ConvertKelvinToRGB(obj.white_balance_kelvin, 1)
    else:
        color = obj.white_balance_custom

    return [color.r/255, color.g/255, color.b/255]

def getUnitsScaleFactor():
    sysScale = rt.units.SystemScale
    sysType = rt.units.SystemType

    if sysType == rt.Name('inches'):
        scale = 0.0254
    elif sysType == rt.Name('feet'):
        scale = 0.3048
    elif sysType == rt.Name('miles'):
        scale = 1609.34
    elif sysType == rt.Name('millimeters'):
        scale = 0.001
    elif sysType == rt.Name('centimeters'):
        scale = 0.01
    elif sysType == rt.Name('meters'):
        scale = 1.0
    elif sysType == rt.Name('kilometers'):
        scale = 1000
    else:
        printLog('ERROR', 'Wrong system units: {}'.format(sysType))
        scale = 1

    return scale * sysScale

def getSunPosition(node):
    if node is None:
        return (0, math.pi/2)

    azimuth = rt.execute('(getNodeByName "{}").azimuth_deg'.format(node.name))
    northOffset = rt.execute('(getNodeByName "{}").north_direction_deg'.format(node.name))
    altitude = rt.execute('(getNodeByName "{}").altitude_deg'.format(node.name))

    return (math.radians(azimuth+northOffset), math.radians(altitude))

def getPlaneRenderScale(mNode):
    """NOTE: currently unused"""

    plane = rt.getNodeByName(mNode.name)
    if plane and rt.classOf(plane) == rt.Plane:
        return plane.renderScale
    else:
        return 1

def createEmptyBoundingBox():
    return rt.Array(rt.Point3(float('inf'), float('inf'), float('inf')),
                    rt.Point3(float('-inf'), float('-inf'), float('-inf')))

def boundingBoxCenter(box):
    return (box[0] + box[1]) * 0.5

def boundingBoxFromPoints(points):
    box = createEmptyBoundingBox()

    for p in points:
        box[0].x = min(box[0].x, p.x)
        box[0].y = min(box[0].y, p.y)
        box[0].z = min(box[0].z, p.z)

        box[1].x = max(box[1].x, p.x)
        box[1].y = max(box[1].y, p.y)
        box[1].z = max(box[1].z, p.z)

    return box

def transformBoundingBox(box, mat):
    boxmin = box[0]
    boxmax = box[1]

    points = []

    points.append(rt.Point3(boxmin.x, boxmin.y, boxmin.z) * mat)
    points.append(rt.Point3(boxmin.x, boxmin.y, boxmax.z) * mat)
    points.append(rt.Point3(boxmin.x, boxmax.y, boxmin.z) * mat)
    points.append(rt.Point3(boxmin.x, boxmax.y, boxmax.z) * mat)
    points.append(rt.Point3(boxmax.x, boxmin.y, boxmin.z) * mat)
    points.append(rt.Point3(boxmax.x, boxmin.y, boxmax.z) * mat)
    points.append(rt.Point3(boxmax.x, boxmax.y, boxmin.z) * mat)
    points.append(rt.Point3(boxmax.x, boxmax.y, boxmax.z) * mat)

    return boundingBoxFromPoints(points)

def enlargeBoundingBox(box, box2):

    box[0].x = min(box[0].x, box2[0].x)
    box[0].y = min(box[0].y, box2[0].y)
    box[0].z = min(box[0].z, box2[0].z)

    box[1].x = max(box[1].x, box2[1].x)
    box[1].y = max(box[1].y, box2[1].y)
    box[1].z = max(box[1].z, box2[1].z)

def calcBoundBoxFarthestDistanceFromPoint(box, point):
    dx = max(abs(point.x - box[0].x), abs(point.x - box[1].x))
    dy = max(abs(point.y - box[0].y), abs(point.y - box[1].y))
    dz = max(abs(point.z - box[0].z), abs(point.z - box[1].z))
    return math.sqrt(dx**2 + dy**2 + dz**2)

def isIdentity(mat):
    identityEpsilon = 0.000001
    delta = mat - rt.matrix3(1)

    for i in range(4):
        for j in range(3):
            if abs(delta[i][j]) > identityEpsilon:
                return False

    return True

def getLightTargetDistance(mLight):
    if mLight.type == rt.Name('targetSpot') or mLight.type == rt.Name('targetDirect'):
        transLight, _, _ = decomposeMatrix3(mLight.transform)
        transTarget, _, _ = decomposeMatrix3(mLight.target.transform)
        return rt.length(transLight - transTarget)

    elif mLight.type == rt.Name('freeSpot') or mLight.type == rt.Name('freeDirect'):
        return mLight.target_distance

    # other lights are always considered "point" (e.g. photometric target light
    # and free light) or just unsupported
    return -1


def findSkeletonRoot(mNode):

    parent = mNode.parent

    while True:
        if parent and parent != rt.rootNode:
            if extract.extractCustomProp(parent, 'V3DAnimData', 'animSkeletonRoot'):
                return parent
            parent = parent.parent
        else:
            break

    return None

def textPlusGetText(mNode):
    '''
    COMPAT: <2021, it's possible to use pymxs.byref() since 2021
    '''
    rt.execute('fn v3dGetTextPlusText obj = (str = ""; obj.GetPlaintextString(&str); return str)')
    return rt.v3dGetTextPlusText(mNode)

def textPlusGetFont(mNode):
    '''
    COMPAT: <2021, it's possible to use pymxs.byref() since 2021
    '''
    rt.execute('fn v3dGetTextPlusFont obj = (font = ""; obj.GetCharacter 1 0 &font 0 0; return font)')
    return rt.v3dGetTextPlusFont(mNode)


