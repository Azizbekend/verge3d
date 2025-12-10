from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

import json, math, os, shutil, struct, sys

join = os.path.join
norm = os.path.normpath

from pymxs import runtime as rt

import pluginUtils as pu
from pluginUtils.log import printLog
from pluginUtils.manager import AppManagerConn
from pluginUtils import gltf, clamp
import pluginUtils.rawdata

from progressDialog import ProgressDialog

import extract
import maxcpp
import maxUtils
import utils

import sysfont

extractProp = extract.extractProp
extractCustomProp = extract.extractCustomProp
extractTexCoordIndex = extract.extractTexCoordIndex
extractColor = extract.extractColor
extractColor4 = extract.extractColor4
extractVec = extract.extractVec

getPtr = maxUtils.getPtr

PRIMITIVE_MODE_LINES = 1
PRIMITIVE_MODE_TRI = 4

EPSILON = 0.001

SKIN_MOD_CLASSNAME = 'Skin'

AUX_BONE_PREFIX = 'v3d_auxiliary_bone_'

LIGHT_DIST_MULT = 2
SHADOW_BB_Z_COEFF = 1.01
SLOPE_SCALED_BIAS_COEFF = 1.5
ABS_BIAS_COEFF = -0.002

# TODO: support value from exposure control settings
LIGHT_INT_MULT = 1500 / math.pi

CAM_ANGLE_EPSILON = math.pi / 180

WORLD_NODE_MAT_NAME = 'Verge3D_Environment'


class GLTF():

    def __init__(self, exportSettings):
        self.exportSettings = exportSettings
        self.data = {}

    def checkFormat(self, format):
        return (self.exportSettings['format'] == format)

    def generateAsset(self, collector):
        asset = {}

        asset['version'] = '2.0'
        asset['generator'] = 'Verge3D for 3ds Max'

        copyright = extractCustomProp(collector.rootNode, 'V3DExportSettingsData', 'copyright', '')
        if copyright != '':
            asset['copyright'] = copyright

        self.data['asset'] = asset

    def createAnimation(self, collector, mNode):
        channels = []
        samplers = []

        animUsePlaybackRange = (1 if hasattr(collector.rootNode, 'V3DExportSettingsData') and
                extractCustomProp(collector.rootNode, 'V3DExportSettingsData', 'animUsePlaybackRange') else 0)

        # convert MXSWrapperBase to int
        playbackFrom = int(self.exportSettings['anim_playback_original_from'])
        playbackTo = int(self.exportSettings['anim_playback_original_to'])

        animPlaybackRangeFrom = (int(maxUtils.framesToTicks(playbackFrom))
                if animUsePlaybackRange else 0)
        animPlaybackRangeTo = (int(maxUtils.framesToTicks(playbackTo))
                if animUsePlaybackRange else 0)

        animProxyNode = maxUtils.findSkeletonRoot(mNode)
        if not animProxyNode:
            animProxyNode = mNode

        animUseCustomRange = (1 if hasattr(animProxyNode.baseObject, 'V3DAnimData') and
                extractCustomProp(animProxyNode.baseObject, 'V3DAnimData', 'animUseCustomRange') else 0)
        animCustomRangeFrom = (int(maxUtils.framesToTicks(extractCustomProp(animProxyNode.baseObject, 'V3DAnimData', 'animCustomRangeFrom')))
                if animUseCustomRange and hasattr(animProxyNode.baseObject, 'V3DAnimData') else 0)
        animCustomRangeTo = (int(maxUtils.framesToTicks(extractCustomProp(animProxyNode.baseObject, 'V3DAnimData', 'animCustomRangeTo')))
                if animUseCustomRange and hasattr(animProxyNode.baseObject, 'V3DAnimData') else 0)

        animStartWithZero = (1 if hasattr(collector.rootNode, 'V3DExportSettingsData') and
                extractCustomProp(collector.rootNode, 'V3DExportSettingsData', 'animStartWithZero') else 0)

        animKeys = maxcpp.extractNodeAnimation(getPtr(mNode),
                animUsePlaybackRange, animPlaybackRangeFrom, animPlaybackRangeTo,
                animUseCustomRange, animCustomRangeFrom, animCustomRangeTo,
                animStartWithZero)

        samplerIdx = 0

        binary = self.exportSettings['binary']

        posController = rt.getPropertyController(mNode.controller, 'Position')
        pathControl = rt.getPropertyController(posController, 'Path_Constraint') if posController else None
        # do not export position/rotation if path consraint is in use
        needPosAnim = pathControl is None
        needRotAnim = pathControl is None or not pathControl.follow

        if needPosAnim and 'posTimes' in animKeys and 'posValues' in animKeys:
            channels.append(gltf.createAnimChannel(samplerIdx,
                    gltf.getNodeIndex(self.data, getPtr(mNode)), 'translation'))
            samplers.append(gltf.createAnimSampler(self.data, binary,
                    animKeys['posTimes'], animKeys['posValues'], 3))
            samplerIdx += 1

        if needRotAnim and 'rotTimes' in animKeys and 'rotValues' in animKeys:
            channels.append(gltf.createAnimChannel(samplerIdx,
                    gltf.getNodeIndex(self.data, getPtr(mNode)), 'rotation'))
            samplers.append(gltf.createAnimSampler(self.data, binary,
                    animKeys['rotTimes'], animKeys['rotValues'], 4))
            samplerIdx += 1

        if 'scaTimes' in animKeys and 'scaValues' in animKeys:
            channels.append(gltf.createAnimChannel(samplerIdx,
                    gltf.getNodeIndex(self.data, getPtr(mNode)), 'scale'))
            samplers.append(gltf.createAnimSampler(self.data, binary,
                    animKeys['scaTimes'], animKeys['scaValues'], 3))
            samplerIdx += 1

        if 'weightTimes' in animKeys and 'weightValues' in animKeys:
            channels.append(gltf.createAnimChannel(samplerIdx,
                    gltf.getNodeIndex(self.data, getPtr(mNode)), 'weights'))
            samplers.append(gltf.createAnimSampler(self.data, binary,
                    animKeys['weightTimes'], animKeys['weightValues'], 1))
            samplerIdx += 1

        for mMat in extract.extractMaterials(mNode):
            if mMat and gltf.getMaterialIndex(self.data, getPtr(mMat)) > -1:
                mat = self.data['materials'][gltf.getMaterialIndex(self.data, getPtr(mMat))]

                nodeGraph = gltf.getNodeGraph(mat)
                if nodeGraph:
                    for matNode in nodeGraph['nodes']:
                        if 'tmpAnimControl' in matNode:

                            ctl = matNode['tmpAnimControl']

                            if matNode['type'] == 'RGB_MX':
                                if rt.superClassOf(ctl) == rt.Point4Controller:
                                    dimension = 4
                                else:
                                    dimension = 3
                            else:
                                dimension = 1

                            animKeys = maxcpp.extractControlAnimation(getPtr(ctl), dimension,
                                    animUsePlaybackRange, animPlaybackRangeFrom, animPlaybackRangeTo,
                                    animUseCustomRange, animCustomRangeFrom, animCustomRangeTo,
                                    animStartWithZero)

                            if 'values' in animKeys:

                                keys = animKeys['keys']
                                values = animKeys['values']

                                # add alpha channel
                                if dimension == 3:
                                    for i in range(len(keys)):
                                        values.insert(4*i + 3, 1);
                                    dimension = 4

                                binary = self.exportSettings['binary']

                                if matNode['type'] == 'RGB_MX':
                                    channel = gltf.createAnimChannel(samplerIdx, gltf.getNodeIndex(self.data,
                                            getPtr(mNode)), 'material.nodeRGB["' + matNode['name'] + '"]')
                                    sampler = gltf.createAnimSampler(self.data, binary, keys, values, dimension)
                                else:
                                    channel = gltf.createAnimChannel(samplerIdx, gltf.getNodeIndex(self.data,
                                            getPtr(mNode)), 'material.nodeValue["' + matNode['name'] + '"]')
                                    sampler = gltf.createAnimSampler(self.data, binary, keys, values, dimension)

                                channels.append(channel)
                                samplers.append(sampler)

                                samplerIdx += 1

        if pathControl:
            consName = None
            nodeIdx = gltf.getNodeIndex(self.data, getPtr(mNode))
            if nodeIdx > -1:
                node = self.data['nodes'][nodeIdx]
                v3dExt = gltf.getAssetExtension(node, 'S8S_v3d_node')
                if v3dExt and v3dExt.get('constraints', None):
                    for cons in v3dExt['constraints']:
                        if cons.get('type', None) == 'motionPath':
                            consName = cons.get('name')

            if consName is not None:
                percentPontrol = extract.extractAnimatableController(pathControl, 'percent')
                animKeys = maxcpp.extractControlAnimation(getPtr(percentPontrol), 1,
                                        animUsePlaybackRange, animPlaybackRangeFrom, animPlaybackRangeTo,
                                        animUseCustomRange, animCustomRangeFrom, animCustomRangeTo,
                                        animStartWithZero)

                if 'values' in animKeys:

                    keys = animKeys['keys']
                    values = animKeys['values']
                    binary = self.exportSettings['binary']

                    channel = gltf.createAnimChannel(samplerIdx, gltf.getNodeIndex(self.data,
                            getPtr(mNode)), 'constraint["' + consName + '"].value')
                    sampler = gltf.createAnimSampler(self.data, binary, keys, values, 1)

                    channels.append(channel)
                    samplers.append(sampler)

                    samplerIdx += 1
            else:
                printLog('WARNING', 'Can not export animation of Path Constraint for ' + mNode.name)

        if len(channels) and len(samplers):
            animation = {
                'name': animProxyNode.name,
                'channels': channels,
                'samplers': samplers
            }

            if hasattr(animProxyNode.baseObject, 'V3DAnimData'):
                v3dExt = gltf.appendExtension(self.data, 'S8S_v3d_animation', animation)

                v3dExt['auto'] = extractCustomProp(animProxyNode.baseObject, 'V3DAnimData', 'animAuto')

                mAnimLoop = extractCustomProp(animProxyNode.baseObject, 'V3DAnimData', 'animLoop')
                # Max to glTF
                animLoopConvDict = {
                    'Repeat': 'REPEAT',
                    'Once': 'ONCE',
                    'Ping Pong': 'PING_PONG'
                }
                v3dExt['loop'] = animLoopConvDict[mAnimLoop]

                v3dExt['repeatInfinite'] = extractCustomProp(animProxyNode.baseObject, 'V3DAnimData', 'animRepeatInfinite')
                v3dExt['repeatCount'] = extractCustomProp(animProxyNode.baseObject, 'V3DAnimData', 'animRepeatCount')
                v3dExt['offset'] = extractCustomProp(animProxyNode.baseObject, 'V3DAnimData', 'animOffset') / maxUtils.getFPS()

        else:
            animation = None

        return animation

    def generateAnimations(self, collector):

        if not extractCustomProp(collector.rootNode, 'V3DExportSettingsData', 'animExport', False):
            return

        animations = []

        for mNode in collector.nodes:

            mCtlTm = rt.getTMController(mNode)
            if mCtlTm:
                anim = self.createAnimation(collector, mNode)
                if anim:
                    animations.append(anim)

        if len(animations) > 0:
            self.data['animations'] = gltf.mergeAnimations(self.data, animations)

    def generateCameraFromNode(self, mNode, aspectRatio, mSceneBox):
        camera = {}

        camera['name'] = mNode.name

        near = extractProp(mNode, ['nearclip', 'clip_near'])
        far = extractProp(mNode, ['farclip', 'clip_far'])

        # extend distance to prevent situation when no objects are visible
        clipOn = extractProp(mNode, ['clipManually', 'clip_on'], True)
        if not clipOn:
            far = max(maxUtils.calcFarDistance(mNode, mSceneBox), far)

        viewportFitType = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'viewportFitType')
        viewportFitType = viewportFitType.upper() # Max to glTF

        if maxUtils.isOrthoCamera(mNode):
            camera['type'] = 'orthographic'

            orthographic = {}
            camera['orthographic'] = orthographic

            orthographic['znear'] = near
            orthographic['zfar'] = far

            # NOTE: fixes issue with free camera which has None as target distance
            targetDist = mNode.targetDistance
            if targetDist is None:
                targetDist = mNode.baseObject.targetDistance

            xmag, ymag = utils.calcOrthoScales(math.radians(mNode.fov), targetDist, aspectRatio)

            orthographic['xmag'] = xmag
            orthographic['ymag'] = ymag

        else:
            camera['type'] = 'perspective'

            perspective = {}
            camera['perspective'] = perspective

            yfov = None
            if aspectRatio >= 1:
                if viewportFitType != 'VERTICAL':
                    yfov = utils.recalcFOV(math.radians(mNode.fov), 1/aspectRatio)
                else:
                    yfov = math.radians(mNode.fov)
            else:
                if viewportFitType != 'HORIZONTAL':
                    yfov = math.radians(mNode.fov)
                else:
                    yfov = utils.recalcFOV(math.radians(mNode.fov), 1/aspectRatio)

            perspective['aspectRatio'] = aspectRatio
            perspective['yfov'] = yfov

            perspective['znear'] = near
            perspective['zfar'] = far


        v3dExt = gltf.appendExtension(self.data, 'S8S_v3d_camera', camera)

        v3dExt['viewportFitType'] = viewportFitType
        v3dExt['viewportFitInitialAspect'] = aspectRatio

        if hasattr(mNode.baseObject, 'V3DCameraData'):
            v3dExt['enablePan'] = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'panningEnabled')
            v3dExt['rotateSpeed'] = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'rotateSpeed')
            v3dExt['moveSpeed'] = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'moveSpeed')

            disableControls = mNode.baseObject.V3DCameraData.disableControls
            fpsEnabled = mNode.baseObject.V3DCameraData.fpsEnabled
        else:
            printLog('WARNING', 'No custom Verge3D attributes in camera: ' + mNode.name)
            v3dExt['enablePan'] = True
            v3dExt['rotateSpeed'] = 1
            v3dExt['moveSpeed'] = 1

            disableControls = False
            fpsEnabled = False

        target = mNode.target

        if disableControls:
            v3dExt['controls'] = 'NONE'
        elif fpsEnabled:
            v3dExt['controls'] = 'FIRST_PERSON'
            v3dExt['fpsGazeLevel'] = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'fpsGazeLevel')
            v3dExt['fpsStoryHeight'] = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'fpsStoryHeight')
            v3dExt['enablePointerLock'] = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'enablePointerLock')
        elif target:
            v3dExt['controls'] = 'ORBIT'

            # target node is set later in generateNodes() when all nodes are
            # gathered
            v3dExt['orbitTarget'] = [0, 0, 0]

            if hasattr(mNode.baseObject, 'V3DCameraData'):
                v3dExt['orbitMinDistance'] = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'minDist')
                v3dExt['orbitMaxDistance'] = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'maxDist')
                v3dExt['orbitMinZoom'] = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'minZoom')
                v3dExt['orbitMaxZoom'] = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'maxZoom')
                v3dExt['orbitMinPolarAngle'] = math.radians(extractCustomProp(mNode.baseObject,
                                                                              'V3DCameraData', 'minAngle'))
                v3dExt['orbitMaxPolarAngle'] = math.radians(extractCustomProp(mNode.baseObject,
                                                                              'V3DCameraData', 'maxAngle'))

                min_azim_angle = math.radians(extractCustomProp(mNode.baseObject, 'V3DCameraData', 'minAzimuthAngle'))
                max_azim_angle = math.radians(extractCustomProp(mNode.baseObject, 'V3DCameraData', 'maxAzimuthAngle'))

                # export only when needed
                if abs(2 * math.pi - (max_azim_angle - min_azim_angle)) > CAM_ANGLE_EPSILON:
                    v3dExt['orbitMinAzimuthAngle'] = min_azim_angle
                    v3dExt['orbitMaxAzimuthAngle'] = max_azim_angle
            else:
                v3dExt['orbitMinDistance'] = 0
                v3dExt['orbitMaxDistance'] = 5000
                v3dExt['orbitMinPolarAngle'] = 0
                v3dExt['orbitMaxPolarAngle'] = math.pi
        else:
            v3dExt['controls'] = 'FLYING'

        return camera

    def generateCameraFromView(self, aspectRatio):

        printLog('INFO', 'Generating default camera')

        mView = extract.extractActiveViewport()
        activeViewportSave = rt.viewport.activeViewport
        rt.viewport.activeViewport = mView

        camera = {}

        camera['name'] = '__DEFAULT__'

        # NOTE: just some reasonable values, would be better to calculate them
        # based on the general scene scale
        near = 0.1
        far = 1000

        if rt.viewport.IsPerspView():
            camera['type'] = 'perspective'

            perspective = {}
            camera['perspective'] = perspective

            perspective['aspectRatio'] = aspectRatio
            perspective['yfov'] = utils.recalcFOV(math.radians(rt.viewport.GetFOV()), 1/aspectRatio)

            perspective['znear'] = near
            perspective['zfar'] = far
        else:
            camera['type'] = 'orthographic'

            orthographic = {}
            camera['orthographic'] = orthographic

            orthographic['znear'] = near
            orthographic['zfar'] = far

            ymag = rt.viewport.GetScreenScaleFactor(rt.point3(0, 0, 0))
            xmag = ymag * aspectRatio

            orthographic['xmag'] = xmag / 2
            orthographic['ymag'] = ymag / 2

        v3dExt = gltf.appendExtension(self.data, 'S8S_v3d_camera', camera)

        v3dExt['viewportFitType'] = 'VERTICAL'
        v3dExt['viewportFitInitialAspect'] = aspectRatio

        v3dExt['enablePan'] = True
        v3dExt['rotateSpeed'] = 1
        v3dExt['moveSpeed'] = 1

        v3dExt['controls'] = 'ORBIT'

        v3dExt['orbitTarget'] = [0, 0, 0]
        v3dExt['orbitMinDistance'] = 0
        v3dExt['orbitMaxDistance'] = 10000
        v3dExt['orbitMinPolarAngle'] = 0
        v3dExt['orbitMaxPolarAngle'] = math.pi

        rt.viewport.activeViewport = activeViewportSave

        return camera


    def generateCameras(self, collector):
        cameras = []

        mNodes = collector.cameraNodes

        aspectRatio = rt.renderWidth / rt.renderHeight

        if len(mNodes):
            for mNode in mNodes:
                cameras.append(self.generateCameraFromNode(mNode, aspectRatio, collector.sceneBox))
        else:
            cameras.append(self.generateCameraFromView(aspectRatio))

        self.data['cameras'] = cameras


    def generateLight(self, mNode, mSceneBox, mSceneShadowCastersBox):

        light = {}
        light['name'] = mNode.name
        light['profile'] = 'max'

        lightType = mNode.type

        if lightType in [rt.Name('omni'), rt.Name('Free_Point'), rt.Name('Target_Point')]:
            light['type'] = 'point'
        elif lightType in [rt.Name('freeSpot'), rt.Name('targetSpot')]:
            light['type'] = 'spot'
        elif lightType in [rt.Name('freeDirect'), rt.Name('targetDirect')]:
            light['type'] = 'directional'
        elif lightType in [rt.Name('Free_Rectangle'), rt.Name('Target_Rectangle')]:
            light['type'] = 'area'
        else:
            printLog('ERROR', 'Unknown light type: ' + str(lightType))
            light['type'] = 'point'

        isPhotometric = maxUtils.isPhotometricLight(mNode)

        if isPhotometric:
            light['color'] = extractColor(getattr(mNode, 'rgbFilter'))
            light['intensity'] = mNode.intensity / LIGHT_INT_MULT
        else:
            light['color'] = extractColor(getattr(mNode, 'rgb', getattr(mNode, 'color')))
            light['intensity'] = mNode.multiplier * math.pi

        dist = maxUtils.calcFarDistance(mNode, mSceneBox)

        if hasattr(mNode, 'shadowGenerator') and rt.classOf(mNode.shadowGenerator) == rt.ShadowMap:

            trans, rot, _ = maxUtils.decomposeMatrix3(mNode.transform)

            if light['type'] == 'directional':
                matRot = rt.matrix3(1)
                rt.rotate(matRot, rot)
                rt.translate(matRot, trans)
                matRot = rt.inverse(matRot)

                rotatedBox = maxUtils.transformBoundingBox(
                        mSceneShadowCastersBox, matRot)
                bbMin = rotatedBox[0]
                bbMax = rotatedBox[1]

                l = clamp(bbMin.x, -mNode.falloff, mNode.falloff)
                r = clamp(bbMax.x, -mNode.falloff, mNode.falloff)
                b = clamp(bbMin.y, -mNode.falloff, mNode.falloff)
                t = clamp(bbMax.y, -mNode.falloff, mNode.falloff)

                cameraFov = -1
                cameraFar = maxUtils.calcBoundBoxFarthestDistanceFromPoint(
                        mSceneShadowCastersBox, trans)

            elif light['type'] == 'spot':
                l = -1
                r = 1
                b = -1
                t = 1

                cameraFov = math.radians(mNode.falloff)
                cameraFar = maxUtils.calcBoundBoxFarthestDistanceFromPoint(
                        mSceneShadowCastersBox, trans)

            elif light['type'] == 'point' or light['type'] == 'area':
                l = -1
                r = 1
                b = -1
                t = 1

                cameraFov = -1
                cameraFar = dist

            cameraFar = max(cameraFar, maxUtils.getLightTargetDistance(mNode))
            cameraFar = clamp(cameraFar, 0.1, 10000) * SHADOW_BB_Z_COEFF

            absBias = 0
            if hasattr(mNode.baseObject, 'V3DLightData'):
                absBias = extractCustomProp(mNode.baseObject, 'V3DLightData', 'shadowBias')

            expBias = 1
            if hasattr(mNode.baseObject, 'V3DLightData'):
                expBias = extractCustomProp(mNode.baseObject, 'V3DLightData', 'esmExponent')

            shadowMap = mNode.shadowGenerator

            light['shadow'] = {
                'enabled': mNode.baseObject.castShadows,
                'mapSize': shadowMap.mapsize,

                # relevant only for directional lights
                'cameraOrthoLeft': l,
                'cameraOrthoRight': r,
                'cameraOrthoBottom': b,
                'cameraOrthoTop': t,

                # seems like a constant value for lights
                'cameraNear': 0.1,
                # increase far value to prevent potential Z-fighting issues
                'cameraFar': cameraFar,
                'cameraFov': cameraFov,

                'radius': mNode.samplerange / 2,
                # some small bias to prevent artifacts in some cases
                'bias': ABS_BIAS_COEFF * absBias,
                'slopeScaledBias': SLOPE_SCALED_BIAS_COEFF * max(mNode.mapbias, 2),
                'expBias': expBias,
            }

        if (light['type'] == 'point' or light['type'] == 'spot'):

            if isPhotometric:
                # physical light dims close to distance
                light['distance'] = LIGHT_DIST_MULT * dist
            elif getattr(mNode, 'useFarAtten', False):
                light['distance'] = mNode.farAttenEnd

            if isPhotometric:
                light['decay'] = 2.0
            else:
                light['decay'] = mNode.attenDecay - 1 # 1,2,3

        if light['type'] == 'spot':
            # outer angle (degrees)
            angle = mNode.falloff

            light['angle'] = 0.5 * math.radians(angle)

            # ratio
            light['penumbra'] = (angle - mNode.hotspot) / angle

        elif light['type'] == 'area':

            width = mNode.light_Width
            height = mNode.light_length

            light['width'] = width
            light['height'] = height

            # light intensity per square meter
            scale = maxUtils.getUnitsScaleFactor()
            width *= scale
            height *= scale

            light['intensity'] /= (width * height)

            light['ltcMat1'] = pluginUtils.rawdata.ltcMat1
            light['ltcMat2'] = pluginUtils.rawdata.ltcMat2

        return light

    def generateAmbientLight(self):

        light = {
            'name': '__AMBIENT__',
            'profile': 'max',
            'type': 'ambient',
            'color': extract.extractAmbColor(),
            'intensity': 1
        }

        return light

    def generateDefaultLight(self):
        printLog('INFO', 'Generating default light')

        light = {
            'name': '__DEFAULT__',
            'profile': 'max',
            'color': [1,1,1],
            'intensity': math.pi,
            'decay': 0,
            'type': 'point',
        }

        return light

    def generateLights(self, collector):

        lights = []

        mNodes = collector.lightNodes

        if len(mNodes):
            for mNode in mNodes:
                lights.append(self.generateLight(mNode, collector.sceneBox, collector.sceneShadowCastersBox))
        # generate default light for scenes with no lights and no environment
        elif not extract.extractEnvMap():
            lights.append(self.generateDefaultLight())

        lights.append(self.generateAmbientLight())

        gltf.appendExtension(self.data, 'S8S_v3d_lights', self.data, {'lights': lights})

    def generateLightProbe(self, mNode):

        probe = {
            'name': mNode.name,
            'influenceDistance': mNode.influenceDistance,
            'clipStart': mNode.clipStart,
            'visibilityGroup': (mNode.visibilitySet if mNode.visibilitySet != "" else None),
            'visibilityGroupInv': mNode.visibilitySetInv
        }

        if maxUtils.mNodeIsLightProbe(mNode, 'CUBEMAP'):
            probe['type'] = 'CUBEMAP'
            probe['influenceType'] = mNode.influenceType

            probe['parallaxType'] = (mNode.parallaxType if mNode.useCustomParallax
                    else mNode.influenceType)
            probe['parallaxDistance'] = (mNode.parallaxDistance if mNode.useCustomParallax
                    else mNode.influenceDistance)

            probe['intensity'] = mNode.intensity
            probe['clipEnd'] = mNode.clipEnd

            probe['influenceGroup'] = (mNode.influenceSet
                    if mNode.useCustomInfluence and mNode.influenceSet != "" else None)
            probe['influenceGroupInv'] = mNode.influenceSetInv
        else:
            probe['type'] = 'PLANAR'
            probe['falloff'] = mNode.falloff
            probe['planeSize'] = [mNode.planeSizeX, mNode.planeSizeY]

        return probe

    def generateLightProbes(self, collector):

        probes = []

        mNodes = collector.lightProbeNodes
        for mNode in mNodes:
            probes.append(self.generateLightProbe(mNode))

        if len(probes):
            gltf.appendExtension(self.data, 'S8S_v3d_light_probes', self.data, {'lightProbes': probes})

    def generateMeshes(self, collector, progCb=None):

        meshes = []

        for node in collector.meshNodes:

            if progCb is not None:
                progCb((collector.meshNodes.index(node) + 1) / len(collector.meshNodes))

            optimizeAttrs = extractCustomProp(collector.rootNode, 'V3DExportSettingsData', 'optimizeAttrs', False)

            isLine = maxUtils.mNodeUsesLineRendering(node)

            if isLine:
                extrPrimitives = extract.extractLinePrimitives(self.exportSettings, node, optimizeAttrs)
            else:
                extrPrimitives = extract.extractPrimitives(self.exportSettings, node, optimizeAttrs)

            if not len(extrPrimitives):
                continue

            mesh = {
                'name': node.name,
                'id': getPtr(node),
                'primitives': []
            }

            if isLine:
                v3dExt = gltf.appendExtension(self.data, 'S8S_v3d_mesh', mesh)

                v3dExt['lineColor'] = extractColor(extractCustomProp(node, 'V3DLineData', 'lineColor'))
                v3dExt['lineWidth'] = extractCustomProp(node, 'V3DLineData', 'lineWidth')


            binary = self.exportSettings['binary']

            for extrPrim in extrPrimitives:

                prim = {}

                prim['mode'] = PRIMITIVE_MODE_LINES if isLine else PRIMITIVE_MODE_TRI

                extr_indices = extrPrim['indices']

                max_index = max(extr_indices)

                # NOTE: avoiding WebGL2 PRIMITIVE_RESTART_FIXED_INDEX behavior
                # see: https://www.khronos.org/registry/webgl/specs/latest/2.0/#5.18
                if max_index < 255:
                    indCompType = 'UNSIGNED_BYTE'
                elif max_index < 65535:
                    indCompType = 'UNSIGNED_SHORT'
                elif max_index < 4294967295:
                    indCompType = 'UNSIGNED_INT'
                else:
                    printLog('ERROR', 'Invalid max_index: ' + str(max_index))
                    continue

                ind = gltf.generateAccessor(self.data, binary,
                        extr_indices, indCompType, len(extr_indices),
                        'SCALAR', 'ELEMENT_ARRAY_BUFFER')

                prim['indices'] = ind

                attrs = {}

                extrAttrs = extrPrim['attributes']

                pos = gltf.generateAccessor(self.data, binary,
                        extrAttrs['POSITION'], 'FLOAT', len(extrAttrs['POSITION']) // 3,
                        'VEC3', 'ARRAY_BUFFER')

                attrs['POSITION'] = pos

                if extrAttrs.get('NORMAL'):
                    nor = gltf.generateAccessor(self.data, binary,
                            extrAttrs['NORMAL'], 'FLOAT', len(extrAttrs['NORMAL']) // 3,
                            'VEC3', 'ARRAY_BUFFER')
                    attrs['NORMAL'] = nor

                if extrAttrs.get('TANGENT'):
                    tan = gltf.generateAccessor(self.data, binary,
                            extrAttrs['TANGENT'], 'FLOAT', len(extrAttrs['TANGENT']) // 4,
                            'VEC4', 'ARRAY_BUFFER')
                    attrs['TANGENT'] = tan

                if extrAttrs.get('COLOR_0'):
                    col = gltf.generateAccessor(self.data, binary,
                            extrAttrs['COLOR_0'], 'FLOAT', len(extrAttrs['COLOR_0']) // 3,
                            'VEC3', 'ARRAY_BUFFER')
                    attrs['COLOR_0'] = col

                skinAttrIndex = 0
                loop = True
                while loop:
                    jointId = 'JOINTS_' + str(skinAttrIndex)
                    weightId = 'WEIGHTS_' + str(skinAttrIndex)

                    if extrAttrs.get(jointId) and extrAttrs.get(weightId):
                        jointAcc = gltf.generateAccessor(self.data, binary,
                                extrAttrs[jointId], 'UNSIGNED_SHORT',
                                len(extrAttrs[jointId]) // 4, 'VEC4', 'ARRAY_BUFFER')
                        weightAcc = gltf.generateAccessor(self.data, binary,
                                extrAttrs[weightId], 'FLOAT', len(extrAttrs[weightId]) // 4,
                                'VEC4', 'ARRAY_BUFFER')
                        attrs[jointId] = jointAcc
                        attrs[weightId] = weightAcc

                        skinAttrIndex += 1
                    else:
                        loop = False

                texIndex = 0
                loop = True
                while loop:
                    texId = 'TEXCOORD_' + str(texIndex)

                    if extrAttrs.get(texId):
                        tc = gltf.generateAccessor(self.data, binary,
                                extrAttrs[texId], 'FLOAT', len(extrAttrs[texId]) // 2,
                                'VEC2', 'ARRAY_BUFFER')
                        attrs[texId] = tc

                        texIndex+=1
                    else:
                        loop = False

                prim['attributes'] = attrs

                if extrPrim.get('targets') is not None:

                    prim['targets'] = []
                    for extr_target in extrPrim['targets']:

                        target = {}

                        target['POSITION'] = gltf.generateAccessor(
                                self.data, binary ,extr_target['POSITION'], 'FLOAT',
                                len(extr_target['POSITION']) // 3, 'VEC3',
                                'ARRAY_BUFFER')
                        target['NORMAL'] = gltf.generateAccessor(
                                self.data, binary ,extr_target['NORMAL'], 'FLOAT',
                                len(extr_target['NORMAL']) // 3, 'VEC3',
                                'ARRAY_BUFFER')
                        if extr_target.get('TANGENT'):
                            target['TANGENT'] = gltf.generateAccessor(
                                    self.data, binary, extr_target['TANGENT'], 'FLOAT',
                                    len(extr_target['TANGENT']) // 3, 'VEC3',
                                    'ARRAY_BUFFER')

                        prim['targets'].append(target)

                material = gltf.getMaterialIndex(self.data, extrPrim['material'])

                if material > -1:
                    prim['material'] = material

                mesh['primitives'].append(prim)

            if (extrPrimitives[0].get('targetWeights') is not None
                    and extrPrimitives[0].get('targetNames') is not None):
                mesh['weights'] = extrPrimitives[0]['targetWeights']
                mesh['extras'] = {
                    'targetNames': extrPrimitives[0]['targetNames']
                }

            meshes.append(mesh)

        if len(meshes) > 0:
            self.data['meshes'] = meshes

    def createClippingPlane(self, mNode):

        plane = {
            'name': mNode.name,
            'clippingGroup': mNode.affectedObjects if mNode.affectedObjects != '' else None,
            'negated': mNode.negated,
            'clipShadows': mNode.clipShadows,
            'clipIntersection': not mNode.unionPlanes,
            'crossSection': mNode.crossSection if mNode.unionPlanes else False,
            'color': extractVec(mNode.crossSectionColor),
            'opacity': mNode.crossSectionColor.w,
            'renderSide': mNode.crossSectionRenderSide,
            'size': mNode.crossSectionSize
        }

        return plane

    def generateClippingPlanes(self, collector):

        planes = []

        mNodes = collector.clippingPlaneNodes
        for mNode in mNodes:
            planes.append(self.createClippingPlane(mNode))

        if len(planes):
            gltf.appendExtension(self.data, 'S8S_v3d_clipping_planes', self.data, {'clippingPlanes': planes})


    def createFont(self, name):

        font = {
            'name': name,
            'id': name
        }

        filePath = sysfont.match_font(name)
        if filePath:
            fileNameExp = os.path.basename(filePath)
            mimeType = 'font/ttf'
        else:
            printLog('WARNING', 'Font {} not found, switching to Arial'.format(name))
            # open-source and visually similar to default Arial font
            filePath = join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'liberation_sans.woff')
            fileNameExp = 'liberation_sans.woff'
            mimeType = 'font/woff'

        if self.checkFormat('ASCII'):
            destPath = join(self.exportSettings['filedirectory'], fileNameExp)
            if filePath != destPath:
                shutil.copyfile(filePath, destPath)
            font['uri'] = fileNameExp
        else:
            with open(filePath, 'rb') as f:
                fontBytes = f.read()

            bufferView = gltf.generateBufferView(self.data, self.exportSettings['binary'], fontBytes, 0, 0)
            font['bufferView'] = bufferView
            font['mimeType'] = mimeType

        return font

    def generateFonts(self, collector):

        fonts = []

        for mTextPlus in collector.curveNodes:
            font = maxUtils.textPlusGetFont(mTextPlus)
            fonts.append(self.createFont(font))

        if len(fonts) > 0:
            gltf.appendExtension(self.data, 'S8S_v3d_curves', self.data, { 'fonts': fonts })


    def generateCurves(self, collector):
        curves = []

        for mTextPlus in collector.curveNodes:

            curve = {
                'name': mTextPlus.name,
                'id': getPtr(mTextPlus),
                'type': 'font' # NOTE: currently only font curves supported
            }

            curve['text'] = maxUtils.textPlusGetText(mTextPlus)

            fontName = maxUtils.textPlusGetFont(mTextPlus)
            fontIndex = gltf.getFontIndex(self.data, fontName)
            if fontIndex >= 0:
                curve['font'] = fontIndex

            # NOTE: empirical coefficient
            curve['size'] = mTextPlus.size * 0.88
            curve['height'] = mTextPlus.extrudeamount
            curve['curveSegments'] = mTextPlus.interpolationsteps

            if mTextPlus.applybevel:
                curve['bevelThickness'] = mTextPlus.beveldepth
                curve['bevelSize'] = mTextPlus.bevelwidth if mTextPlus.usebevelwidth else mTextPlus.beveldepth
                curve['bevelSegments'] = mTextPlus.bevelsteps
            else:
                curve['bevelThickness'] = 0
                curve['bevelSize'] = 0
                curve['bevelSegments'] = 5

            alignX = mTextPlus.alignment
            if alignX == 0:
                curve['alignX'] = 'left'
            elif alignX == 1:
                curve['alignX'] = 'center'
            elif alignX == 2:
                curve['alignX'] = 'right'
            else:
                printLog('WARNING', 'Unsupported font alignment: ' + str(alignX))
                curve['alignX'] = 'center'

            curve['alignY'] = 'topBaseline'

            materials = extract.extractMaterials(mTextPlus)
            mMat = getPtr(materials[0]) if materials[0] else utils.defaultMatName(mTextPlus)
            materialIndex = gltf.getMaterialIndex(self.data, mMat)
            if materialIndex >= 0:
                curve['material'] = materialIndex
            else:
                printLog('WARNING', 'Material ' + mMat + ' not found')

            curves.append(curve)

        if len(curves) > 0:
            gltf.appendExtension(self.data, 'S8S_v3d_curves', self.data, { 'curves': curves })

    def generateNode(self, mNode):

        node = {}

        node['name'] = mNode.name
        node['id'] = getPtr(mNode)

        if (not maxUtils.isIdentity(extract.extractOffsetTM(mNode)) and
                (maxUtils.mNodeIsCamera(mNode) or maxUtils.mNodeIsLight(mNode))):
            printLog('WARNING', 'Node "' + mNode.name
                    + '" has a non-identity offset matrix, which isn\'t '
                    + 'supported for lights and cameras.')

        if mNode.parent:
            localTM = mNode.transform * rt.inverse(mNode.parent.transform)
        else:
            localTM = mNode.transform

        trans, rot, scale = maxUtils.decomposeMatrix3(localTM)

        if trans.X != 0.0 or trans.Y != 0.0 or trans.Z != 0.0:
            node['translation'] = [trans.X, trans.Z, -trans.Y]

        if rot.X != 0.0 or rot.Y != 0.0 or rot.Z != 0.0 or rot.W != 1.0:
            # left-handed rotation convention
            node['rotation'] = [-rot.X, -rot.Z, rot.Y, rot.W]

        if scale.X != 1.0 or scale.Y != 1.0 or scale.Z != 1.0:
            node['scale'] = [scale.X, scale.Z, scale.Y]

        v3dExt = gltf.appendExtension(self.data, 'S8S_v3d_node', node)

        mesh = gltf.getMeshIndex(self.data, getPtr(mNode))
        if mesh >= 0:
            node['mesh'] = mesh

            v3dExt['renderOrder'] = extractCustomProp(mNode.baseObject, 'V3DMeshData', 'renderOrder', 0)
            v3dExt['frustumCulling'] = extractCustomProp(mNode.baseObject, 'V3DMeshData', 'frustumCulling', True)
            v3dExt['useShadows'] = mNode.receiveShadows
            v3dExt['useCastShadows'] = mNode.castShadows

        # TODO: use pointers
        camera = gltf.getCameraIndex(self.data, mNode.name)
        if camera >= 0:
            node['camera'] = camera

        light = gltf.getLightIndex(self.data, mNode.name)
        if light >= 0:
            gltf.appendExtension(self.data, 'S8S_v3d_lights', node, {'light': light})

        lightProbe = gltf.getLightProbeIndex(self.data, mNode.name)
        if lightProbe >= 0:
            gltf.appendExtension(self.data, 'S8S_v3d_light_probes', node, {'lightProbe': lightProbe})

        clippingPlane = gltf.getClippingPlaneIndex(self.data, mNode.name)
        if clippingPlane >= 0:
            gltf.appendExtension(self.data, 'S8S_v3d_clipping_planes', node, {'clippingPlane': clippingPlane})

        curve = gltf.getCurveIndex(self.data, getPtr(mNode))
        if curve >= 0:
            gltf.appendExtension(self.data, 'S8S_v3d_curves', node, {'curve': curve})

            v3dExt['renderOrder'] = extractCustomProp(mNode.baseObject, 'V3DMeshData', 'renderOrder', 0)
            v3dExt['frustumCulling'] = extractCustomProp(mNode.baseObject, 'V3DMeshData', 'frustumCulling', True)
            v3dExt['useShadows'] = mNode.receiveShadows
            v3dExt['useCastShadows'] = mNode.castShadows

        v3dExt['hidden'] = mNode.isHidden
        v3dExt['hidpiCompositing'] = extractCustomProp(mNode.baseObject, 'V3DAdvRenderData', 'hidpiCompositing', False)

        groupNames = (extract.extractGroupNames(mNode)
                + extract.extractSelectionSetNames(mNode))
        if len(groupNames):
            v3dExt['groupNames'] = groupNames

        customProps = extract.extractCustomProps(mNode)
        if customProps:
            node['extras'] = {
                'customProps': customProps
            }

        return node

    def generateCameraNodeFromView(self):
        printLog('INFO', 'Generating default camera node')

        node = {}

        node['name'] = '__DEFAULT_CAMERA__'

        mView = extract.extractActiveViewport()
        activeViewportSave = rt.viewport.activeViewport
        rt.viewport.activeViewport = mView

        viewTM = rt.Inverse(rt.getViewTM())

        viewCamera = rt.viewport.getCamera(index=mView)

        trans, rot, scale = maxUtils.decomposeMatrix3(viewTM)

        node['translation'] = [trans.x, trans.z, -trans.y]
        node['rotation'] = [rot.x, rot.z, -rot.y, rot.w]
        node['scale'] = [1, 1, 1]

        camera = gltf.getCameraIndex(self.data, '__DEFAULT__')
        if camera >= 0:
            node['camera'] = camera

        rt.viewport.activeViewport = activeViewportSave

        return node

    def generateAmbientLightNode(self):
        node = {}

        node['name'] = '__AMBIENT__'
        light = gltf.getLightIndex(self.data, '__AMBIENT__')
        if light >= 0:
            gltf.appendExtension(self.data, 'S8S_v3d_lights', node, {'light': light})

        return node

    def generateDefaultLightNode(self, mSceneBox):
        printLog('INFO', 'Generating default light node')

        node = {}

        node['name'] = '__DEFAULT_LIGHT__'

        # empirical coefficients
        size = 5 * maxUtils.calcSceneSize(mSceneBox)
        node['translation'] = [size * 0.5, size, -size * 0.75]

        light = gltf.getLightIndex(self.data, '__DEFAULT__')
        if light >= 0:
            gltf.appendExtension(self.data, 'S8S_v3d_lights', node, {'light': light})

        return node

    def generateNodes(self, collector):
        nodes = []

        for mNode in collector.nodes:

            node = self.generateNode(mNode)
            nodes.append(node)

        if gltf.getCameraIndex(self.data, '__DEFAULT__') >= 0:
            nodes.append(self.generateCameraNodeFromView())

        self.data['nodes'] = nodes

        for mNode in collector.nodes:

            nodeIdx = gltf.getNodeIndex(self.data, getPtr(mNode))
            if nodeIdx > -1:
                node = self.data['nodes'][nodeIdx]

                # process constraints
                v3dExt = gltf.getAssetExtension(node, 'S8S_v3d_node')
                if v3dExt:
                    constraints = extract.extractConstraints(self.data, mNode)

                    if len(constraints):
                        v3dExt['constraints'] = constraints

                children = []

                for mChild in mNode.children:
                    if mChild in collector.nodes:
                        children.append(gltf.getNodeIndex(self.data, getPtr(mChild)))

                if len(children):
                    node['children'] = children


        for mNode in collector.lightNodes:
            nodeIdx = gltf.getNodeIndex(self.data, getPtr(mNode))
            if nodeIdx > -1:
                node = self.data['nodes'][nodeIdx]
            else:
                continue

            target = mNode.target
            if target:
                targetNodeIdx = gltf.getNodeIndex(self.data, getPtr(target))
                if targetNodeIdx > -1:
                    lights = gltf.getAssetExtension(self.data, 'S8S_v3d_lights')['lights']
                    lightIdx = gltf.getAssetExtension(node, 'S8S_v3d_lights')['light']

                    lights[lightIdx]['target'] = targetNodeIdx


        for mNode in collector.cameraNodes:
            nodeIdx = gltf.getNodeIndex(self.data, getPtr(mNode))
            if nodeIdx > -1:
                node = self.data['nodes'][nodeIdx]
            else:
                continue

            v3dExt = gltf.getAssetExtension(self.data['cameras'][node['camera']], 'S8S_v3d_camera')
            if v3dExt:
                mMat = extractCustomProp(mNode.baseObject, 'V3DCameraData', 'fpsCollisionMaterial')
                if mMat:
                    mat = gltf.getMaterialIndex(self.data, getPtr(mMat))
                    if mat >= 0:
                        v3dExt['fpsCollisionMaterial'] = mat

                if v3dExt['controls'] == 'ORBIT':
                    target = mNode.target
                    if target:
                        targetNodeIdx = gltf.getNodeIndex(self.data, getPtr(target))
                        if targetNodeIdx >= 0:
                            v3dExt['orbitTarget'] = targetNodeIdx


        if gltf.getLightIndex(self.data, '__DEFAULT__') >= 0:
            # NOTE: find camera node linked to first camera
            for node in nodes:
                if 'camera' in node and node['camera'] == 0:
                    lightNode = self.generateDefaultLightNode(collector.sceneBox)

                    if 'children' in node:
                        node['children'].append(len(nodes))
                    else:
                        node['children'] = [len(nodes)]

                    nodes.append(lightNode)

        lightNode = self.generateAmbientLightNode()
        nodes.append(lightNode)

        self.preprocessCamLampNodes(nodes)

    def generateSkins(self, collector):

        skins = []

        for mNode in collector.meshNodes:
            nodePtr = getPtr(mNode)

            skinBonePtrs = maxcpp.extractSkinBonePointers(nodePtr)

            if skinBonePtrs is not None:
                skin = {}

                joints = [gltf.getNodeIndex(self.data, ptr) for ptr in skinBonePtrs]
                invBindMatrices = maxcpp.extractSkinInvBindMatrices(nodePtr)

                if (self.exportSettings['skinned_mesh_use_aux_bone']
                        and maxcpp.skinnedMeshHasNonSkinnedVertices(nodePtr)):
                    self.data['nodes'].append({ 'name': AUX_BONE_PREFIX + str(len(skins)) })
                    joints.append(len(self.data['nodes']) - 1)

                    worldTM = mNode.GetWorldTM()
                    worldTM.PreRotateX(math.pi/2)
                    worldTM.RotateX(-math.pi/2)

                    a = worldTM.GetRow(0); b = worldTM.GetRow(1)
                    c = worldTM.GetRow(2); d = worldTM.GetRow(3)

                    invBindMatrices.extend([a.X, a.Y, a.Z, 0, b.X, b.Y, b.Z, 0,
                            c.X, c.Y, c.Z, 0, d.X, d.Y, d.Z, 1])

                skin['joints'] = joints
                if invBindMatrices is not None:
                    # not an attribute, so 'target' is None
                    skin['inverseBindMatrices'] = gltf.generateAccessor(
                            self.data, self.exportSettings['binary'],
                            invBindMatrices, 'FLOAT', len(invBindMatrices) // 16,
                            'MAT4', None)

                skins.append(skin)

                skinnedNodeIdx = gltf.getNodeIndex(self.data, nodePtr)
                skinnedNode = self.data['nodes'][skinnedNodeIdx]
                skinnedMeshIdx = skinnedNode.get('mesh', -1)

                if skinnedMeshIdx > -1:
                    skinnedMesh = self.data['meshes'][skinnedMeshIdx]
                    meshCanBeSkinned = True

                    for prim in skinnedMesh['primitives']:
                        if ('JOINTS_0' not in prim['attributes']
                                or 'WEIGHTS_0' not in prim['attributes']):
                            meshCanBeSkinned = False
                            break

                    if meshCanBeSkinned:
                        skinnedNode['skin'] = len(skins) - 1

        if len(skins) > 0:
            self.data['skins'] = skins

    def preprocessCamLampNodes(self, nodes):
        """
        Rotate cameras and lamps by 90 degrees around the X local axis, apply the
        inverted rotation to their children.
        """

        rot_x_90 = rt.matrix3(1)
        rt.rotateX(rot_x_90, -90)

        rot_x_90_inv = rt.matrix3(1)
        rt.rotateX(rot_x_90_inv, 90)

        # rotate cameras and lamps by 90 around X axis prior(!) to applying their TRS,
        # the matrix is still decomposable after such operation
        for node in nodes:
            if 'camera' in node or utils.nodeIsLamp(node) or utils.nodeIsCurve(node):
                mat = rot_x_90 * self.nodeComposeMat(node)

                trans, rot, scale = maxUtils.decomposeMatrix3(mat)
                node['translation'] = extractVec(trans)
                node['rotation'] = extract.extractQuat(rot)
                node['scale'] = extractVec(scale)

                if 'children' in node:
                    for childIndex in node['children']:
                        childNode = nodes[childIndex]
                        childMat = self.nodeComposeMat(childNode) * rot_x_90_inv

                        trans, rot, scale = maxUtils.decomposeMatrix3(childMat)
                        childNode['translation'] = extractVec(trans)
                        childNode['rotation'] = extract.extractQuat(rot)
                        childNode['scale'] = extractVec(scale)

    def nodeComposeMat(self, node):
        matTrans = rt.matrix3(1)
        if 'translation' in node:
            rt.translate(matTrans, maxUtils.createMPoint3(node['translation']))

        matRot = rt.matrix3(1)
        if 'rotation' in node:
            rt.rotate(matRot, maxUtils.createMQuat(node['rotation']))

        matSca = rt.matrix3(1)
        if 'scale' in node:
            rt.scale(matSca, maxUtils.createMPoint3(node['scale']))

        # right to left
        return matSca * matRot * matTrans

    def generateImages(self, collector):

        images = []

        for mTex in collector.textures:
            try:
                image = self.createImage(mTex)
                if image:
                    images.append(image)
            except pu.convert.CompressionFailed:
                mTex.compressionErrorStatus = True
                # try again without compression
                image = self.createImage(mTex)
                if image:
                    images.append(image)

        if len(images) > 0:
            self.data['images'] = images

    def createImage(self, mTex):

        texPath = extract.extractTexFileName(mTex)

        uri = extract.extractImageExportedURI(self.exportSettings, mTex)

        if uri not in self.exportSettings['uri_cache']['uri']:

            image = {}

            image['id'] = uri

            if self.checkFormat('ASCII'):
                # use external file

                old_path = texPath
                new_path = norm(self.exportSettings['filedirectory'] + uri)

                if os.path.normcase(old_path) != os.path.normcase(new_path):
                    # copy an image to a new location

                    if extract.texNeedsConversion(mTex):

                        imgData = extract.extractImageBindataPNG(mTex)

                        with open(new_path, 'wb') as f:
                            f.write(imgData)

                    elif maxUtils.imgNeedsCompression(mTex):
                        if os.path.splitext(uri)[1] == '.xz':
                            pu.manager.AppManagerConn.compressLZMA(old_path, dstPath=new_path)
                        else:
                            method = extractCustomProp(mTex, 'V3DTextureData', 'compressionMethod', 'AUTO')
                            pu.convert.compressKTX2(old_path, dstPath=new_path, method=method)

                    else:
                        shutil.copyfile(old_path, new_path)

                image['uri'] = uri

            else:
                # store image in glb

                if extract.texNeedsConversion(mTex):
                    imgData = extract.extractImageBindataPNG(mTex)
                else:
                    imgData = extract.extractImageBindataAsIs(texPath)

                mimeType = gltf.imageMimeType(texPath)

                if maxUtils.imgNeedsCompression(mTex):
                    if mimeType == 'image/vnd.radiance':
                        imgData = pu.manager.AppManagerConn.compressLZMABuffer(imgData)
                        mimeType = 'application/x-xz'
                    else:
                        method = extractCustomProp(mTex, 'V3DTextureData', 'compressionMethod', 'AUTO')
                        imgData = pu.convert.compressKTX2(srcData=imgData, method=method)
                        mimeType = 'image/ktx2'

                bufferView = gltf.generateBufferView(self.data, self.exportSettings['binary'], imgData, 0, 0)

                image['mimeType'] = mimeType
                image['bufferView'] = bufferView

            self.exportSettings['uri_cache']['uri'].append(uri)
            self.exportSettings['uri_cache']['obj'].append(mTex)

            return image
        else:
            return None

    def generateTextures(self, collector):
        textures = []

        for mTex in collector.textures:
            texture = {}

            texture['name'] = mTex.name
            # names are used for searching ORM, base-alpha textures
            texture['id'] = (mTex.name if mTex.name.startswith(utils.ORM_PREFIX) or
                    mTex.name.startswith(utils.BASE_ALPHA_PREFIX) else getPtr(mTex))

            magFilter = gltf.WEBGL_FILTERS['LINEAR']

            wrapS = gltf.WEBGL_WRAPPINGS['CLAMP_TO_EDGE']
            wrapT = gltf.WEBGL_WRAPPINGS['CLAMP_TO_EDGE']

            if maxUtils.hasUVParams(mTex):
                if mTex.coords.U_Mirror:
                    wrapS = gltf.WEBGL_WRAPPINGS['MIRRORED_REPEAT']
                elif mTex.coords.U_Tile:
                    wrapS = gltf.WEBGL_WRAPPINGS['REPEAT']

                if mTex.coords.V_Mirror:
                    wrapT = gltf.WEBGL_WRAPPINGS['MIRRORED_REPEAT']
                elif mTex.coords.V_Tile:
                    wrapT = gltf.WEBGL_WRAPPINGS['REPEAT']

            texture['sampler'] = gltf.createSampler(self.data, magFilter, wrapS, wrapT)


            v3dExt = gltf.appendExtension(self.data, 'S8S_v3d_texture', texture)

            uri = extract.extractImageExportedURI(self.exportSettings, mTex)

            imgIndex = gltf.getImageIndex(self.data, uri)
            if imgIndex >= 0:
                if os.path.splitext(uri)[1] == '.ktx2':
                    gltf.appendExtension(self.data, 'KHR_texture_basisu', texture, { 'source' : imgIndex })
                elif os.path.splitext(uri)[1] in ['.hdr', '.xz']: # HDR or compressed HDR
                    v3dExt['source'] = imgIndex
                else:
                    texture['source'] = imgIndex

            texPath = extract.extractTexFileName(mTex)

            # Possible values from Blender: linear, non-color, srgb...
            # make normal maps with gamma=1 and non-compatible images linear
            if (extract.isBitmapTex(mTex) and mTex.bitmap and
                    abs(mTex.bitmap.gamma - 1.0) < EPSILON):
                v3dExt['colorSpace'] = 'linear'
            elif not gltf.isCompatibleImagePath(texPath):
                v3dExt['colorSpace'] = 'linear'
            else:
                v3dExt['colorSpace'] = 'srgb'

            if hasattr(mTex, 'V3DTextureData'):
                v3dExt['anisotropy'] = int(extractCustomProp(mTex, 'V3DTextureData', 'anisotropy'))

            if extract.checkUvTransform(mTex):
                v3dExt['uvTransform'] = extract.extractUvTransform(mTex)

            textures.append(texture)

        if len(textures) > 0:
            self.data['textures'] = textures


    def generateMaterials(self, collector):

        materials = []

        for mMat in collector.materials:
            material = {}

            name = mMat.name
            material['name'] = name

            # NOTE: do not assign ID for default materials
            if not '__DEFAULT__' in name:
                material['id'] = getPtr(mMat)

            isPbr = extract.isPbrMaterial(mMat)
            if isPbr:
                if maxUtils.isGLTFMaterial(mMat):

                    material['pbrMetallicRoughness'] = {}
                    pbr = material['pbrMetallicRoughness']

                    baseColorMap = getattr(mMat, 'baseColorMap')
                    occlusionMap = getattr(mMat, 'ambientOcclusionMap')

                    alphaMap = getattr(mMat, 'AlphaMap')
                    if extract.extractAlphaMode(mMat) != 'OPAQUE' and alphaMap:
                        # baseMap texture with alphaMap
                        index = gltf.getTextureIndex(self.data, utils.baseAlphaTexName(name))
                        if index >= 0:
                            baseAlphaTexture = {
                                'index' : index
                            }
                            texCoord = extractTexCoordIndex(baseColorMap or alphaMap)
                            if texCoord > 0:
                                baseAlphaTexture['texCoord'] = texCoord

                            pbr['baseColorTexture'] = baseAlphaTexture

                    elif baseColorMap:
                        # Base color texture
                        index = gltf.getTextureIndex(self.data, getPtr(baseColorMap))
                        if index >= 0:
                            baseColorTexture = {
                                'index' : index
                            }
                            texCoord = extractTexCoordIndex(baseColorMap)
                            if texCoord > 0:
                                baseColorTexture['texCoord'] = texCoord

                            pbr['baseColorTexture'] = baseColorTexture

                    baseColorFactor = extractColor4(getattr(mMat, 'baseColor'))
                    if (baseColorFactor[0] != 1.0 or
                            baseColorFactor[1] != 1.0 or
                            baseColorFactor[2] != 1.0 or
                            baseColorFactor[3] != 1.0):
                        pbr['baseColorFactor'] = baseColorFactor

                    if extract.extractAlphaMode(mMat) == 'MASK':
                        material['alphaCutoff'] = getattr(mMat, 'alphaCutoff')

                    roughnessMap = getattr(mMat, 'roughnessMap')
                    metalnessMap = getattr(mMat, 'metalnessMap')

                    # Metallic factor
                    metallicFactor = getattr(mMat, 'metalness')
                    if metallicFactor != 1.0 and not metalnessMap:
                        pbr['metallicFactor'] = metallicFactor

                    # Roughness factor
                    roughnessFactor = getattr(mMat, 'roughness')
                    if roughnessFactor != 1.0 and not roughnessMap:
                        pbr['roughnessFactor'] = roughnessFactor


                    # Occlusion/Roughness/Metallic texture (ORM)
                    ormIndex = gltf.getTextureIndex(self.data, utils.ormTexName(name))
                    mrMap = metalnessMap or roughnessMap
                    if mrMap:
                        index = ormIndex
                        if index >= 0:
                            metallicRoughnessTexture = {
                                'index' : index
                            }
                            texCoord = extractTexCoordIndex(mrMap)
                            if texCoord > 0:
                                metallicRoughnessTexture['texCoord'] = texCoord

                            pbr['metallicRoughnessTexture'] = metallicRoughnessTexture

                    # Occlusion texture
                    if occlusionMap:
                        index = ormIndex
                        # occlusion texture only
                        if index == -1:
                            index = gltf.getTextureIndex(self.data, getPtr(occlusionMap))

                        if index >= 0:
                            occlusionTexture = {
                                'index' : index
                            }
                            texCoord = extractTexCoordIndex(occlusionMap)
                            if texCoord > 0:
                                occlusionTexture['texCoord'] = texCoord

                            material['occlusionTexture'] = occlusionTexture

                    # Emissive texture
                    emissiveMap = getattr(mMat, 'emissionMap')
                    if emissiveMap:
                        index = gltf.getTextureIndex(self.data, getPtr(emissiveMap))
                        if index >= 0:
                            emissiveTexture = {
                                'index' : index
                            }
                            texCoord = extractTexCoordIndex(emissiveMap)
                            if texCoord > 0:
                                emissiveTexture['texCoord'] = texCoord

                            material['emissiveTexture'] = emissiveTexture

                        material['emissiveFactor'] = [1, 1, 1]
                    else:
                        material['emissiveFactor'] = extractColor(getattr(mMat, 'emissionColor'))

                    # Normal texture

                    normalMap = getattr(mMat, 'normalMap')
                    index = gltf.getTextureIndex(self.data, getPtr(normalMap))
                    if index >= 0:
                        normalTexture = {
                            'index' : index
                        }
                        texCoord = extractTexCoordIndex(normalMap)
                        if texCoord > 0:
                            normalTexture['texCoord'] = texCoord

                        scale = getattr(mMat, 'normal')
                        if scale != 1.0:
                            normalTexture['scale'] = scale

                        material['normalTexture'] = normalTexture

                    # extensions

                    if getattr(mMat, 'unlit'):
                        gltf.appendExtension(self.data, 'KHR_materials_unlit', material)
                    else:
                        if getattr(mMat, 'enableClearCoat'):
                            clearcoatExt = gltf.appendExtension(self.data, 'KHR_materials_clearcoat', material)

                            clearcoatMap = getattr(mMat, 'clearcoatMap')
                            if clearcoatMap:
                                index = gltf.getTextureIndex(self.data, getPtr(clearcoatMap))
                                if index >= 0:
                                    clearcoatTexture = {
                                        'index' : index
                                    }
                                    texCoord = extractTexCoordIndex(clearcoatMap)
                                    if texCoord > 0:
                                        clearcoatTexture['texCoord'] = texCoord

                                    clearcoatExt['clearcoatTexture'] = clearcoatTexture

                            clearcoatExt['clearcoatFactor'] = getattr(mMat, 'clearcoat')

                            # roughness
                            clearcoatRoughnessMap = getattr(mMat, 'clearcoatRoughnessMap')
                            if clearcoatRoughnessMap:
                                index = gltf.getTextureIndex(self.data, getPtr(clearcoatRoughnessMap))
                                if index >= 0:
                                    clearcoatRoughnessTexture = {
                                        'index' : index
                                    }
                                    texCoord = extractTexCoordIndex(clearcoatRoughnessMap)
                                    if texCoord > 0:
                                        clearcoatRoughnessTexture['texCoord'] = texCoord

                                    clearcoatExt['clearcoatRoughnessTexture'] = clearcoatRoughnessTexture

                            clearcoatExt['clearcoatRoughnessFactor'] = getattr(mMat, 'clearcoatRoughness')

                            # normal
                            clearcoatNormalMap = getattr(mMat, 'clearcoatNormalMap')
                            if clearcoatNormalMap:
                                index = gltf.getTextureIndex(self.data, getPtr(clearcoatNormalMap))
                                if index >= 0:
                                    clearcoatNormalTexture = {
                                        'index' : index
                                    }
                                    texCoord = extractTexCoordIndex(clearcoatNormalMap)
                                    if texCoord > 0:
                                        clearcoatNormalTexture['texCoord'] = texCoord

                                    clearcoatExt['clearcoatNormalTexture'] = clearcoatNormalTexture

                        if getattr(mMat, 'enableSheen'):
                            sheenExt = gltf.appendExtension(self.data, 'KHR_materials_sheen', material)

                            sheenColorMap = getattr(mMat, 'sheenColorMap')
                            if sheenColorMap:
                                index = gltf.getTextureIndex(self.data, getPtr(sheenColorMap))
                                if index >= 0:
                                    sheenColorTexture = {
                                        'index' : index
                                    }
                                    texCoord = extractTexCoordIndex(sheenColorMap)
                                    if texCoord > 0:
                                        sheenColorTexture['texCoord'] = texCoord

                                    sheenExt['sheenColorTexture'] = sheenColorTexture

                            sheenExt['sheenColorFactor'] = extractColor(getattr(mMat, 'sheenColor'))

                            # roughness
                            sheenRoughnessMap = getattr(mMat, 'sheenRoughnessMap')
                            if sheenRoughnessMap:
                                index = gltf.getTextureIndex(self.data, getPtr(sheenRoughnessMap))
                                if index >= 0:
                                    sheenRoughnessTexture = {
                                        'index' : index
                                    }
                                    texCoord = extractTexCoordIndex(sheenRoughnessMap)
                                    if texCoord > 0:
                                        sheenRoughnessTexture['texCoord'] = texCoord

                                    sheenExt['sheenRoughnessTexture'] = sheenRoughnessTexture

                            sheenExt['sheenRoughnessFactor'] = getattr(mMat, 'sheenRoughness')

                        if getattr(mMat, 'enableSpecular'):
                            specularExt = gltf.appendExtension(self.data, 'KHR_materials_specular', material)

                            specularMap = getattr(mMat, 'specularMap')
                            if specularMap:
                                index = gltf.getTextureIndex(self.data, getPtr(specularMap))
                                if index >= 0:
                                    specularTexture = {
                                        'index' : index
                                    }
                                    texCoord = extractTexCoordIndex(specularMap)
                                    if texCoord > 0:
                                        specularTexture['texCoord'] = texCoord

                                    specularExt['specularTexture'] = specularTexture

                            specularExt['specularFactor'] = getattr(mMat, 'Specular')

                            specularColorMap = getattr(mMat, 'specularColorMap')
                            if specularColorMap:
                                index = gltf.getTextureIndex(self.data, getPtr(specularColorMap))
                                if index >= 0:
                                    specularColorTexture = {
                                        'index' : index
                                    }
                                    texCoord = extractTexCoordIndex(specularColorMap)
                                    if texCoord > 0:
                                        specularColorTexture['texCoord'] = texCoord

                                    specularExt['specularColorTexture'] = specularColorTexture

                            specularExt['specularColorFactor'] = extractColor(getattr(mMat, 'specularcolor'))

                        if getattr(mMat, 'enableTransmission'):
                            transmissionExt = gltf.appendExtension(self.data, 'KHR_materials_transmission', material)

                            transmissionMap = getattr(mMat, 'transmissionMap')
                            if transmissionMap:
                                index = gltf.getTextureIndex(self.data, getPtr(transmissionMap))
                                if index >= 0:
                                    transmissionTexture = {
                                        'index' : index
                                    }
                                    texCoord = extractTexCoordIndex(transmissionMap)
                                    if texCoord > 0:
                                        transmissionTexture['texCoord'] = texCoord

                                    transmissionExt['transmissionTexture'] = transmissionTexture

                            transmissionExt['transmissionFactor'] = getattr(mMat, 'transmission')

                            # transmission volume extension
                            if getattr(mMat, 'enableVolume'):
                                volumeThicknessExt = gltf.appendExtension(self.data, 'KHR_materials_volume', material)

                                volumeThicknessMap = getattr(mMat, 'volumeThicknessMap')
                                if volumeThicknessMap:
                                    index = gltf.getTextureIndex(self.data, getPtr(volumeThicknessMap))
                                    if index >= 0:
                                        thicknessTexture = {
                                            'index' : index
                                        }
                                        texCoord = extractTexCoordIndex(volumeThicknessMap)
                                        if texCoord > 0:
                                            thicknessTexture['texCoord'] = texCoord

                                        volumeThicknessExt['thicknessTexture'] = thicknessTexture

                                volumeThicknessExt['thicknessFactor'] = getattr(mMat, 'volumeThickness')
                                volumeThicknessExt['attenuationDistance'] = getattr(mMat, 'volumeDistance')
                                volumeThicknessExt['attenuationColor'] = extractColor(getattr(mMat, 'volumeColor'))

                        if getattr(mMat, 'enableIndexOfRefraction'):
                            refractionExt = gltf.appendExtension(self.data, 'KHR_materials_ior', material)
                            refractionExt['ior'] = getattr(mMat, 'indexOfRefraction')

                elif maxUtils.isUsdPreviewSurfaceMaterial(mMat):
                    material['pbrMetallicRoughness'] = {}
                    pbr = material['pbrMetallicRoughness']

                    baseColorMap = getattr(mMat, 'diffuse_color_map')
                    occlusionMap = getattr(mMat, 'occlusion_map')

                    if baseColorMap:
                        # Base color texture
                        index = gltf.getTextureIndex(self.data, getPtr(baseColorMap))
                        if index >= 0:
                            baseColorTexture = {
                                'index' : index
                            }

                            texCoord = extractTexCoordIndex(baseColorMap)
                            if texCoord > 0:
                                baseColorTexture['texCoord'] = texCoord

                            pbr['baseColorTexture'] = baseColorTexture

                    # Base color factor

                    if baseColorMap:
                        baseColorFactor = [1, 1, 1, 1]
                    else:
                        baseColorFactor = extractColor4(getattr(mMat, 'diffuseColor'))

                    transpMap = getattr(mMat, 'opacity_map')
                    if transpMap:
                        baseColorFactor[3] = 1
                    else:
                        baseColorFactor[3] = getattr(mMat, 'opacity')

                    if (baseColorFactor[0] != 1.0 or
                            baseColorFactor[1] != 1.0 or
                            baseColorFactor[2] != 1.0 or
                            baseColorFactor[3] != 1.0):
                        pbr['baseColorFactor'] = baseColorFactor


                    opacityThreshold = getattr(mMat, 'opacityThreshold')
                    material['alphaCutoff'] = opacityThreshold

                    roughnessMap = getattr(mMat, 'roughness_map')
                    metalnessMap = getattr(mMat, 'metallic_map')

                    # Metallic factor
                    metallicFactor = getattr(mMat, 'metallic')
                    if metallicFactor != 1.0 and not metalnessMap:
                        pbr['metallicFactor'] = metallicFactor

                    # Roughness factor
                    roughnessFactor = getattr(mMat, 'roughness')
                    if roughnessFactor != 1.0 and not roughnessMap:
                        pbr['roughnessFactor'] = roughnessFactor


                    # Occlusion/Roughness/Metallic texture (ORM)
                    ormIndex = gltf.getTextureIndex(self.data, utils.ormTexName(name))

                    mrMap = metalnessMap or roughnessMap
                    if mrMap:
                        index = ormIndex
                        if index >= 0:
                            metallicRoughnessTexture = {
                                'index' : index
                            }

                            texCoord = extractTexCoordIndex(mrMap)
                            if texCoord > 0:
                                metallicRoughnessTexture['texCoord'] = texCoord

                            pbr['metallicRoughnessTexture'] = metallicRoughnessTexture

                    # Occlusion texture
                    if occlusionMap:
                        index = ormIndex
                        # occlusion texture only
                        if index == -1:
                            index = gltf.getTextureIndex(self.data, getPtr(occlusionMap))

                        if index >= 0:
                            occlusionTexture = {
                                'index' : index
                            }

                            texCoord = extractTexCoordIndex(occlusionMap)
                            if texCoord > 0:
                                occlusionTexture['texCoord'] = texCoord

                            material['occlusionTexture'] = occlusionTexture


                    # Emissive texture

                    emissiveMap = getattr(mMat, 'emissive_color_map')
                    if emissiveMap:
                        index = gltf.getTextureIndex(self.data, getPtr(emissiveMap))
                        if index >= 0:
                            emissiveTexture = {
                                'index' : index
                            }

                            texCoord = extractTexCoordIndex(emissiveMap)
                            if texCoord > 0:
                                emissiveTexture['texCoord'] = texCoord

                            material['emissiveTexture'] = emissiveTexture
                        material['emissiveFactor'] = [1, 1, 1]

                    else:
                        material['emissiveFactor'] = extractColor(getattr(mMat, 'emissiveColor'))


                    # Normal texture

                    normalMap = getattr(mMat, 'normal_map')
                    index = gltf.getTextureIndex(self.data, getPtr(normalMap))
                    if index >= 0:
                        normalTexture = {
                            'index' : index
                        }

                        texCoord = extractTexCoordIndex(normalMap)
                        if texCoord > 0:
                            normalTexture['texCoord'] = texCoord

                        material['normalTexture'] = normalTexture

                else:
                    material['pbrMetallicRoughness'] = {}
                    pbr = material['pbrMetallicRoughness']

                    baseColorMap = getattr(mMat, 'base_color_map')
                    occlusionMap = getattr(mMat, 'base_weight_map')

                    if baseColorMap:
                        # Base color texture
                        index = gltf.getTextureIndex(self.data, getPtr(baseColorMap))
                        if index >= 0:
                            baseColorTexture = {
                                'index' : index
                            }

                            texCoord = extractTexCoordIndex(baseColorMap)
                            if texCoord > 0:
                                baseColorTexture['texCoord'] = texCoord

                            pbr['baseColorTexture'] = baseColorTexture

                    # Base color factor

                    if baseColorMap:
                        baseColorFactor = [1, 1, 1, 1]
                    else:
                        baseColorFactor = extractColor4(getattr(mMat, 'base_color'))

                    if not occlusionMap:
                        baseWeight = getattr(mMat, 'base_weight')

                        baseColorFactor[0] *= baseWeight
                        baseColorFactor[1] *= baseWeight
                        baseColorFactor[2] *= baseWeight

                    transpMap = getattr(mMat, 'transparency_map')

                    if transpMap:
                        baseColorFactor[3] = 1
                    else:
                        baseColorFactor[3] = 1 - getattr(mMat, 'transparency')

                    if (baseColorFactor[0] != 1.0 or
                            baseColorFactor[1] != 1.0 or
                            baseColorFactor[2] != 1.0 or
                            baseColorFactor[3] != 1.0):
                        pbr['baseColorFactor'] = baseColorFactor

                    roughnessMap = getattr(mMat, 'roughness_map')
                    metalnessMap = getattr(mMat, 'metalness_map')

                    # Metallic factor
                    metallicFactor = getattr(mMat, 'metalness')
                    if metallicFactor != 1.0 and not metalnessMap:
                        pbr['metallicFactor'] = metallicFactor

                    # Roughness factor
                    roughnessFactor = getattr(mMat, 'roughness')
                    if roughnessFactor != 1.0 and not roughnessMap:
                        pbr['roughnessFactor'] = roughnessFactor


                    # Occlusion/Roughness/Metallic texture (ORM)
                    ormIndex = gltf.getTextureIndex(self.data, utils.ormTexName(name))

                    mrMap = metalnessMap or roughnessMap
                    if mrMap:
                        index = ormIndex
                        if index >= 0:
                            metallicRoughnessTexture = {
                                'index' : index
                            }

                            texCoord = extractTexCoordIndex(mrMap)
                            if texCoord > 0:
                                metallicRoughnessTexture['texCoord'] = texCoord

                            pbr['metallicRoughnessTexture'] = metallicRoughnessTexture

                    # Occlusion texture
                    if occlusionMap:
                        index = ormIndex
                        # occlusion texture only
                        if index == -1:
                            index = gltf.getTextureIndex(self.data, getPtr(occlusionMap))

                        if index >= 0:
                            occlusionTexture = {
                                'index' : index
                            }

                            texCoord = extractTexCoordIndex(occlusionMap)
                            if texCoord > 0:
                                occlusionTexture['texCoord'] = texCoord

                            material['occlusionTexture'] = occlusionTexture


                    # Emissive texture

                    emissiveMap = getattr(mMat, 'emit_color_map')
                    if emissiveMap:
                        index = gltf.getTextureIndex(self.data, getPtr(emissiveMap))
                        if index >= 0:
                            emissiveTexture = {
                                'index' : index
                            }

                            texCoord = extractTexCoordIndex(emissiveMap)
                            if texCoord > 0:
                                emissiveTexture['texCoord'] = texCoord

                            material['emissiveTexture'] = emissiveTexture

                    # Emissive factor

                    emissiveFactor = getattr(mMat, 'emission')

                    if emissiveFactor != 1.0:
                        material['emissiveFactor'] = [emissiveFactor, emissiveFactor, emissiveFactor]

                    # Normal texture

                    bumpMap = getattr(mMat, 'bump_map')
                    if bumpMap and rt.classOf(bumpMap) == rt.Normal_Bump and getattr(bumpMap, 'normal_map'):
                        normalMap = getattr(bumpMap, 'normal_map')
                        index = gltf.getTextureIndex(self.data, getPtr(normalMap))
                        if index >= 0:
                            normalTexture = {
                                'index' : index
                            }

                            texCoord = extractTexCoordIndex(normalMap)
                            if texCoord > 0:
                                normalTexture['texCoord'] = texCoord

                            scale = getattr(mMat, 'bump_map_amt')
                            if scale != 1.0:
                                normalTexture['scale'] = scale

                            material['normalTexture'] = normalTexture

            mStdMat = mMat if rt.classOf(mMat) == rt.Standardmaterial else None

            if mStdMat:
                material['doubleSided'] = mStdMat.twoSided
            elif maxUtils.isGLTFMaterial(mMat):
                material['doubleSided'] = getattr(mMat, 'DoubleSided')
            else:
                twoSided = extractCustomProp(mMat, 'V3DMaterialData', 'twoSided')
                material['doubleSided'] = twoSided if twoSided is not None else False

            alphaMode = extract.extractAlphaMode(mMat)

            if alphaMode != 'OPAQUE':
                material['alphaMode'] = alphaMode

            if not isPbr:
                v3dExt = gltf.appendExtension(self.data, 'S8S_v3d_materials', material)

                v3dExt['profile'] = 'max'

                v3dExt['nodeGraph'] = extract.extractNodeGraph(mMat, self.data)

                alphaModeProp = extractCustomProp(mMat, 'V3DMaterialData', 'alphaMode', '').upper()

                if alphaModeProp == 'COVERAGE':
                    v3dExt['alphaToCoverage'] = True
                elif alphaModeProp == 'ADD':
                    blendMode = gltf.createBlendMode('FUNC_ADD', 'ONE', 'ONE')
                    v3dExt['blendMode'] = blendMode

                # disable GTAO for BLEND materials due to implementation issues
                v3dExt['gtaoVisible'] = extractCustomProp(mMat, 'V3DMaterialData',
                        'alphaMode', '').upper() != 'BLEND'

                if (hasattr(mMat, 'V3DMaterialData') and
                        not extractCustomProp(mMat, 'V3DMaterialData', 'depthWrite')):
                    v3dExt['depthWrite'] = False

                if (hasattr(mMat, 'V3DMaterialData') and
                        not extractCustomProp(mMat, 'V3DMaterialData', 'depthTest')):
                    v3dExt['depthTest'] = False

                if (hasattr(mMat, 'V3DMaterialData') and
                        extractCustomProp(mMat, 'V3DMaterialData', 'dithering')):
                    v3dExt['dithering'] = True

                # for Max useShadows/useCastShadows assigned to the node

            materials.append(material)

        materials.append(self.generateWorldMaterial())

        if len(materials) > 0:
            self.data['materials'] = materials

    def generateWorldMaterial(self):

        envTex = extract.extractEnvMap()
        if envTex:
            nodeGraph = extract.extractNodeGraph(envTex, self.data)
        else:
            nodeGraph = { 'nodes': [], 'edges': [] }

            nodeGraph['nodes'].append({
                'name' : 'World Output',
                'type' : 'OUTPUT_MX',
                'inputs': [extract.extractEnvColor() + [1]],
                'outputs': [],
                'is_active_output': True
            })

        worldMat = {
            'name': WORLD_NODE_MAT_NAME,
            'extensions': {
                'S8S_v3d_materials': {
                    'profile': 'max',
                    'nodeGraph': nodeGraph
                }
            }
        }

        # add to extensionsUsed
        gltf.appendExtension(self.data, 'S8S_v3d_materials')

        return worldMat

    def getPostprocessingEffects(self, rootNode):
        ppEffects = []

        if not hasattr(rootNode, 'V3DExportSettingsData'):
            printLog('WARNING', 'No custom Verge3D attributes in root node')
            return ppEffects

        # ambient occlusion
        if extractCustomProp(rootNode, 'V3DExportSettingsData', 'aoEnabled'):
            ppEffects.append({
                'type': 'gtao',
                'distance': extractCustomProp(rootNode, 'V3DExportSettingsData', 'aoDistance'),
                'factor': extractCustomProp(rootNode, 'V3DExportSettingsData', 'aoFactor'),
                'precision': extractCustomProp(rootNode, 'V3DExportSettingsData', 'aoTracePrecision'),
                'bentNormals': extractCustomProp(rootNode, 'V3DExportSettingsData', 'aoBentNormals'),
                'bounceApprox': False
            })

        # outline
        if extractCustomProp(rootNode, 'V3DExportSettingsData', 'outlineEnabled'):
            ppEffects.append({
                'type': 'outline',
                'edgeStrength': extractCustomProp(rootNode, 'V3DExportSettingsData', 'edgeStrength'),
                'edgeGlow': extractCustomProp(rootNode, 'V3DExportSettingsData', 'edgeGlow'),
                'edgeThickness': extractCustomProp(rootNode, 'V3DExportSettingsData', 'edgeThickness'),
                'pulsePeriod': extractCustomProp(rootNode, 'V3DExportSettingsData', 'pulsePeriod'),
                'visibleEdgeColor': extractColor4(
                        extractCustomProp(rootNode, 'V3DExportSettingsData', 'visibleEdgeColor')),
                'hiddenEdgeColor': extractColor4(
                        extractCustomProp(rootNode, 'V3DExportSettingsData', 'hiddenEdgeColor')),
                'renderHiddenEdge': extractCustomProp(rootNode, 'V3DExportSettingsData', 'renderHiddenEdge')
            })

        return ppEffects


    def generateScenes(self, collector):
        scenes = []

        scene = {}

        scene['name'] = 'Scene'

        scene['nodes'] = []

        for i in range(len(self.data['nodes'])):
            # top level nodes only
            if utils.getParentNode(self.data, i) == -1:
                scene['nodes'].append(i)

        v3dExt = gltf.appendExtension(self.data, 'S8S_v3d_scene', scene)

        scene['extras'] = {
            'animFrameRate': maxUtils.getFPS(),
            'coordSystem': 'Z_UP_RIGHT'
        }

        worldMatIdx = gltf.getMaterialIndex(self.data, WORLD_NODE_MAT_NAME)
        if worldMatIdx >= 0:
            v3dExt['worldMaterial'] = worldMatIdx

        v3dExt['physicallyCorrectLights'] = True
        v3dExt['unitsScaleFactor'] = maxUtils.getUnitsScaleFactor()

        cameraNode = collector.cameraNodes[0] if len(collector.cameraNodes) else None
        toneMap = maxUtils.getToneMappingParams(cameraNode)
        if toneMap:
            v3dExt['toneMapping'] = toneMap

        if extractCustomProp(collector.rootNode, 'V3DExportSettingsData', 'useHDR', False):
            v3dExt['useHDR'] = True

        if extractCustomProp(collector.rootNode, 'V3DExportSettingsData', 'useOIT', False):
            v3dExt['useOIT'] = True

        if hasattr(collector.rootNode, 'V3DExportSettingsData'):
            aaMethodConvDict = {
                'Auto': 'AUTO',
                'MSAA 4x': 'MSAA4',
                'MSAA 8x': 'MSAA8',
                'MSAA 16x': 'MSAA16',
                'FXAA': 'FXAA',
                'None': 'NONE'
            }
            v3dExt['aaMethod'] = aaMethodConvDict[extractCustomProp(
                    collector.rootNode, 'V3DExportSettingsData', 'aaMethod')]

        if hasattr(collector.rootNode, 'V3DExportSettingsData'):
            v3dExt['pmremMaxTileSize'] = int(extractCustomProp(collector.rootNode,
                                         'V3DExportSettingsData', 'pmremMaxTileSize'))
            v3dExt['iblEnvironmentMode'] = extractCustomProp(collector.rootNode,
                                           'V3DExportSettingsData', 'iblEnvironmentMode')

        if hasattr(collector.rootNode, 'V3DExportSettingsData'):
            shadowFilteringType = extractCustomProp(collector.rootNode,
                    'V3DExportSettingsData', 'shadowFilteringType')
        else:
            shadowFilteringType = 'PCFPOISSON'

        if hasattr(collector.rootNode, 'V3DExportSettingsData'):
            esmDistanceScale = extractCustomProp(collector.rootNode,
                    'V3DExportSettingsData', 'esmDistanceScale')
        else:
            esmDistanceScale = 0.3

        v3dExt['shadowMap'] = {
            'type' : shadowFilteringType,
            'renderReverseSided' : False,
            'renderSingleSided' : True,
            'esmDistanceScale': esmDistanceScale,
        }

        ppEffects = self.getPostprocessingEffects(collector.rootNode)
        if len(ppEffects):
            v3dExt['postprocessing'] = ppEffects

        scenes.append(scene)

        if len(scenes) > 0:
            self.data['scenes'] = scenes

    def generateScene(self):
        """
        Generates the top level scene entry.
        """

        self.data['scene'] = 0

    def generate(self, collector):

        self.generateAsset(collector)
        ProgressDialog.setValue(1)
        self.generateImages(collector)
        ProgressDialog.setValue(3)
        self.generateTextures(collector)
        ProgressDialog.setValue(6)
        self.generateMaterials(collector)
        ProgressDialog.setValue(9)
        self.generateCameras(collector)
        ProgressDialog.setValue(10)
        self.generateLights(collector)
        ProgressDialog.setValue(11)
        self.generateLightProbes(collector)
        ProgressDialog.setValue(12)
        self.generateFonts(collector)
        ProgressDialog.setValue(13)
        self.generateMeshes(collector, lambda frac: ProgressDialog.setValue(12 + 70 * frac))
        ProgressDialog.setValue(82)
        self.generateClippingPlanes(collector)
        ProgressDialog.setValue(83)
        self.generateCurves(collector)
        ProgressDialog.setValue(84)
        self.generateNodes(collector)
        ProgressDialog.setValue(85)
        self.generateSkins(collector)
        ProgressDialog.setValue(86)
        self.generateAnimations(collector)
        ProgressDialog.setValue(93)
        self.generateScenes(collector)
        ProgressDialog.setValue(96)
        self.generateScene()
        ProgressDialog.setValue(97)

        byteLength = len(self.exportSettings['binary'])

        if byteLength > 0:
            self.data['buffers'] = []

            buffer = {
                'byteLength' : byteLength
            }

            if self.checkFormat('ASCII'):
                uri = self.exportSettings['binaryfilename']
                buffer['uri'] = uri

            self.data['buffers'].append(buffer)

        ProgressDialog.setValue(100)

    def cleanupDataKeys(self, data):
        """
        Remove "id" keys used in the exporter to assign entity indices
        """
        for key, val in data.items() if (sys.version_info[0] == 3) else data.iteritems():
            if type(val) == list:
                for entity in val:
                    if 'id' in entity:
                        del entity['id']
            elif key == 'extensions' and 'S8S_v3d_lights' in val:
                self.cleanupDataKeys(val['S8S_v3d_lights'])
            elif key == 'extensions' and 'S8S_v3d_light_probes' in val:
                self.cleanupDataKeys(val['S8S_v3d_light_probes'])
            elif key == 'extensions' and 'S8S_v3d_curves' in val:
                self.cleanupDataKeys(val['S8S_v3d_curves'])
            elif key == 'extensions' and 'S8S_v3d_clipping_planes' in val:
                self.cleanupDataKeys(val['S8S_v3d_clipping_planes'])

        if 'materials' in data:
            for mat in data['materials']:
                nodeGraph = gltf.getNodeGraph(mat)
                if nodeGraph:
                    for matNode in nodeGraph['nodes']:
                        if 'id' in matNode:
                            del matNode['id']
                        if 'tmpAnimControl' in matNode:
                            del matNode['tmpAnimControl']

    def compressLZMA(self, path, collector):

        settings = self.exportSettings

        if settings['sneak_peek']:
            return

        if not (hasattr(collector.rootNode, 'V3DExportSettingsData') and
                extractCustomProp(collector.rootNode, 'V3DExportSettingsData', 'lzmaEnabled')):
            return

        AppManagerConn.compressLZMA(path)

    def save(self, collector):

        self.cleanupDataKeys(self.data)

        indent = None
        separators = separators=(',', ':')

        if (self.checkFormat('ASCII') and not self.exportSettings['strip']) or self.exportSettings['sneak_peek']:
            indent = 4
            separators = separators=(', ', ' : ')

        gltfEncoded = json.dumps(self.data, indent=indent, separators=separators, sort_keys=True)

        if self.checkFormat('ASCII'):
            file = open(self.exportSettings['filepath'], "w", encoding="utf8", newline="\n")
            file.write(gltfEncoded)
            file.write("\n")
            file.close()

            self.compressLZMA(self.exportSettings['filepath'], collector)

            binary = self.exportSettings['binary']
            if len(binary) > 0:
                file = open(self.exportSettings['filedirectory'] + self.exportSettings['binaryfilename'], 'wb')
                file.write(binary)
                file.close()

                self.compressLZMA(self.exportSettings['filedirectory'] + self.exportSettings['binaryfilename'], collector)
        else:
            file = open(self.exportSettings['filepath'], 'wb')

            gltfData = gltfEncoded.encode()
            binary = self.exportSettings['binary']

            lengthGLTF = len(gltfData)
            spacesGLTF = (4 - (lengthGLTF & 3)) & 3
            lengthGLTF += spacesGLTF

            lengthBin = len(binary)
            zeros_bin = (4 - (lengthBin & 3)) & 3
            lengthBin += zeros_bin

            length = 12 + 8 + lengthGLTF
            if lengthBin > 0:
                length += 8 + lengthBin

            # Header (Version 2)
            file.write('glTF'.encode())
            file.write(struct.pack('I', 2))
            file.write(struct.pack('I', length))

            # Chunk 0 (JSON)
            file.write(struct.pack('I', lengthGLTF))
            file.write('JSON'.encode())
            file.write(gltfData)
            for i in range(0, spacesGLTF):
                file.write(' '.encode())

            # Chunk 1 (BIN)
            if lengthBin > 0:
                file.write(struct.pack('I', lengthBin))
                file.write('BIN\0'.encode())
                file.write(binary)
                for i in range(0, zeros_bin):
                    file.write('\0'.encode())

            file.close()

            self.compressLZMA(self.exportSettings['filepath'], collector)
