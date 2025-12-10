from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

from pymxs import runtime as rt

from pluginUtils.log import printLog

import extract
import maxUtils
import utils

IDENTITY_M3 = rt.Matrix3(1)

class Collector():
    def __init__(self, exportSettings):

        self.exportSettings = exportSettings;

        self.rootNode = None
        self.nodes = []
        self.meshNodes = []
        self.curveNodes = []
        self.cameraNodes = []
        self.lightNodes = []
        self.lightProbeNodes = []
        self.clippingPlaneNodes = []
        self.materials = []
        self.textures = []
        # "empty" by default
        self.sceneBox = maxUtils.createEmptyBoundingBox()
        self.sceneShadowCastersBox = maxUtils.createEmptyBoundingBox()

    def collect(self):

        self.rootNode = rt.rootNode

        for node in self.rootNode.children:
            self.collectNode(node)

        # put active camera (if any) first
        activeCamera = rt.getActiveCamera()
        if activeCamera:
            name = activeCamera.name

            for cam in self.cameraNodes:
                if cam == activeCamera:
                    self.cameraNodes.remove(cam)
                    self.cameraNodes.insert(0, cam)
                    break

        tmpDir = self.exportSettings['tmp_dir']

        for node in (self.meshNodes + self.curveNodes):
            for mMat in extract.extractMaterials(node):
                self.collectMat(mMat, node)

        # collect textures
        for mat in self.materials:
            if extract.isPbrMaterial(mat):
                for tex in extract.extractPBRBitmatTextures(mat, tmpDir):
                    if extract.extractTexFileName(tex):
                        if tex not in self.textures:
                            self.textures.append(tex)
                    else:
                        printLog('ERROR', 'Missing BPR texture path or unsupported map type: ' + tex.name)
            else:
                for tex in extract.extractBitmapTextures(mat):
                    if extract.extractTexFileName(tex):
                        if tex not in self.textures:
                            self.textures.append(tex)
                    else:
                        printLog('ERROR', 'Missing texture path or unsupported map type: ' + tex.name)

        envTex = extract.extractEnvMap()
        if envTex:
            for tex in extract.extractBitmapTextures(envTex):
                if extract.extractTexFileName(tex):
                    if tex not in self.textures:
                        self.textures.append(tex)
                else:
                    printLog('ERROR', 'Missing texture path or unsupported map type: ' + tex.name)

    def collectNode(self, node):
        self.nodes.append(node)

        # NOTE: objects of some type (e.g. Line) may return non-accurate values
        # when calculating their bounding box
        # to avoid this issue obtain bounding from object snapshot

        if rt.canConvertTo(node, rt.TriMeshGeometry):

            bakeText = extract.extractCustomProp(self.rootNode, 'V3DExportSettingsData', 'bakeText', False)

            if not bakeText and maxUtils.mNodeIsTextPlus(node):
                self.curveNodes.append(node)
            else:
                self.meshNodes.append(node)

            worldBB = rt.nodeGetBoundingBox(node, IDENTITY_M3)
            maxUtils.enlargeBoundingBox(self.sceneBox, worldBB)
            if node.castShadows:
                maxUtils.enlargeBoundingBox(self.sceneShadowCastersBox, worldBB)

        if maxUtils.mNodeIsLight(node):
            self.lightNodes.append(node)

        if maxUtils.mNodeIsLightProbe(node):
            self.lightProbeNodes.append(node)

        if maxUtils.mNodeIsClippingPlane(node):
            self.clippingPlaneNodes.append(node)

        if maxUtils.mNodeIsCamera(node):
            self.cameraNodes.append(node)

        for child in node.children:
            self.collectNode(child)

    def collectMat(self, mMat, mNode):
        if mMat and mMat not in self.materials:
            self.materials.append(mMat)
        elif not mMat and not extract.extractByName(self.materials, utils.defaultMatName(mNode)):
            self.materials.append(maxUtils.createDefaultMaterial(utils.defaultMatName(mNode), mNode.WireColor))

        return False

    def hasPhysicalMats(self):
        for mat in self.materials:
            if maxUtils.isGLTFMaterial(mat) or maxUtils.isPhysicalMaterial(mat) or maxUtils.isUsdPreviewSurfaceMaterial(mat) or maxUtils.isStandardSurfaceMaterial(mat) or maxUtils.isLambertMaterial(mat):
                return True

        return False
