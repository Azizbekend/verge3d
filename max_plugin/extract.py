from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

import io, math, os, re, sys, subprocess, tempfile

join = os.path.join
norm = os.path.normpath

from pymxs import runtime as rt

from pluginUtils.log import printLog
from pluginUtils.path import getRoot
from pluginUtils.gltf import getNodeIndex, getTextureIndex, isCompatibleImagePath
from pluginUtils import clamp

import maxcpp
import maxUtils
import utils

import pcpp, pyosl.oslparse, pyosl.glslgen

from PySide2.QtGui import QImage, QColor, QPainter, qRed, qBlue, qGreen
from PySide2 import QtCore

import struct

getPtr = maxUtils.getPtr

CONVERTIBLE_NODE_CLASSES = [
    # General
    rt.BlendedBoxMap,
    rt.Camera_Map_Per_Pixel,
    rt.Cellular,
    rt.Checker,
    rt.Dent,
    rt.Gradient,
    rt.Gradient_Ramp,
    rt.Marble,
    rt.MultiTile,
    rt.Particle_Age,
    rt.Particle_MBlur,
    rt.Perlin_Marble,
    rt.Raytrace,
    rt.ShapeMap,
    rt.Smoke,
    rt.Speckle,
    rt.Splat,
    rt.Stucco,
    rt.Substance,
    rt.swirl,
    rt.TextMap,
    rt.TextureObjMask,
    rt.tiles,
    rt.Vector_Displacement,
    rt.Vector_Map,
    rt.Water,
    rt.Wood,

    # Scanline
    rt.Flat_Mirror,
    rt.Thin_Wall_Refraction
]

CONVERTED_MAP_NAME = 'converted_map.png'

CURVE_DATA_SIZE = 256
RAMP_DATA_SIZE = 512
CONN_TABLE_SIZE = 100

GRADRAMP_INTERP_CUSTOM = 0
GRADRAMP_INTERP_EASE_IN = 1
GRADRAMP_INTERP_EASE_IN_OUT = 2
GRADRAMP_INTERP_EASE_OUT = 3
GRADRAMP_INTERP_LINEAR = 4
GRADRAMP_INTERP_SOLID = 5

DEFAULT_IOR = 1.52

def extractPrimitives(exportSettings, node, optimizeAttrs):

    primitives = []

    # NOTE: ignore the special Biped Footsteps node, it's of type Mesh but
    # let's treat as a helper
    if maxUtils.mNodeIsBipedFootsteps(node):
        return []

    triMesh = extractTriMeshFromNode(node)

    mMats = extractMaterials(node)
    for i in range(len(mMats)):
        mMat = mMats[i]

        extractTan = (not optimizeAttrs) or (bool(mMat) and checkUseTangents(mMat))

        # NOTE: check for both faces and tex vertices to prevent crash in vcMap.GetTextureFace()
        extractVC = bool(mMat and triMesh.numcpvverts and checkUseVC(mMat))

        # NOTE: if the mesh has 0 or 1 material its faces have several different
        # materials anyway, -1 is a workaround for this case to specify that the
        # whole mesh should be extracted
        matID = -1 if len(mMats) == 1 else i

        attrs, indices, targetData = maxcpp.extractGeometry(getPtr(node), matID,
                exportSettings['skinned_mesh_use_aux_bone'], extractTan, extractVC)

        if 'POSITION' not in attrs:
            continue

        prim = {
            'attributes': attrs,
            'indices': indices,
            'material': getPtr(mMat) if mMat else utils.defaultMatName(node)
        }

        if len(targetData) > 0:
            prim['targets'] = targetData['deltas']

            # NOTE: extracted weight and name data is the same for all primitives, can
            # be optimized
            prim['targetWeights'] = targetData['weights']
            prim['targetNames'] = targetData['names']

        primitives.append(prim)

    return primitives

def extractLinePrimitives(exportSettings, node, optimizeAttrs):
    """
    Extracting line primitives from a mesh.
    TODO(Nury): primitives are splitted up, if the indices range is exceeded.
    """
    printLog('INFO', 'Extracting line primitive {}'.format(node.name))

    result_primitives = []

    mMats = extractMaterials(node)

    matName = (mMats[0].name if mMats and mMats[0] is not None else utils.defaultMatName(node))

    if rt.superClassOf(node) == rt.Shape:
        # default values, NURBSCurve objects use only them
        steps = -1
        optimize = False

        for curveIndx in range(maxcpp.shapeNumberOfCurves(getPtr(node))):
            positions, indices = maxcpp.extractLineGeometry(getPtr(node), steps,
                optimize, curveIndx)

            primitive = {
                'attributes': { 'POSITION': positions },
                'indices': indices,
                'material': matName
            }

            result_primitives.append(primitive)
    else:
        primitive = {
            'attributes': { 'POSITION': [] },
            'indices': [],
            'material': matName
        }

        orig_indices = primitive['indices']
        orig_positions = primitive['attributes']['POSITION']
        vertex_index_to_new_index = {}

        node = rt.snapshot(node)
        mat3 = rt.inverse(rt.getNodeTM(node))
        ePoly = rt.convertTo(node, rt.Editable_Poly)
        edges = ePoly.Edges
        for n in range(1, edges.count + 1):
            for i in rt.polyop().getEdgeVerts(ePoly, n):
                new_index = vertex_index_to_new_index.get(i, -1)
                if new_index == -1:
                    pos = ePoly.verts[i-1].pos
                    pos = pos * mat3
                    orig_positions.extend([pos[0], pos[2], -pos[1]])
                    new_index = len(orig_positions) // 3 - 1
                    vertex_index_to_new_index[i] = new_index

                orig_indices.append(new_index)
        rt.delete(ePoly)
        result_primitives.append(primitive)

    return result_primitives

def checkUseTangents(mMat):
    # to get a reference in a closure-like style
    useTan = [False]

    def setUseTan(matOrTex):
        if (rt.classOf(matOrTex) == rt.Normal_Bump or
                rt.classOf(matOrTex) == getattr(rt, 'ai_normal_map', None)):
            useTan[0] = True

    traverseMtlBase(mMat, setUseTan)

    return useTan[0]

def checkUseVC(mMat):
    # to get a reference in a closure-like style
    useVC = [False]

    def setUseVC(matOrTex):
        if rt.classOf(matOrTex) == rt.Vertex_Color:
            useVC[0] = True

    traverseMtlBase(mMat, setUseVC)

    return useVC[0]

def extractOffsetTM(mNode):
    localOffsetTM = mNode.objectTransform * rt.inverse(mNode.transform)
    return localOffsetTM

def extractValue(c):
    classID = rt.classOf(c)

    if classID == rt.Color:
        return extractColor(c)
    elif classID == rt.Point3:
        return extractVec(c)
    elif classID in [rt.Integer, rt.Double]:
        return c
    elif classID == rt.String:
        return c
    else:
        printLog('WARNING', 'Unsupported value type: ' + str(classID))
        return c


def extractColor(c):
    return [c.r/255.0, c.g/255.0, c.b/255.0]

def extractColorAverage(c):
    return (c.r + c.g + c.b) / 3.0 / 255.0

def extractColorAlpha(c):
    return [c.r/255.0, c.g/255.0, c.b/255.0, c.a/255.0]

def extractColor4(c):
    return [c.r/255.0, c.g/255.0, c.b/255.0, 1.0]

def extractVec(p):
    return [p.x, p.y, p.z]

def extractVecAsColor(p):
    return [p.x/255.0, p.y/255.0, p.z/255.0]

def extractVecAngle(p):
    return [math.radians(p.x), math.radians(p.y), math.radians(p.z)]

def extractQuat(q):
    # left-handed
    return [-q.x, -q.y, -q.z, q.w]

def checkUvTransform(mTex):
    """Return True if texture has non-idenity UV transform"""
    if maxUtils.hasUVParams(mTex):
        mat = mTex.coords.UVTransform
        return not maxUtils.isIdentity(mat)
    else:
        return False

def extractUvTransform(mTex=None):

    identity = [1, 0, 0,
                0, 1, 0,
                0, 0, 1]

    # for some procedural textures UV params will be encoded in the baked image
    if not mTex or not isBitmapTex(mTex):
        return identity

    # NOTE: using maxcpp intead of buggy ParameterBlock
    coords = maxcpp.extractTexCoordsParams(getPtr(mTex))

    uOffset = coords['U_Offset']
    vOffset = coords['V_Offset']
    uTiling = coords['U_Tiling']
    vTiling = coords['V_Tiling']

    rotation = coords['W_angle']

    sx = uTiling
    sy = vTiling

    tx = -uOffset
    ty = -vOffset

    cx = uOffset + 0.5
    cy = vOffset + 0.5

    # based on Matrix3.setUvTransform()
    c = math.cos(rotation)
    s = math.sin(rotation)

    # column major
    return [sx * c,                           -sy * s,                             0,
            sx * s,                            sy * c,                             0,
           -sx * (c * cx + s * cy) + cx + tx, -sy * (- s * cx + c * cy) + cy + ty, 1]


def listProps(obj):
    [print(p, getattr(obj, str(p))) for p in list(rt.getpropnames(obj))]

def listAnims(obj, indent=0):
    for i in range(1, obj.numSubs+1):
        subAnim = rt.getSubAnim(obj, i)
        subAnimName = rt.getSubAnimName(obj, i)

        if subAnim:
            print('  ' * indent + subAnimName + ', ' + str(rt.classOf(subAnim)))
            listAnims(subAnim, indent+1)

def listRefs(obj, indent=0):
    for i in range(1, rt.refs.getNumRefs(obj)+1):
        refTarg = rt.refs.getReference(obj, i)
        if refTarg:
            listRefs(refTarg, indent+1)

def extractProp(obj, nameOrNames, default=None):
    if isinstance(nameOrNames, list):
        for name in nameOrNames:
            if hasattr(obj, name):
                return getattr(obj, name)

        return default

    return getattr(obj, nameOrNames, default)

def extractCustomProp(animatableObj, contName, name, default=None):
    return getattr(getattr(animatableObj, contName, default), name, default)

def processNode(mtlBase, gltf, nodes, edges):

    node = {}

    node['name'] = mtlBase.name
    node['id'] = getPtr(mtlBase)

    mMat = mtlBase if rt.superClassOf(mtlBase) == rt.material else None
    mTex = mtlBase if rt.superClassOf(mtlBase) == rt.textureMap else None

    node['inputs'] = []
    node['outputs'] = []
    node['is_active_output'] = False

    # default connection table: 0,1,2... => 0,1,2...
    connTable = [i for i in range(CONN_TABLE_SIZE)]

    stopTraverse = False

    if mMat and rt.classOf(mMat) == rt.Blend:
        node['type'] = 'BLEND_MX'

        # exporting normalized values
        connTable = processNodeInputs(node, mMat, [
            [0, lambda mTex: [0,0,0,0], ('map1', 'map1Enabled', None), None],
            [1, lambda mTex: [0,0,0,0], ('map2', 'map2Enabled', None), None],
            [2, lambda mTex: mMat.mixAmount / 100, ('mask', 'maskEnabled', None), 'MixAmount'],
        ])

        node['useCurve'] = bool(mMat.mask and mMat.maskEnabled and mMat.useCurve)

        node['curveLower'] = mMat.lower
        node['curveUpper'] = mMat.upper

        node['outputs'].append([0,0,0,0])

    elif mMat and rt.classOf(mMat) == rt.Shellac:
        node['type'] = 'SHELLAC_MX'

        node['inputs'].append([0,0,0,0])
        node['inputs'].append([0,0,0,0])

        node['inputs'].append(mMat.shellacColorBlend / 100)

        node['outputs'].append([0,0,0,0])

    elif mMat and maxUtils.isPhysicalMaterial(mMat):
        node['type'] = 'PHYSICAL_MX'

        node['emitLuminance'] = mMat.emit_luminance

        node['brdfMode'] = mMat.brdf_mode
        node['brdfLow'] = mMat.brdf_low
        node['brdfHigh'] = mMat.brdf_high
        node['brdfCurve'] = mMat.brdf_curve

        node['roughnessInv'] = mMat.roughness_inv
        node['transRoughnessLock'] = mMat.trans_roughness_lock
        node['transRoughnessInv'] = mMat.trans_roughness_inv
        node['thinWalled'] = mMat.thin_walled

        roughnessHack = bool(extractAnimatableController(mMat, 'Coating_Rougness'))

        connTable = processNodeInputs(node, mMat, [
            [ 0, lambda mMat: mMat.base_weight,
                ('base_weight_map', 'base_weight_map_on', None), 'Base_Weight'],
            [ 1, lambda mMat: extractColor4(mMat.base_color),
                ('base_color_map', 'base_color_map_on', None), 'Base_Color'],
            [ 2, lambda mMat: mMat.reflectivity,
                ('reflectivity_map', 'reflectivity_map_on', None), 'Reflectivity'],
            [ 3, lambda mMat: extractColor4(mMat.refl_color),
                ('refl_color_map', 'refl_color_map_on', None), 'Reflection_Color'],
            [ 4, lambda mMat: mMat.roughness,
                ('roughness_map', 'roughness_map_on', None), 'Roughness'],
            [ 5, lambda mMat: mMat.metalness,
                ('metalness_map', 'metalness_map_on', None), 'Metalness'],
            [ 6, lambda mMat: mMat.diff_roughness,
                ('diff_rough_map', 'diff_rough_map_on', None), 'Diffuse_Roughness'],
            [ 7, lambda mMat: mMat.anisotropy,
                ('anisotropy_map', 'anisotropy_map_on', None), 'Anisotropy'],
            [ 8, lambda mMat: mMat.anisoangle,
                ('aniso_angle_map', 'aniso_angle_map_on', None), 'Anisotropy_Angle'],
            [ 9, lambda mMat: mMat.transparency,
                ('transparency_map', 'transparency_map_on', None), 'Transparency'],
            [10, lambda mMat: extractColor4(mMat.trans_color),
                ('trans_color_map', 'trans_color_map_on', None), 'Transparency_Color'],
            [11, lambda mMat: mMat.trans_roughness,
                ('trans_rough_map', 'trans_rough_map_on', None), 'Transparency_Roughness'],
            [12, lambda mMat: mMat.trans_ior,
                ('trans_ior_map', 'trans_ior_map_on', None), 'Index_of_Refraction'],
            [13, lambda mMat: mMat.scattering,
                ('scattering_map', 'scattering_map_on', None), 'Scattering'],
            [14, lambda mMat: extractColor4(mMat.sss_color),
                ('sss_color_map', 'sss_color_map_on', None), 'SSS_Color'],
            [15, lambda mMat: mMat.sss_scale,
                ('sss_scale_map', 'sss_scale_map_on', None), 'SSS_Scale'],
            [16, lambda mMat: mMat.emission,
                ('emission_map', 'emission_map_on', None), 'Emission'],
            [17, lambda mMat: extractColor4(mMat.emit_color),
                ('emit_color_map', 'emit_color_map_on', None), 'Emission_Color'],
            [18, lambda mMat: mMat.coating,
                ('coat_map', 'coat_map_on', None), 'Coating Weight'],
            [19, lambda mMat: extractColor4(mMat.coat_color),
                ('coat_color_map', 'coat_color_map_on', None), 'Coating_Color'],
            [20, lambda mMat: mMat.coat_roughness,
                ('coat_rough_map', 'coat_rough_map_on', None),
                # HACK: workaround for <= 2019 mispelled roughness
                ('Coating_Rougness' if roughnessHack else 'Coating_Roughness')],
            # NOTE: no animation possible since Bump Map Amount affects non-presented input
            [30, lambda mMat: [0,0,0], ('bump_map', 'bump_map_on', 'bump_map_amt'), None],
            # NOTE: no animation possible since Coating Bump Map Amount affects non-presented input
            [31, lambda mMat: [0,0,0],
                ('coat_bump_map', 'coat_bump_map_on', 'clearcoat_bump_map_amt'), None],
            [32, lambda mMat: 0,
                ('displacement_map', 'displacement_map_on', 'displacement_map_amt'), 'Displacement_Map_Amount'],
            [33, lambda mMat: 1, ('cutout_map', 'cutout_map_on', None), None]
        ])

        node['outputs'].append([0,0,0,0])

        if roughnessHack:
            if 'tmpAnimControls' in node:
                for animControl in node['tmpAnimControls']:
                    animControl['name'] = animControl['name'].replace('Coating Rougness', 'Coating Roughness')


    elif mMat and maxUtils.isStandardSurfaceMaterial(mMat):
        node['type'] = 'STANDARD_SURFACE_AR'

        node['thinWalled'] = mMat.thin_walled

        if hasattr(mMat, 'transmission_depth_shader'):
            connTable = processNodeInputs(node, mMat, [
                [ 0, lambda mMat: mMat.base, ('base_shader', 'base_connected', None), 'base'],
                [ 1, lambda mMat: extractColor(mMat.base_color),
                    ('base_color_shader', 'base_color_connected', None), 'base_color'],
                [ 2, lambda mMat: mMat.diffuse_roughness,
                    ('diffuse_roughness_shader', 'diffuse_roughness_connected', None), 'diffuse_roughness'],
                [ 9, lambda mMat: mMat.metalness,
                    ('metalness_shader', 'metalness_connected', None), 'metalness'],
                [ 3, lambda mMat: mMat.specular,
                    ('specular_shader', 'specular_connected', None), 'specular'],
                [ 4, lambda mMat: extractColor(mMat.specular_color),
                    ('specular_color_shader', 'specular_color_connected', None), 'specular_color'],
                [ 5, lambda mMat: mMat.specular_roughness,
                    ('specular_roughness_shader', 'specular_roughness_connected', None), 'specular_roughness'],
                [10, lambda mMat: mMat.transmission,
                    ('transmission_shader', 'transmission_connected', None), 'transmission'],
                [11, lambda mMat: extractColor(mMat.transmission_color),
                    ('transmission_color_shader', 'transmission_color_connected', None), 'transmission_color'],
                [12, lambda mMat: mMat.transmission_depth,
                    ('transmission_depth_shader', 'transmission_depth_connected', None), 'transmission_depth'],
                [13, lambda mMat: extractColor(mMat.transmission_scatter),
                    ('transmission_scatter_shader', 'transmission_scatter_connected', None),
                    'transmission_scatter'],
                [16, lambda mMat: mMat.transmission_extra_roughness,
                    ('transmission_extra_roughness_shader', 'transmission_extra_roughness_connected', None),
                    'transmission_extra_roughness'],
                [17, lambda mMat: mMat.subsurface,
                    ('subsurface_shader', 'subsurface_connected', None), 'subsurface'],
                [18, lambda mMat: extractColor(mMat.subsurface_color),
                    ('subsurface_color_shader', 'subsurface_color_connected', None), 'subsurface_color'],
                [19, lambda mMat: extractColor(mMat.subsurface_radius),
                    ('subsurface_radius_shader', 'subsurface_radius_connected', None), 'subsurface_radius'],
                [27, lambda mMat: mMat.coat, ('coat_shader', 'coat_connected', None), 'coat'],
                [28, lambda mMat: extractColor(mMat.coat_color),
                    ('coat_color_shader', 'coat_color_connected', None), 'coat_color'],
                [29, lambda mMat: mMat.coat_roughness,
                    ('coat_roughness_shader', 'coat_roughness_connected', None), 'coat_roughness'],
                [22, lambda mMat: mMat.sheen, ('sheen_shader', 'sheen_connected', None), 'sheen'],
                [23, lambda mMat: extractColor(mMat.sheen_color),
                    ('sheen_color_shader', 'sheen_color_connected', None), 'sheen_color'],
                [24, lambda mMat: mMat.sheen_roughness,
                    ('sheen_roughness_shader', 'sheen_roughness_connected', None), 'sheen_roughness'],
                [38, lambda mMat: mMat.emission,
                    ('emission', 'emission_connected', None), 'emission'],
                [39, lambda mMat: extractColor(mMat.emission_color),
                    ('emission_color_shader', 'emission_color_connected', None), 'emission_color'],
                [40, lambda mMat: extractColor(mMat.opacity),
                    ('opacity_shader', 'opacity_connected', None), 'opacity'],
                [25, lambda mMat: [0, 0, 0], ('normal_shader', 'normal_connected', None), None],
                [33, lambda mMat: [0, 0, 0], ('coat_normal_shader', 'coat_normal_connected', None), None],
                [ 6, lambda mMat: mMat.specular_IOR,
                    ('specular_IOR_shader', 'specular_IOR_connected', None), 'specular_IOR']
            ])
        # COMPAT: 2020
        else:
            connTable = processNodeInputs(node, mMat, [
                [ 0, lambda mMat: mMat.base, ('base_shader', 'base_connected', None), 'base'],
                [ 1, lambda mMat: extractColor(mMat.base_color),
                    ('base_color_shader', 'base_color_connected', None), 'base_color'],
                [ 2, lambda mMat: mMat.diffuse_roughness,
                    ('diffuse_roughness_shader', 'diffuse_roughness_connected', None), 'diffuse_roughness'],
                [ 9, lambda mMat: mMat.metalness,
                    ('metalness_shader', 'metalness_connected', None), 'metalness'],
                [ 3, lambda mMat: mMat.specular,
                    ('specular_shader', 'specular_connected', None), 'specular'],
                [ 4, lambda mMat: extractColor(mMat.specular_color),
                    ('specular_color_shader', 'specular_color_connected', None), 'specular_color'],
                [ 5, lambda mMat: mMat.specular_roughness,
                    ('specular_roughness_shader', 'specular_roughness_connected', None), 'specular_roughness'],
                [10, lambda mMat: mMat.transmission,
                    ('transmission_shader', 'transmission_connected', None), 'transmission'],
                [11, lambda mMat: extractColor(mMat.transmission_color),
                    ('transmission_color_shader', 'transmission_color_connected', None), 'transmission_color'],
                [-1, lambda mMat: mMat.transmission_depth, None, 'transmission_depth'],
                [12, lambda mMat: extractColor(mMat.transmission_scatter),
                    ('transmission_scatter_shader', 'transmission_scatter_connected', None),
                    'transmission_scatter'],
                [15, lambda mMat: mMat.transmission_extra_roughness,
                    ('transmission_extra_roughness_shader', 'transmission_extra_roughness_connected', None),
                    'transmission_extra_roughness'],
                [16, lambda mMat: mMat.subsurface,
                    ('subsurface_shader', 'subsurface_connected', None), 'subsurface'],
                [17, lambda mMat: extractColor(mMat.subsurface_color),
                    ('subsurface_color_shader', 'subsurface_color_connected', None), 'subsurface_color'],
                [18, lambda mMat: extractColor(mMat.subsurface_radius),
                    ('subsurface_radius_shader', 'subsurface_radius_connected', None), 'subsurface_radius'],
                [26, lambda mMat: mMat.coat, ('coat_shader', 'coat_connected', None), 'coat'],
                [27, lambda mMat: extractColor(mMat.coat_color),
                    ('coat_color_shader', 'coat_color_connected', None), 'coat_color'],
                [28, lambda mMat: mMat.coat_roughness,
                    ('coat_roughness_shader', 'coat_roughness_connected', None), 'coat_roughness'],
                [21, lambda mMat: mMat.sheen,
                    ('sheen_shader', 'sheen_connected', None), 'sheen'],
                [22, lambda mMat: extractColor(mMat.sheen_color),
                    ('sheen_color_shader', 'sheen_color_connected', None), 'sheen_color'],
                [23, lambda mMat: mMat.sheen_roughness,
                    ('sheen_roughness_shader', 'sheen_roughness_connected', None), 'sheen_roughness'],
                [37, lambda mMat: mMat.emission,
                    ('emission', 'emission_connected', None), 'emission'],
                [38, lambda mMat: extractColor(mMat.emission_color),
                    ('emission_color_shader', 'emission_color_connected', None), 'emission_color'],
                [39, lambda mMat: extractColor(mMat.opacity),
                    ('opacity_shader', 'opacity_connected', None), 'opacity'],
                [24, lambda mMat: [0, 0, 0],
                    ('normal_shader', 'normal_connected', None), None],
                [32, lambda mMat: [0, 0, 0],
                    ('coat_normal_shader', 'coat_normal_connected', None), None],
                [ 6, lambda mMat: mMat.specular_IOR,
                    ('specular_IOR_shader', 'specular_IOR_connected', None), 'specular_IOR']
            ])

        node['outputs'].append([0, 0, 0, 0])
        node['outputs'].append([0, 0, 0])

    # NOTE: Arnold is optional in older Max versions
    elif mMat and rt.classOf(mMat) == getattr(rt, 'ArnoldMapToMtl', None):
        node['type'] = 'MAP_TO_MTL_AR'

        node['opaqueEnabled'] = mMat.OpaqueEnabled

        connTable = processNodeInputs(node, mMat, [
            [0, [0, 0, 0, 1], ('SurfaceShader', 'SurfaceShaderEnabled', None), None]
        ])

        node['outputs'].append([0,0,0,0])

    elif mMat and rt.classOf(mMat) == getattr(rt, 'ai_lambert', None):
        node['type'] = 'LAMBERT_AR'

        connTable = processNodeInputs(node, mMat, [
            [0, lambda mMat: mMat.Kd, ('Kd_shader', 'Kd_connected', None), 'Kd'],
            [1, lambda mMat: extractColor(mMat.Kd_color), ('Kd_color_shader', 'Kd_color_connected', None), 'Kd_color'],
            [2, lambda mMat: [0, 0, 0], ('normal_shader', 'normal_connected', None), None],
            [3, lambda mMat: extractColor(mMat.opacity), ('opacity_shader', 'opacity_connected', None), 'opacity']
        ])

        node['outputs'].append([0, 0, 0, 0])
        node['outputs'].append(0)
        node['outputs'].append([0, 0, 0])

    elif mMat and rt.classOf(mMat) == getattr(rt, 'ai_mix_shader', None):
        node['type'] = 'MIX_SHADER_AR'

        node['mode'] = mMat.mode

        connTable = processNodeInputs(node, mMat, [
            [2, lambda mMat: mMat.mix, ('mix_shader', 'mix_connected', None), 'mix'],
            [0, lambda mMat: [0, 0, 0, 1], ('shader1', 'shader1', None), None],
            [1, lambda mMat: [0, 0, 0, 1], ('shader2', 'shader2', None), None]
        ])

        node['outputs'].append([0, 0, 0, 0])

    elif mMat and rt.classOf(mMat) == getattr(rt, 'ai_ray_switch_shader', None):
        node['type'] = 'RAY_SWITCH_AR'

        connTable = processNodeInputs(node, mMat, [
            [0, lambda mMat: [0, 0, 0], None, None],
            [2, lambda mMat: [0, 0, 0], None, None],
            [3, lambda mMat: [0, 0, 0], None, None],
            [1, lambda mMat: [0, 0, 0], None, None],
            [4, lambda mMat: [0, 0, 0], None, None],
            [5, lambda mMat: [0, 0, 0], None, None],
        ])

        node['outputs'].append([0, 0, 0])
        node['outputs'].append(1)

    elif mMat and rt.classOf(mMat) == getattr(rt, 'ai_two_sided', None):
        node['type'] = 'TWO_SIDED_AR'

        connTable = processNodeInputs(node, mMat, [
            [0, lambda mMat: [0, 0, 0], None, None],
            [1, lambda mMat: [0, 0, 0], None, None],

        ])

        node['outputs'].append([0, 0, 0])

    elif mMat and rt.classOf(mMat) == rt.Shell_Material:
        if mMat.bakedMaterial is not None:
            processNode(mMat.bakedMaterial, gltf, nodes, edges)
            # the Shell node is just replaced by its baked material, so don't
            # export anything for the Shell node itself
            return
        else:
            node['type'] = 'MATERIAL_MX'
            inputDefs = [
                [-1, lambda mMat: [1, 1, 1, 1], None, None],
                [-1, lambda mMat: [1, 1, 1, 1], None, None],
                [-1, lambda mMat: [0, 0, 0, 1], None, None],
                [-1, lambda mMat: 0, None, None],
                [-1, lambda mMat: 0, None, None],
                [-1, lambda mMat: [0, 0, 0, 1], None, None],
                [-1, lambda mMat: 1.0, None, None],
                [-1, lambda mMat: [0,0,0,0], None, None],
                [-1, lambda mMat: [0,0,0], None, None],
                [-1, lambda mMat: [0,0,0,0], None, None],
                [-1, lambda mMat: [0,0,0,0], None, None],
                [-1, lambda mMat: 0, None, None]
            ]

            node['inputFactors'] = [0] * len(inputDefs)
            node['selfIllumColorOn'] = False
            node['IOR'] = DEFAULT_IOR
            connTable = processNodeInputs(node, mMat, inputDefs)
            node['outputs'].append([0,0,0,0])

            # the Shell node is exported as a dummy node, so don't traverse
            # further into the originalMaterial submaterial
            stopTraverse = True

    elif mMat and rt.classOf(mMat) == rt.MatteShadow:
        node['type'] = 'MATTE_SHADOW_MX'

        node['receiveShadows'] = mMat.receiveshadows
        node['shadowBrightness'] = mMat.ShadowBrightness
        node['color'] = extractColor(mMat.color)

        node['outputs'].append([0,0,0,0])

    elif mMat:
        node['type'] = 'MATERIAL_MX'

        if maxUtils.isStandardMaterial(mMat):
            mStdMat = mMat

            inputDefs = [
                [ 0, lambda mMat: extractColor4(mMat.ambient),
                    ('ambientMap', 'ambientMapEnable', 'ambientMapAmount'),
                    ('Shader_Basic_Parameters', 'Ambient_Color')],
                [ 1, lambda mMat: extractColor4(mMat.diffuse),
                    ('diffuseMap', 'diffuseMapEnable', 'diffuseMapAmount'),
                    ('Shader_Basic_Parameters', 'Diffuse_Color')],
                [ 2, lambda mMat: extractColor4(mMat.specular),
                    ('specularMap', 'specularMapEnable', 'specularMapAmount'),
                    ('Shader_Basic_Parameters', 'Specular_Color')],
                [ 3, lambda mMat: mMat.Glossiness / 100,
                    ('glossinessMap', 'glossinessMapEnable', 'glossinessMapAmount'),
                    ('Shader_Basic_Parameters', 'glossiness')],
                [ 4, lambda mMat: mMat.specularLevel / 100,
                    ('specularLevelMap', 'specularLevelMapEnable', 'specularLevelMapAmount'),
                    ('Shader_Basic_Parameters', 'Specular_Level')],
                [ 5, lambda mMat: extractColor4(mMat.selfIllumColor),
                    ('selfIllumMap', 'selfIllumMapEnable', 'selfIllumMapAmount'),
                    ('Shader_Basic_Parameters', 'Self_Illum_Color')],
                [ 6, lambda mMat: mMat.opacity / 100,
                    ('opacityMap', 'opacityMapEnable', 'opacityMapAmount'),
                    ('Extended_Parameters', 'opacity')],
                [ 7, lambda mMat: [0,0,0,0],
                    ('filterMap', 'filterMapEnable', 'filterMapAmount'),
                    ('Extended_Parameters', 'filter_Color')],
                [ 8, lambda mMat: [0,0,0],
                    ('bumpMap', 'bumpMapEnable', 'bumpMapAmount'), None],
                [ 9, lambda mMat: [0,0,0,0],
                    ('reflectionMap', 'reflectionMapEnable', 'reflectionMapAmount'), None],
                [10, lambda mMat: [0,0,0,0],
                    ('refractionMap', 'refractionMapEnable', 'refractionMapAmount'), None],
                [11, lambda mMat: 0,
                    ('displacementMap', 'displacementMapEnable', 'displacementMapAmount'), None]
            ]

            if mMat.useSelfIllumColor:
                node['selfIllumColorOn'] = True
            else:
                node['selfIllumColorOn'] = False

                def selfIllumInputFunc(mMat):
                    si = mMat.selfIllumAmount
                    return [si/100, si/100, si/100, 1.0]

                inputDefs[5][1] = selfIllumInputFunc

            node['IOR'] = mStdMat.ior

            connTable = processNodeInputs(node, mStdMat, inputDefs, 1/100)

        else:
            printLog('WARNING', 'Unsupported material type: ' + str(rt.classOf(mMat)))

            inputDefs = [
                [-1, lambda mMat: [1, 1, 1, 1], None, None],
                [-1, lambda mMat: [1, 1, 1, 1], None, None],
                [-1, lambda mMat: [0, 0, 0, 1], None, None],
                [-1, lambda mMat: 0, None, None],
                [-1, lambda mMat: 0, None, None],
                [-1, lambda mMat: [0, 0, 0, 1], None, None],
                [-1, lambda mMat: 1.0, None, None],
                [-1, lambda mMat: [0,0,0,0], None, None],
                [-1, lambda mMat: [0,0,0], None, None],
                [-1, lambda mMat: [0,0,0,0], None, None],
                [-1, lambda mMat: [0,0,0,0], None, None],
                [-1, lambda mMat: 0, None, None]
            ]

            node['inputFactors'] = [0] * len(inputDefs)
            node['selfIllumColorOn'] = True
            node['IOR'] = DEFAULT_IOR

            connTable = processNodeInputs(node, mMat, inputDefs, 1/100)

        node['outputs'].append([0,0,0,0])

    elif mTex:
        bitMapTex = mTex if isBitmapTex(mTex) else None
        classID = rt.classOf(mTex)

        if bitMapTex:
            index = getTextureIndex(gltf, getPtr(bitMapTex))

            if index == -1:
                node['type'] = 'BITMAP_NONE_MX'
            else:
                node['type'] = 'BITMAP_MX'
                node['texture'] = index

                node['uvIndex'] = bitMapTex.coords.mapChannel - 1

                node['clampToEdgeNoExtend'] = [not bitMapTex.coords.U_Tile,
                                               not bitMapTex.coords.V_Tile]
                # true - alpha, false - rgb intensity
                node['alphaAsMono'] = bool(bitMapTex.monoOutput)
                # true - alpha, false - rgb
                node['alphaAsRGB'] = bool(bitMapTex.RGBOutput)

                if bitMapTex.alphaSource == 0:
                    node['alphaSource'] = 'FILE'
                elif bitMapTex.alphaSource == 1:
                    node['alphaSource'] = 'RGB'
                else:
                    node['alphaSource'] = 'NONE'

                mapType = bitMapTex.coords.mappingType
                if mapType == 1:
                    node['type'] = 'BITMAP_ENV_MX'

                processNodeOutput(node, mTex.output)

                coords = maxcpp.extractTexCoordsParams(getPtr(mTex))

                node['mapping'] = coords['mapping']
                node['axis'] = coords['axis']

                connTable = processNodeInputs(node, bitMapTex, [
                    [0, lambda bitMapTex: coords['U_Offset'], None, ('Coordinates', 'U_Offset')],
                    [1, lambda bitMapTex: coords['V_Offset'], None, ('Coordinates', 'V_Offset')],
                    [2, lambda bitMapTex: coords['U_Tiling'], None, ('Coordinates', 'U_Tiling')],
                    [3, lambda bitMapTex: coords['V_Tiling'], None, ('Coordinates', 'V_Tiling')],
                    [4, lambda bitMapTex: coords['W_angle'], None, ('Coordinates', 'W_Angle')],
                ])

            node['outputs'].append([0,0,0,0])

        elif classID == rt.Color_Correction:
            node['type'] = 'COLOR_CORRECTION_MX'

            node['rewireR'] = mTex.rewireR
            node['rewireG'] = mTex.rewireG
            node['rewireB'] = mTex.rewireB
            node['rewireA'] = mTex.rewireA

            # exporting multiplied values
            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColorAlpha(mTex.color), None, 'Color_1'],
                [1, lambda mTex: mTex.hueShift, None, 'HueShift'],
                [2, lambda mTex: mTex.saturation, None, 'Saturation'],
                [3, lambda mTex: extractColor4(mTex.tint), None, 'Hue_Tint'],
                [4, lambda mTex: mTex.tintStrength, None, 'Hue_Strength'],
                [5, lambda mTex: mTex.brightness, None, 'Brightness'],
                [6, lambda mTex: mTex.contrast, None, 'Contrast']
            ])

            node['outputs'].append([0,0,0,0])
        elif classID == rt.ColorMap:
            node['type'] = 'COLOR_MAP_MX'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor4(mTex.solidcolor),
                    ('map', 'mapEnabled', None), 'Solid_Color'],
                [1, lambda mTex: mTex.gamma, None, 'Gamma'],
                [2, lambda mTex: mTex.gain, None, 'Gain']
            ])

            node['reverseGamma'] = mTex.ReverseGamma

            node['outputs'].append([0,0,0,0])

        elif classID == rt.CompositeTexturemap:
            node['type'] = 'COMPOSITE_MX'
            numLayers = len(mTex.layerName)

            node['mapEnabled'] = []
            node['maskEnabled'] = []
            node['blendMode'] = []
            node['opacity'] = []

            for i in range(numLayers):

                node['mapEnabled'].append(mTex.mapEnabled[i])
                node['maskEnabled'].append(mTex.maskEnabled[i])
                node['blendMode'].append(mTex.blendMode[i])
                node['opacity'].append(mTex.opacity[i] / 100)

                # foreground
                node['inputs'].append([0,0,0,1])
                # mask
                node['inputs'].append([1,1,1,1])

                node['outputs'].append([0,0,0,0])

        elif classID == rt.falloff:
            node['type'] = 'FALLOFF_MX'
            node['IOR'] = mTex.ior

            inputFactors = []
            inputFactors.append(mTex.map1Amount / 100)
            inputFactors.append(mTex.map2Amount / 100)
            node['inputFactors'] = inputFactors

            node['falloffType'] = mTex.type
            node['falloffDirection'] = mTex.direction
            node['mtlIOROverride'] = mTex.mtlIOROverride

            node['inputs'].append(extractColor4(mTex.color1))
            node['inputs'].append(extractColor4(mTex.color2))
            node['outputs'].append([0,0,0,0])

        elif classID == rt.Mask:
            node['type'] = 'MASK_MX'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: [1,1,1,1], ('map', 'mapEnabled', None), None],
                [1, lambda mTex: [1,1,1,1], ('mask', 'maskEnabled', None), None],
            ])

            node['maskInverted'] = mTex.maskInverted

            node['outputs'].append([0,0,0,0])

        elif classID == rt.Mix:
            node['type'] = 'MIX_MX'

            # exporting normalized values
            connTable = processNodeInputs(node, mTex, [
                [ 0, lambda mTex: extractColor4(mTex.color1), None, 'Color_1'],
                [ 1, lambda mTex: extractColor4(mTex.color2), None, 'Color_2'],
                [ 2, lambda mTex: mTex.mixAmount / 100, None, 'MixAmount']
            ])
            node['outputs'].append([0,0,0,0])

        elif classID == rt.MultiOutputChannelTexmapToTexmap:

            node['type'] = 'OSL_OUTPUT_SELECTOR_MX'

            oslCode = preprocessOSL(mTex.sourceMap.oslCode)
            oslAST = pyosl.oslparse.get_ast(oslCode)
            _, outputs = parseOSLInOuts(oslAST, '')

            output = outputs[mTex.outputChannelIndex-1]

            connTable = processNodeInputs(node, mTex, [[0, lambda mTex: output[2], None, None]])

            node['outputs'] = [output[2]]


        elif classID == rt.Noise:
            node['type'] = 'NOISE_MX'

            node['noiseType'] = mTex.type
            node['coordType'] = mTex.coords.coordType
            node['uvIndex'] = mTex.coords.mapChannel - 1

            processNodeOutput(node, mTex.output)

            connTable = processNodeInputs(node, mTex, [
                [ 0, lambda mTex: extractColor(mTex.color1), ('map1', 'map1Enabled', None), 'Color_1'],
                [ 1, lambda mTex: extractColor(mTex.color2), ('map2', 'map2Enabled', None), 'Color_2'],
                [ 2, lambda mTex: mTex.size, None, 'Noise_Size'],
                [ 3, lambda mTex: mTex.thresholdLow, None, 'Low_Threshold'],
                [ 4, lambda mTex: mTex.thresholdHigh, None, 'High_Threshold'],
                [ 5, lambda mTex: mTex.levels, None, 'Noise_Levels'],
                [ 6, lambda mTex: mTex.phase, None, 'phase'],
                [ 7, lambda mTex: extractVec(mTex.coords.offset), None, ('Coordinates', 'offset')],
                [ 8, lambda mTex: extractVec(mTex.coords.tiling), None, ('Coordinates', 'tiling')],
                [ 9, lambda mTex: extractVecAngle(mTex.coords.angle), None, ('Coordinates', 'angle')],
            ])
            node['outputs'].append([0,0,0,0])

        elif classID == rt.Normal_Bump:
            node['type'] = 'NORMAL_BUMP_MX'

            node['flip'] = [mTex.flipred, mTex.flipgreen]

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: [0,0,0,0], ('normal_map', 'map1on', None), None],
                [1, lambda mTex: [0,0,0], ('bump_map', 'map2on', None), None],
                [2, lambda mTex: mTex.mult_spin, None, 'Multiplier'],
                [3, lambda mTex: mTex.bump_spin, None, 'Bump_Multiplier']
            ])

            node['outputs'].append([0,0,0])

        elif classID == rt.OSLMap:
            node['type'] = 'OSL_NODE'

            shaderName = 'node_osl_' + mTex.OSLShaderName.lower()

            # NOTE: fix mispelled shader name
            if mTex.OSLPath and os.path.basename(mTex.OSLPath) == 'Color1ofN.osl':
                shaderName = 'node_osl_color1ofn'

            node['shaderName'] = shaderName

            oslCode = preprocessOSL(mTex.oslCode)
            oslAST = pyosl.oslparse.get_ast(oslCode)
            inputs, outputs = parseOSLInOuts(oslAST, shaderName)

            if isOSLBitmapTex(mTex):
                index = getTextureIndex(gltf, getPtr(mTex))
                node['texture'] = index

            node['globalVariables'] = [varName for _, varName in pyosl.glslgen.find_global_variables(oslAST)]

            inputDefs = []
            inputTypes = []
            initializers = []

            for i in range(len(inputs)):
                name = inputs[i][1]

                if name == 'Filename' or name == 'HDRI' or name == 'LightName1':
                    ext = os.path.splitext(getattr(mTex, name))[1]
                    value = pyosl.glslgen.string_to_osl_const(ext)
                elif name == 'Filename_UDIMList' or name == 'LoadUDIM':
                    value = 'OSL_EMPTY'
                else:
                    value = extractValue(getattr(mTex, name)) if hasattr(mTex, name) else inputs[i][2]
                    # COMPAT: < 2021
                    if type(value) == str or (sys.version_info[0] == 2 and type(value) == unicode):
                        value = pyosl.glslgen.string_to_osl_const(value)

                inputDefs.append([i, value, None, name])
                inputTypes.append(inputs[i][0])

                if inputs[i][3]:
                    initializers.append([inputs[i][3], inputs[i][4]])
                else:
                    initializers.append(None)

            node['initializers'] = initializers

            node['inputTypes'] = inputTypes
            node['outputTypes'] = []

            connTable = processNodeInputs(node, mTex, inputDefs)

            for o in outputs:
                node['outputs'].append(o[2])
                node['outputTypes'].append(o[0])

            node['fragCode'] = genOSLCode(oslAST, shaderName)


        elif classID == rt.output:
            node['type'] = 'OUTPUT_MAP_MX'

            output = maxcpp.extractTexOutParams(getPtr(mTex.output), CURVE_DATA_SIZE)

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: [1,1,1,1], ('map1', 'map1Enabled', None), None],
                [1, lambda mTex: output['rgbLevel'], None, ('Output', 'RGB_Level')],
                [2, lambda mTex: output['rgbOffset'], None, ('Output', 'RGB_Offset')],
                [3, lambda mTex: output['outputAmount'], None, ('Output', 'Output_Amount')],
                [4, lambda mTex: output['bumpAmount'], None, ('Output', 'Bump_Amount')]
            ])

            node['alphaFromRGB'] = output['alphaFromRGB']
            node['clamp'] = output['clamp']
            node['invert'] = output['invert']

            if 'colorMap' in output:
                node['colorMap'] = output['colorMap']

            node['outputs'].append([0,0,0,0])

        elif classID == rt.PhysicalSunSkyEnv:
            node['type'] = 'PHY_SUN_SKY_ENV_MX'

            node['globalIntensity'] = mTex.global_intensity
            node['groundColor'] = extractColor4(mTex.ground_color)
            node['haze'] = mTex.haze

            sunPos = maxUtils.getSunPosition(mTex.sun_position_object)
            node['sunAzimuthAngle'] = sunPos[0]
            node['sunPolarAngle'] = sunPos[1]

            node['outputs'].append([0,0,0,0])

        elif classID == rt.Reflect_Refract:

            envTex = extractEnvMap()
            if envTex and mTex.useAtmosphericMap:
                node['type'] = 'REFLECT_REFRACT_MX'
                node['texture'] = getTextureIndex(gltf, getPtr(envTex))
            else:
                node['type'] = 'REFLECT_REFRACT_COLOR_MX'

            envColor = extractEnvColor()
            envColor.append(1.0)
            node['outputs'].append(envColor)

        elif classID == rt.RGB_Multiply:
            node['type'] = 'RGB_MULTIPLY_MX'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor4(mTex.color1), None, 'Color_1'],
                [1, lambda mTex: extractColor4(mTex.color2), None, 'Color_2']
            ])

            node['outputs'].append([0,0,0,0])

        elif classID == rt.RGB_Tint:
            node['type'] = 'RGB_TINT_MX'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: [1,1,1,1], ('map1', 'map1Enabled', None), None],
                [1, lambda mTex: extractColor4(mTex.red), None, 'Red'],
                [2, lambda mTex: extractColor4(mTex.green), None, 'Green'],
                [3, lambda mTex: extractColor4(mTex.blue), None, 'Blue']
            ])

            node['outputs'].append([0,0,0,0])

        elif classID == rt.Vertex_Color:
            node['type'] = 'VERTEX_COLOR_MX'
            node['outputs'].append([0,0,0,0])

        elif classID == rt.Gradient_Ramp and not gradRampNeedsConversion(mTex):
            node['type'] = 'GRADIENT_RAMP_MX'

            node['gradientData'] = extractGradientRampData(mTex)
            node['gradientType'] = mTex.Gradient_Type
            node['uvIndex'] = mTex.Coordinates.mapChannel - 1

            node['clampToEdgeNoExtend'] = [not mTex.Coordinates.U_Tile,
                                           not mTex.Coordinates.V_Tile]

            coords = maxcpp.extractTexCoordsParams(getPtr(mTex))

            node['mapping'] = coords['mapping']
            node['axis'] = coords['axis']

            connTable = processNodeInputs(node, mTex, [
                # source map
                [0, lambda mTex: [0,0,0,0], None, None],
                [1, lambda mTex: coords['U_Offset'], None, ('Coordinates', 'U_Offset')],
                [2, lambda mTex: coords['V_Offset'], None, ('Coordinates', 'V_Offset')],
                [3, lambda mTex: coords['U_Tiling'], None, ('Coordinates', 'U_Tiling')],
                [4, lambda mTex: coords['V_Tiling'], None, ('Coordinates', 'V_Tiling')],
                [5, lambda mTex: coords['W_angle'], None, ('Coordinates', 'W_Angle')],
            ])

            node['outputs'].append([0,0,0,0])

            processNodeOutput(node, mTex.output)

        # Arnold maps

        elif classID == getattr(rt, 'ai_abs', None):
            node['type'] = 'ABS_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_add', None):
            node['type'] = 'ADD_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input1), ('input1_shader', 'input1_connected', None), None],
                [1, lambda mTex: extractColor(mTex.input2), ('input2_shader', 'input2_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_atan', None):
            node['type'] = 'ATAN_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.y), ('y_shader', 'y_connected', None), None],
                [1, lambda mTex: extractColor(mTex.x), ('x_shader', 'x_connected', None), None]
            ])

            node['units'] = mTex.units

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_checkerboard', None):
            node['type'] = 'CHECKERBOARD_AR'

            node['uvIndex'] = 0

            connTable = processNodeInputs(node, mTex, [
                # source map
                [0, lambda mTex: extractColor(mTex.color1), None, ('color1_shader', 'color1_connected')],
                [1, lambda mTex: extractColor(mTex.color2), None, ('color2_shader', 'color2_connected')],
                [2, lambda mTex: mTex.u_frequency, None, ('u_frequency_shader', 'u_frequency_connected')],
                [3, lambda mTex: mTex.v_frequency, None, ('v_frequency_shader', 'v_frequency_connected')],
                [4, lambda mTex: mTex.u_offset, None, ('u_offset_shader', 'u_offset_connected')],
                [5, lambda mTex: mTex.v_offset, None, ('v_offset_shader', 'v_offset_connected')],
                [6, lambda mTex: mTex.contrast, None, ('contrast_shader', 'contrast_connected')],
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_clamp', None):
            node['type'] = 'CLAMP_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None],
                [2, lambda mTex: mTex.min, ('min_shader', 'min_connected', None), None],
                [3, lambda mTex: mTex.max, ('max_shader', 'max_connected', None), None],
                [4, lambda mTex: extractColor(mTex.min_color), ('min_color_shader', 'min_color_connected', None), None],
                [5, lambda mTex: extractColor(mTex.max_color), ('max_color_shader', 'max_color_connected', None), None],
            ])
            node['mode'] = mTex.mode

            node['outputs'].append([0, 0, 0])
            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_color_convert', None):
            node['type'] = 'COLOR_CONVERT_AR'

            node['from'] = 0
            node['to'] = 0
            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None],
            ])

            node['outputs'].append([0, 0, 0])
            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_color_correct', None):
            node['type'] = 'COLOR_CORRECT_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None],
                [14, lambda mTex: mTex.Mask, ('mask_shader', 'mask_connected', None), None],
                [6, lambda mTex: mTex.gamma, ('gamma_shader', 'gamma_connected', None), None],
                [7, lambda mTex: mTex.hue_shift, ('hue_shift_shader', 'hue_shift_connected', None), None],
                [8, lambda mTex: mTex.saturation, ('saturation_shader', 'saturation_connected', None), None],
                [9, lambda mTex: mTex.contrast, ('contrast_shader', 'contrast_connected', None), None],
                [10, lambda mTex: mTex.contrast_pivot, ('contrast_pivot_shader', 'contrast_pivot_connected', None), None],
                [11, lambda mTex: mTex.exposure, ('exposure_shader', 'exposure_connected', None), None],
                [12, lambda mTex: extractColor(mTex.multiply), ('multiply_shader', 'multiply_connected', None), None],
                [13, lambda mTex: extractColor(mTex.add), ('add_shader', 'add_connected', None), None],
                [4, lambda mTex: mTex.invert, None, None],
                [1, lambda mTex: mTex.alpha_is_luminance, None, None],
                [2, lambda mTex: mTex.alpha_multiply, ('alpha_multiply_shader', 'alpha_multiply_connected', None), None],
                [3, lambda mTex: mTex.alpha_add, ('alpha_add_shader', 'alpha_add_connected', None), None],
                [5, lambda mTex: mTex.invert_alpha, None, None]

            ])

            node['outputs'].append([0, 0, 0])
            node['outputs'].append(0)

        elif classID == getattr(rt, 'ai_compare', None):
            node['type'] = 'COMPARE_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input1), ('input1_shader', 'input1_connected', None), None],
                [1, lambda mTex: extractColor(mTex.input2), ('input2_shader', 'input2_connected', None), None]
            ])

            node['test'] = mTex.test

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_complement', None):
            node['type'] = 'COMPLEMENT_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_cross', None):
            node['type'] = 'CROSS_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input1), ('input1_shader', 'input1_connected', None), None],
                [1, lambda mTex: extractColor(mTex.input2), ('input2_shader', 'input2_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_divide', None):
            node['type'] = 'DIVIDE_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input1), ('input1_shader', 'input1_connected', None), None],
                [1, lambda mTex: extractColor(mTex.input2), ('input2_shader', 'input2_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_dot', None):
            node['type'] = 'DOT_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input1), ('input1_shader', 'input1_connected', None), None],
                [1, lambda mTex: extractColor(mTex.input2), ('input2_shader', 'input2_connected', None), None]
            ])

            node['outputs'].append(0)

        elif classID == getattr(rt, 'ai_exp', None):
            node['type'] = 'EXP_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_facing_ratio', None):
            node['type'] = 'FACING_RATIO_AR'

            connTable = processNodeInputs(node, mTex, [
                [-1, lambda mTex: mTex.bias, None, 'bias'],
                [-1, lambda mTex: mTex.gain, None, 'gain'],
                [-1, lambda mTex: mTex.invert, None, None],
                [-1, lambda mTex: mTex.linear, None, None]
            ])

            node['outputs'].append(0)

        elif classID == getattr(rt, 'ai_flat', None):
            node['type'] = 'FLAT_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.color), ('color_shader', 'color_connected', None), 'color']
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_float_to_int', None):
            node['type'] = 'FLOAT_TO_INT_AR'

            node['mode'] = mTex.mode

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: mTex.input, ('input_shader', 'input_connected', None), None],
            ])

            node['outputs'].append(0)

        elif classID == getattr(rt, 'ai_float_to_matrix', None):
            node['type'] = 'FLOAT_TO_MATRIX_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: mTex.input_00, ('input_00_shader', 'input_00_connected', None), None],
                [1, lambda mTex: mTex.input_01, ('input_01_shader', 'input_01_connected', None), None],
                [2, lambda mTex: mTex.input_02, ('input_02_shader', 'input_02_connected', None), None],
                [3, lambda mTex: mTex.input_03, ('input_03_shader', 'input_03_connected', None), None],
                [4, lambda mTex: mTex.input_10, ('input_10_shader', 'input_10_connected', None), None],
                [5, lambda mTex: mTex.input_11, ('input_11_shader', 'input_11_connected', None), None],
                [6, lambda mTex: mTex.input_12, ('input_12_shader', 'input_12_connected', None), None],
                [7, lambda mTex: mTex.input_13, ('input_13_shader', 'input_13_connected', None), None],
                [8, lambda mTex: mTex.input_20, ('input_20_shader', 'input_20_connected', None), None],
                [9, lambda mTex: mTex.input_21, ('input_21_shader', 'input_21_connected', None), None],
                [10, lambda mTex: mTex.input_22, ('input_22_shader', 'input_22_connected', None), None],
                [11, lambda mTex: mTex.input_23, ('input_23_shader', 'input_23_connected', None), None],
                [12, lambda mTex: mTex.input_30, ('input_30_shader', 'input_30_connected', None), None],
                [13, lambda mTex: mTex.input_31, ('input_31_shader', 'input_31_connected', None), None],
                [14, lambda mTex: mTex.input_32, ('input_32_shader', 'input_32_connected', None), None],
                [15, lambda mTex: mTex.input_33, ('input_33_shader', 'input_33_connected', None), None],
            ])

            node['outputs'].append([0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0])

        elif classID == getattr(rt, 'ai_float_to_rgb', None):
            node['type'] = 'FLOAT_TO_RGB_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: mTex.r, ('r_shader', 'r_connected', None), None],
                [1, lambda mTex: mTex.g, ('g_shader', 'g_connected', None), None],
                [2, lambda mTex: mTex.b, ('b_shader', 'b_connected', None), None],
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_float_to_rgba', None):
            node['type'] = 'FLOAT_TO_RGBA_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: mTex.r, ('r_shader', 'r_connected', None), None],
                [1, lambda mTex: mTex.g, ('g_shader', 'g_connected', None), None],
                [2, lambda mTex: mTex.b, ('b_shader', 'b_connected', None), None],
                [3, lambda mTex: mTex.a, ('a_shader', 'a_connected', None), None],
            ])

            node['outputs'].append([0, 0, 0, 0])
            node['outputs'].append(0)
            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_fraction', None):
            node['type'] = 'FRACTION_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_image', None):
            index = getTextureIndex(gltf, getPtr(mTex))

            if index == -1:
                printLog('ERROR', 'Texture not found: ' + mTex.name)
                node['type'] = 'IMAGE_AR'
            else:

                node['type'] = 'IMAGE_AR'
                node['texture'] = index


            connTable = processNodeInputs(node, mTex, [
                # source map
                [1, lambda mTex: extractColorAlpha(mTex.multiply), ('multiply_shader', 'multiply_connected', None), None],# extractColor(mTex.multiply), ('multiply_shader', 'multiply_connected'), None],
                [2, lambda mTex: extractColorAlpha(mTex.offset), ('offset_shader', 'offset_connected', None), None],
                [0, lambda mTex: [mTex.uvcoords[0], mTex.uvcoords[1]], ('uvcoords_shader', 'uvcoords_connected', None), None],
                [3, lambda mTex: mTex.sOffset, None, 'sOffset'],
                [4, lambda mTex: mTex.toffset, None, 'toffset'],
                [5, lambda mTex: mTex.sscale,  None, 'sscale'],
                [6, lambda mTex: mTex.tscale, None, 'tscale'],
                [7, lambda mTex: mTex.sflip, None, 'sflip'],
                [8, lambda mTex: mTex.tflip, None, 'tflip'],
                [9, lambda mTex: mTex.swap_st, None, 'swap_st'],

            ])

            node['outputs'].append([0, 0, 0, 0])

        elif classID == getattr(rt, 'ai_is_finite', None):
            node['type'] = 'IS_FINITE_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None]
            ])

            node['outputs'].append(0)

        elif classID == getattr(rt, 'ai_length', None):
            node['type'] = 'LENGTH_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractVec(mTex.input), ('input_shader', 'input_connected', None), None]
            ])

            node['mode'] = mTex.mode

            node['outputs'].append(0)

        elif classID == getattr(rt, 'ai_log', None):
            node['type'] = 'LOG_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None],
                [1, lambda mTex: extractColor(mTex.base), ('base_shader', 'base_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_max', None):
            node['type'] = 'MAX_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input1), ('input1_shader', 'input1_connected', None), None],
                [1, lambda mTex: extractColor(mTex.input2), ('input2_shader', 'input2_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_min', None):
            node['type'] = 'MIN_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input1), ('input1_shader', 'input1_connected', None), None],
                [1, lambda mTex: extractColor(mTex.input2), ('input2_shader', 'input2_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_modulo', None):
            node['type'] = 'MODULO_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None],
                [1, lambda mTex: extractColor(mTex.divisor), ('divisor_shader', 'divisor_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_multiply', None):
            node['type'] = 'MULTIPLY_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input1), ('input1_shader', 'input1_connected', None), None],
                [1, lambda mTex: extractColor(mTex.input2), ('input2_shader', 'input2_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_negate', None):
            node['type'] = 'NEGATE_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_normalize', None):
            node['type'] = 'NORMALIZE_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_normal_map', None):
            node['type'] = 'NORMAL_MAP_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: [0, 0, 0], ('input_shader', 'input_connected', None), None],
                [2, lambda mTex: [0, 0, 0], ('normal_shader', 'normal_connected', None), None],
                [3, lambda mTex: mTex.strength, ('strength_shader', 'strength_connected', None), 'strength'],
                [1, lambda mTex: [0, 0, 0], ('tangent_shader', 'tangent_connected', None), None],
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_pow', None):
            node['type'] = 'POW_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.base), ('base_shader', 'base_connected', None), None],
                [1, lambda mTex: extractColor(mTex.exponent), ('exponent_shader', 'exponent_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_random', None):
            node['type'] = 'RANDOM_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_color_shader', 'input_color_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_range', None):
            node['type'] = 'RANGE_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None],
                [1, lambda mTex: mTex.input_min, ('input_min_shader', 'input_min_connected', None), None],
                [2, lambda mTex: mTex.input_max, ('input_max_shader', 'input_max_connected', None), None],
                [3, lambda mTex: mTex.output_min, ('output_min_shader', 'output_min_connected', None), None],
                [4, lambda mTex: mTex.output_max, ('output_max_shader', 'output_max_connected', None), None],
                [5, lambda mTex: mTex.smoothstep, None, None],
                [6, lambda mTex: mTex.contrast, ('contrast_shader', 'contrast_connected', None), None],
                [7, lambda mTex: mTex.contrast_pivot, ('contrast_pivot_shader', 'contrast_pivot_connected', None), None],
                [8, lambda mTex: mTex.bias, ('bias_shader', 'bias_connected', None), None],
                [9, lambda mTex: mTex.gain, ('gain_shader', 'gain_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])
            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_reciprocal', None):
            node['type'] = 'RECIPROCAL_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_rgb_to_vector', None):
            node['type'] = 'RGB_TO_VECTOR_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None],
            ])
            node['mode'] = mTex.mode

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_rgb_to_float', None):
            node['type'] = 'RGB_TO_FLOAT_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None],
            ])

            node['mode'] = mTex.mode

            node['outputs'].append(0)

        elif classID == getattr(rt, 'ai_rgba_to_float', None):
            node['type'] = 'RGBA_TO_FLOAT_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None],
                [1, lambda mTex: extractColorAlpha(mTex.input)[3], None, None],
            ])
            node['mode'] = mTex.mode

            node['outputs'].append(0)
            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_shadow_matte', None):
            node['type'] = 'SHADOW_MATTE_AR'

            connTable = processNodeInputs(node, mTex, [
                [1, lambda mTex: extractColor(mTex.shadow_color), ('shadow_color_shader', 'shadow_color_connected', None), None],
                [6, lambda mTex: mTex.backlighting, ('backlighting_shader', 'backlighting_connected', None), None],
                [2, lambda mTex: mTex.shadow_opacity, ('shadow_opacity_shader', 'shadow_opacity_connected', None), None],
            ])

            node['outputs'].append([0, 0, 0, 0])
            node['outputs'].append(1)

        elif classID == getattr(rt, 'ai_shuffle', None):
            node['type'] = 'SHUFFLE_AR'

            node['channelR'] = mTex.channel_r
            node['channelG'] = mTex.channel_g
            node['channelB'] = mTex.channel_b
            node['channelA'] = mTex.channel_a

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColorAlpha(mTex.color), ('color_shader', 'color_connected', None), None],
                [1, lambda mTex: mTex.alpha, ('alpha_shader', 'alpha_connected', None), None],
                [2, lambda mTex: mTex.negate_r, None, 'negateR'],
                [3, lambda mTex: mTex.negate_g, None, 'negateG'],
                [4, lambda mTex: mTex.negate_b, None, 'negateB'],
                [5, lambda mTex: mTex.negate_a, None, 'negateA'],

            ])

            node['outputs'].append([0, 0, 0, 0])

        elif classID == getattr(rt, 'ai_sign', None):
            node['type'] = 'SIGN_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_sqrt', None):
            node['type'] = 'SQRT_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])


        elif classID == getattr(rt, 'ai_subtract', None):
            node['type'] = 'SUBTRACT_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input1), ('input1_shader', 'input1_connected', None), None],
                [1, lambda mTex: extractColor(mTex.input2), ('input2_shader', 'input2_connected', None), None]
            ])

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_trigo', None):
            node['type'] = 'TRIGO_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: extractColor(mTex.input), ('input_shader', 'input_connected', None), None],
                [1, lambda mTex: mTex.frequency, ('frequency_shader', 'frequency_connected', None), None],
                [2, lambda mTex: mTex.phase, ('phase_shader', 'phase_connected', None), None]
            ])

            node['function'] = mTex.function
            node['units'] = mTex.units

            node['outputs'].append([0, 0, 0])

        elif classID == getattr(rt, 'ai_vector_to_rgb', None):
            node['type'] = 'VECTOR_TO_RGB_AR'

            connTable = processNodeInputs(node, mTex, [
                [0, lambda mTex: [mTex.input.x, mTex.input.y, mTex.input.z], ('input_shader', 'input_connected', None), None],
            ])
            node['mode'] = mTex.mode

            node['outputs'].append([0, 0, 0])


        elif classID in CONVERTIBLE_NODE_CLASSES:
            index = getTextureIndex(gltf, getPtr(mTex))

            if index == -1:
                printLog('ERROR', 'Texture not found: ' + mTex.name)
                node['type'] = 'BITMAP_NONE_MX'
            else:
                node['type'] = 'BITMAP_MX'
                node['texture'] = index
                node['alphaAsMono'] = False
                node['alphaAsRGB'] = False
                node['alphaSource'] = 'FILE'

                # StandardUVGen
                if hasattr(mTex, 'coords') and hasattr(mTex.coords, 'U_Offset'):
                    coords = maxcpp.extractTexCoordsParams(getPtr(mTex))
                    node['uvIndex'] = mTex.coords.mapChannel-1
                    node['clampToEdgeNoExtend'] = [not mTex.coords.U_Tile, not mTex.coords.V_Tile]
                    node['mapping'] = coords['mapping']
                    node['axis'] = coords['axis']

                    coordsInputName = 'coords' if classID == rt.tiles else 'Coordinates'
                    connTable = processNodeInputs(node, mTex, [
                        [0, lambda bitMapTex: coords['U_Offset'], None, (coordsInputName, 'U_Offset')],
                        [1, lambda bitMapTex: coords['V_Offset'], None, (coordsInputName, 'V_Offset')],
                        [2, lambda bitMapTex: coords['U_Tiling'], None, (coordsInputName, 'U_Tiling')],
                        [3, lambda bitMapTex: coords['V_Tiling'], None, (coordsInputName, 'V_Tiling')],
                        [4, lambda bitMapTex: coords['W_angle'], None, (coordsInputName, 'W_Angle')],
                    ])
                # StandardXYZGen
                else:
                    node['uvIndex'] = 0
                    node['clampToEdgeNoExtend'] = [False, False]
                    node['mapping'] = 'EXPLICIT_MAP_CHANNEL'
                    node['axis'] = 'XY'
                    # offset
                    node['inputs'].append(0)
                    node['inputs'].append(0)
                    # tiling
                    node['inputs'].append(1)
                    node['inputs'].append(1)
                    # angle
                    node['inputs'].append(0)

            node['outputs'].append([0,0,0,0])

        else:
            printLog('WARNING', 'Unsupported map type: ' + str(rt.classOf(mTex)))

            node['type'] = 'MAP_MX'
            node['inputs'].append([0,0,0,0])
            node['outputs'].append([0,0,0,0])

    else:
        node['type'] = 'UNKNOWN'
        printLog('ERROR', 'Unknown node type: ' + node['name'])

    # NOTE: fixes issue with zero normals, since there is no input factors for Arnold nodes,
    if node['type'].endswith('_AR') and 'inputFactors' in node:
        del node['inputFactors']

    nodes.append(node)

    if stopTraverse:
        return

    # current node
    toNode = len(nodes)-1

    # process sub materials

    if mMat:
        numSM = rt.getNumSubMtls(mMat)

        for i in range(numSM):

            subMat = rt.getSubMtl(mMat, i+1)

            if subMat and connTable[i] > -1:
                edge = {
                    # this node will be added in processNode
                    'fromNode' : len(nodes),
                    'fromOutput' : 0,
                    'toNode' : toNode,
                    'toInput' : connTable[i]
                }

                edges.append(edge)

                processNode(subMat, gltf, nodes, edges)

    # process sub textures

    # do not go further than convertible maps
    if mTex and isConvertibleTex(mTex):
        return

    num = rt.getNumSubTexmaps(mtlBase)
    isGradRamp = rt.classOf(mtlBase) == rt.Gradient_Ramp

    for i in range(num):
        # some material nodes such as Blend have both material and map inputs
        inputOffset = numSM if mMat else 0
        toInput = connTable[i + inputOffset]

        if isGradRamp:
            # the last "Source Map" input is exported as the first and the only
            # one input
            if toInput == num - 1 and mtlBase.Source_Map_On and mtlBase.Gradient_Type == 5:
                toInput = 0
            else:
                # the other "Flag" inputs are not supported
                continue

        subTex = rt.getSubTexmap(mtlBase, i+1)

        if subTex and toInput > -1:

            existingNode = utils.getByNameID(nodes, getPtr(subTex))
            fromOutput = mtlBase.outputChannelIndex-1 if rt.classOf(mtlBase) == rt.MultiOutputChannelTexmapToTexmap else 0

            if existingNode:
                edge = {
                    'fromNode': nodes.index(existingNode),
                    'fromOutput': fromOutput,
                    'toNode': toNode,
                    'toInput': toInput
                }
                edges.append(edge)
            else:
                edge = {
                    # this node will be added in processNode
                    'fromNode': len(nodes),
                    'fromOutput': fromOutput,
                    'toNode': toNode,
                    'toInput': toInput
                }

                edges.append(edge)
                processNode(subTex, gltf, nodes, edges)

def processNodeOutput(node, outputBlock):

    output = maxcpp.extractTexOutParams(getPtr(outputBlock), CURVE_DATA_SIZE)

    if (output['invert'] or output['clamp'] or output['alphaFromRGB'] or
            output['outputAmount'] != 1.0 or output['rgbOffset'] != 0.0 or
            output['rgbLevel'] != 1.0 or output['bumpAmount'] != 1.0 or
            'colorMap' in output):
        node['output'] = output


def processNodeInputs(node, matOrMap, inputDefs, inFacMult=1.0):
    """
    inputDefs is a list of
        [connection number, input extactor function, input factor params, animation path tuple]
    connection number
        which input from the "real" node bind to exported input
    where input factor params is
        (map name param, map on/off param name, map amount param name) or None

    use listProps() to find out param names
    and listAnims() to find out anim paths
    """
    inputFactors = []
    animControls = []

    connTable = [-1] * CONN_TABLE_SIZE

    needInputFactors = False

    for i in range(len(inputDefs)):
        inputDef = inputDefs[i]
        if inputDef[2]:
            needInputFactors = True

    for i in range(len(inputDefs)):

        inputDef = inputDefs[i]

        connection = inputDef[0]
        extractInputFunc = inputDef[1]
        inputFactorParams = inputDef[2]
        anim = inputDef[3]

        if connection >= 0:
            connTable[connection] = i

        inputVal = extractInputFunc(matOrMap) if callable(extractInputFunc) else extractInputFunc
        node['inputs'].append(inputVal)

        if inputFactorParams:
            mapNameParam = inputFactorParams[0]
            mapOnParam = inputFactorParams[1]
            matAmtParam = inputFactorParams[2]

            if getattr(matOrMap, mapNameParam) and getattr(matOrMap, mapOnParam):
                if matAmtParam:
                    inputFactors.append(inFacMult * getattr(matOrMap, matAmtParam))
                else:
                    inputFactors.append(1.0)
            else:
                inputFactors.append(0)
                connTable[connection] = -1
        elif needInputFactors:
            inputFactors.append(0)
            connTable[connection] = -1

        if anim:
            control = extractAnimatableController(matOrMap, anim)
            if not control:
                continue

            # TODO: fix issues with different value scaling

            if rt.superClassOf(control) == rt.FloatController:
                type = 'VALUE_MX'
                value = control.value

                # HACK: temporary workaround for MixAmount value multiplied by 100
                if anim == 'MixAmount':
                    value /= 100

            elif rt.superClassOf(control) == rt.Point3Controller:
                type = 'RGB_MX'
                value = extractVecAsColor(control.value) + [1]
            elif rt.superClassOf(control) == rt.Point4Controller:
                type = 'RGB_MX'
                value = extractVec(control.value) + [1]
            else:
                continue

            animControls.append({
                'name': matOrMap.name + '_' + (anim if isinstance(anim, str) else anim[-1]),
                'type': type,
                'control': control,
                'value': value,
                'index': i
            })

    if len(inputFactors):
        node['inputFactors'] = inputFactors

    if len(animControls):
        node['tmpAnimControls'] = animControls;

    return connTable

def extractNodeGraph(mMat, gltf):

    nodes = []
    edges = []

    outNode = {
        'name' : 'MaxOutput',
        'type' : 'OUTPUT_MX',
        'inputs' : [[1, 1, 1, 1]],
        'outputs': [],
        'is_active_output': True
    }
    nodes.append(outNode)

    outEdgeIn0 = {
        'fromNode' : 1,
        'fromOutput' : 0,
        'toNode' : 0,
        'toInput' : 0
    }
    edges.append(outEdgeIn0)

    outEdgeIn1 = {
        'fromNode' : 1,
        'fromOutput' : 1,
        'toNode' : 0,
        'toInput' : 1
    }
    edges.append(outEdgeIn1)

    processNode(mMat, gltf, nodes, edges)

    additionalNodes = []
    additionalEdges = []

    for i in range(len(nodes)):
        node = nodes[i]

        if 'tmpAnimControls' in node:
            for animControl in node['tmpAnimControls']:

                toNode = i
                toInput = animControl['index']

                # do not create/connect controller node if input is already connected (e.g. some Map)
                inputOccupied = False

                for edge in edges:
                    if edge['toNode'] == toNode and edge['toInput'] == toInput:
                        inputOccupied = True

                if inputOccupied:
                    continue

                # deattenuate input factor, it is required since factor of
                # non-connected input is set to 0
                if 'inputFactors' in node:
                    node['inputFactors'][toInput] = 1.0

                additionalNodes.append({
                    'name' : animControl['name'],
                    'type' : animControl['type'],
                    'inputs': [],
                    'outputs': [animControl['value']],
                    'tmpAnimControl': animControl['control']
                })

                additionalEdges.append({
                    'fromNode' : len(nodes) + len(additionalNodes) - 1,
                    'fromOutput' : 0,
                    'toNode' : toNode,
                    'toInput' : toInput
                })

            del node['tmpAnimControls']

    return { 'nodes' : nodes + additionalNodes, 'edges' : edges + additionalEdges }

def traverseMtlBase(matOrTex, callback):

    if rt.superClassOf(matOrTex) == rt.material:
        for i in range(1, rt.getNumSubMtls(matOrTex) + 1):
            subMat = rt.getSubMtl(matOrTex, i)
            if not subMat:
                continue

            # Don't consider "Original Material" socket in shell materials,
            # because only the "Baked Material" input is exported.
            if rt.classOf(matOrTex) == rt.Shell_Material and i == 1:
                continue

            callback(subMat)
            traverseMtlBase(subMat, callback)

    for i in range(1, rt.getNumSubTexmaps(matOrTex) + 1):
        subTex = rt.getSubTexmap(matOrTex, i)
        if subTex:
            callback(subTex)

            # do not go further than convertible maps
            if not isConvertibleTex(subTex):
                traverseMtlBase(subTex, callback)


def extractBitmapTextures(matOrTex):
    """
    Extract BitmapTex and convertible maps from the material
    """

    textures = []

    def addTexMap(matOrTex):
        if rt.superClassOf(matOrTex) == rt.textureMap:
            # NOTE: textures are in non-local scope
            if (isBitmapTex(matOrTex) or isOSLBitmapTex(matOrTex) or isImageTex(matOrTex)) and (matOrTex not in textures):
                textures.append(matOrTex)

            if isConvertibleTex(matOrTex) and (matOrTex not in textures):
                textures.append(matOrTex)

    addTexMap(matOrTex)
    traverseMtlBase(matOrTex, addTexMap)

    return textures

def isConvertibleTex(mTex):

    classID = rt.classOf(mTex)
    if classID in CONVERTIBLE_NODE_CLASSES:
        if classID == rt.Gradient_Ramp:
            return gradRampNeedsConversion(mTex)
        else:
            return True

    return False

def gradRampNeedsConversion(gradRampTex):
    return (hasattr(gradRampTex, 'V3DGradientRampData') and
            extractCustomProp(gradRampTex, 'V3DGradientRampData', 'bakeToBitmap'))

def extractEnvMap():
    # Use Map checker also affects result
    texMap = rt.environmentMap
    if texMap and rt.useEnvironmentMap:
        return texMap

    return None

def extractEnvColor():
    return extractColor(rt.backgroundColor)

def extractAmbColor():
    if rt.classOf(rt.renderers.current) == rt.ART_Renderer:
        if not extractEnvMap():
            return extractEnvColor()
        else:
            return [0, 0, 0]
    else:
        return extractColor(rt.ambientColor)

def extractTexFileName(mTex):

    if isBitmapTex(mTex):
        bitmap = mTex.bitmap if hasattr(mTex, 'bitmap') else None
        if bitmap:
            return rt.mapPaths.getFullFilePath(bitmap.filename)
        elif isORMMap(mTex):
            return mTex.name
        elif isBaseAlphaMap(mTex):
            return mTex.name
        else:
            return ''
    elif isOSLBitmapTex(mTex):
        if mTex.OSLShaderName == 'HDRIenv':
            filename = mTex.HDRI
        elif mTex.OSLShaderName == 'HDRILights':
            filename = mTex.LightName1
        else:
            filename = mTex.filename

        return rt.mapPaths.getFullFilePath(filename)
    elif isImageTex(mTex):
        return rt.mapPaths.getFullFilePath(mTex.filename)

    elif isConvertibleTex(mTex):
        return CONVERTED_MAP_NAME
    else:
        return ''

def isCompatibleTex(mTex):
    if mTex and extractTexFileName(mTex):
        return True
    else:
        return False

def texNeedsConversion(mTex):
    path = extractTexFileName(mTex)
    if not isCompatibleImagePath(path) or path == CONVERTED_MAP_NAME:
        return True
    else:
        return False


def isORMMap(mTex):
    if '__ORM__' in mTex.name:
        return True
    else:
        return False

def isBaseAlphaMap(mTex):
    if '__BASE_ALPHA__' in mTex.name:
        return True
    else:
        return False

def extractTexCoordIndex(mTex):
    if isBitmapTex(mTex) or isImageTex(mTex):
        return mTex.coords.mapChannel - 1
    else:
        return 0

def isBitmapTex(mTex):
    if rt.classOf(mTex) == rt.Bitmaptexture:
        return True
    else:
        return False

def isImageTex(mTex):
    if (rt.classOf(mTex)) == getattr(rt, 'ai_image', None):
        return True
    else:
        return False

def isOSLBitmapTex(mTex):
    if (rt.classOf(mTex) == getattr(rt, 'OSLMap', None) and
            mTex.OSLShaderName in ['CameraProjector', 'HDRIenv', 'HDRILights', 'ObjectProjector', 'OSLBitmap', 'OSLBitmap2', 'SphericalProjector', 'UberBitmap', 'UberBitmap2']):
        return True
    else:
        return False

def preprocessOSL(code):
    out = io.StringIO()

    p = pcpp.Preprocessor()
    p.line_directive = None
    p.parse(code)
    p.write(out)

    return out.getvalue()

def parseOSLInOuts(ast, shaderName):

    inputs, outputs = ast.get_shader_params()

    def typeToVal(type):
        if type in ['color', 'point', 'vector']:
            return [0, 0, 0]
        else:
            return 0

    def typeToGLSLType(type):
        if type in ['color', 'point', 'vector']:
            return 'vec3'
        elif type in ['int', 'string']:
            return 'int'
        else:
            return 'float'

    def getInitCode(ast, n):
        if ast is None:
            return None
        return genOSLCode(ast, shaderName + '_init_' + str(n))

    def getInitGlobVars(ast):
        if ast is None:
            return None
        return [varName for _, varName in pyosl.glslgen.find_global_variables(ast)]

    inputs = [(typeToGLSLType(i[0]), i[1], typeToVal(i[0]), getInitCode(i[2], inputs.index(i)), getInitGlobVars(i[2])) for i in inputs]
    outputs = [(typeToGLSLType(o[0]), o[1], typeToVal(o[0])) for o in outputs]

    return inputs, outputs

def genOSLCode(ast, shaderName):
    ast = pyosl.glslgen.osl_to_glsl(ast)
    pyosl.glslgen.rename_shader(ast, shaderName)

    if shaderName in ['node_osl_noise', 'node_osl_noise3d']:
        replaceInputByDefineGLSL1(ast, 'Octaves', 4)

    code = pyosl.glslgen.generate(ast)
    return code

def replaceInputByDefineGLSL1(ast, name, value):

    begin = '''
#if __VERSION__ == 100
#define {0} {1}
#endif
'''.format(name, value)

    pyosl.glslgen.insert_raw_code(ast, begin, 'shader-begin')

    end = '''
#if __VERSION__ == 100
#undef {0}
#endif
'''.format(name, value)

    pyosl.glslgen.insert_raw_code(ast, end, 'shader-end')


def extractMaterials(meshNode):

    mMat = meshNode.material

    # None - generate default material using object color
    if not mMat:
        return [None]

    materials = []

    numSM = rt.getNumSubMtls(mMat)
    if numSM == 0 or not maxUtils.isMultiMaterial(mMat):
        materials.append(mMat)
    else:
        for i in range(1, numSM+1):
            subMat = rt.getSubMtl(mMat, i)
            if subMat:
                materials.append(subMat)

    return materials

def extractAlphaMode(mMat):

    mode = (extractCustomProp(mMat, 'V3DMaterialData', 'alphaMode').upper()
            if hasattr(mMat, 'V3DMaterialData') else 'AUTO')
    mode = 'BLEND' if (mode == 'ADD' or mode == 'COVERAGE') else mode

    if mode != 'AUTO':
        return mode

    if maxUtils.isPhysicalMaterial(mMat):
        hasTransWeight = mMat.Transparency > 0
        hasTransWeightMap = (mMat.transparency_map_on and bool(mMat.transparency_map))
        hasTransColorMap = (mMat.trans_color_map_on and bool(mMat.trans_color_map))
        hasTransRoughMap = (mMat.trans_rough_map_on and bool(mMat.trans_rough_map))
        hasCutoutMap = (mMat.cutout_map_on and bool(mMat.cutout_map)) # opacity

        # NOTE: needs to be rewritten properly
        return ('BLEND' if hasTransWeight or hasTransWeightMap
                or hasTransColorMap or hasTransRoughMap or hasCutoutMap else 'OPAQUE')

    elif maxUtils.isGLTFMaterial(mMat):
        alpha = getattr(mMat, 'alphaMode')
        if alpha == 1:
            return 'OPAQUE'
        elif alpha == 2:
            return 'MASK'
        elif alpha == 3:
            return 'BLEND'

    elif maxUtils.isUsdPreviewSurfaceMaterial(mMat):
        hasAlpha = mMat.opacity < 1.0
        hasOpacityMap = bool(mMat.opacity_map)
        if hasAlpha or hasOpacityMap:
            if getattr(mMat, 'opacityThreshold') > 0:
                return 'MASK'
            else:
                return 'BLEND'
        return 'OPAQUE'

    elif maxUtils.isStandardMaterial(mMat):
        hasAlpha = mMat.opacity < 100.0
        hasOpacityMap = (mMat.opacityMapEnable and mMat.opacityMapAmount > 0 and bool(mMat.opacityMap))

        return 'BLEND' if hasAlpha or hasOpacityMap else 'OPAQUE'

    elif maxUtils.isVrayMaterial(mMat):
        return 'BLEND' if extractVrayOpacity(mMat) < 1 else 'OPAQUE'

    return 'OPAQUE'

def extractVrayOpacity(mMat):
    opacity = clamp(1 - extractColorAverage(mMat.refraction), 0.1, 1)
    return opacity

def extractByName(arr, name):
    for i in arr:
        if i.name == name:
            return i

    return None

def extractImageBindataAsIs(path):
    with open(path, 'rb') as f:
        return f.read()

def extractImageBindataPNG(mTex):
    tmpImg = extractFPImagePNG(mTex, 512, 512)

    bindata = tmpImg.read()

    tmpImg.close()
    os.unlink(tmpImg.name)

    return bindata

def extractFPImagePNG(mTex, bmWidth, bmHeight):
    """
    extract image (file pointer) from the texture
    """

    tmpImg = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    bitMap = rt.renderMap(mTex, size=rt.point2(bmWidth, bmHeight), filename=tmpImg.name, filter=True)
    rt.save(bitMap)
    rt.close(bitMap)
    return tmpImg

def extractPBRBitmatTextures(mMat, tmpDir):
    textures = []

    texMaps = []

    if maxUtils.isGLTFMaterial(mMat):
        baseColorMap = getattr(mMat, 'baseColorMap')
        texMaps.append(baseColorMap)

        emissiveMap = getattr(mMat, 'emissionMap')
        texMaps.append(emissiveMap)

        normalMap = getattr(mMat, 'normalMap')
        texMaps.append(normalMap)

        occlusionMap = getattr(mMat, 'ambientOcclusionMap')
        roughnessMap = getattr(mMat, 'roughnessMap')
        metalnessMap = getattr(mMat, 'metalnessMap')

        if extractAlphaMode(mMat) != 'OPAQUE':
            alphaMap = getattr(mMat, 'AlphaMap')
            baseColor = getattr(mMat, 'baseColor')
            baseColor = [baseColor.r, baseColor.g, baseColor.b]
            if alphaMap:
                baseAlphaTexName = utils.baseAlphaTexName(mMat.name)
                baseAlphaTex = createBaseAlphaTex(baseAlphaTexName, alphaMap, tmpDir, baseColor, baseColorMap)
                if baseAlphaTex:
                    textures.append(baseAlphaTex)
                    texMaps.remove(baseColorMap)

    elif maxUtils.isUsdPreviewSurfaceMaterial(mMat):
        baseColorMap = getattr(mMat, 'diffuse_color_map')
        texMaps.append(baseColorMap)

        emissiveMap = getattr(mMat, 'emissive_color_map')
        texMaps.append(emissiveMap)

        normalMap = getattr(mMat, 'normal_map')
        texMaps.append(normalMap)

        occlusionMap = getattr(mMat, 'occlusion_map')
        roughnessMap = getattr(mMat, 'roughness_map')
        metalnessMap = getattr(mMat, 'metallic_map')

    else:
        baseColorMap = getattr(mMat, 'base_color_map')
        texMaps.append(baseColorMap)

        emissiveMap = getattr(mMat, 'emit_color_map')
        texMaps.append(emissiveMap)

        bumpMap = getattr(mMat, 'bump_map')
        if bumpMap and rt.classOf(bumpMap) == rt.Normal_Bump and getattr(bumpMap, 'normal_map'):
            normalMap = getattr(bumpMap, 'normal_map')
        else:
            normalMap = None
        texMaps.append(normalMap)

        occlusionMap = getattr(mMat, 'base_weight_map')
        roughnessMap = getattr(mMat, 'roughness_map')
        metalnessMap = getattr(mMat, 'metalness_map')

    ormTex = None

    if roughnessMap or metalnessMap:
        ormTex = createORMTex(utils.ormTexName(mMat.name), occlusionMap, roughnessMap, metalnessMap, tmpDir)
        if ormTex:
            textures.append(ormTex)

    if not ormTex:
        texMaps.append(occlusionMap)

    for texMap in texMaps:
        if not texMap:
            continue

        bitMap = texMap if isBitmapTex(texMap) else None
        if bitMap and (bitMap not in textures):
            textures.append(bitMap)

        if isConvertibleTex(texMap) and (texMap not in textures):
            textures.append(texMap)

    return textures


def createORMTex(name, occlusionMap, roughnessMap, metalnessMap, tmpDir):
    """
    Create occlusion/roughness/metalness BitmapTex
    """

    occlusionMapPath = ''
    occlusionFP = None
    if isCompatibleTex(occlusionMap):
        if texNeedsConversion(occlusionMap):
            occlusionFP = extractFPImagePNG(occlusionMap, width, height)
            occlusionMapPath = occlusionFP.name
        else:
            occlusionMapPath = extractTexFileName(occlusionMap)

    roughnessMapPath = ''
    roughnessFP = None
    if isCompatibleTex(roughnessMap):
        if texNeedsConversion(roughnessMap):
            roughnessFP = extractFPImagePNG(roughnessMap, width, height)
            roughnessMapPath = roughnessFP.name
        else:
            roughnessMapPath = extractTexFileName(roughnessMap)

    metalnessMapPath = ''
    metalnessFP = None
    if isCompatibleTex(metalnessMap):
        if texNeedsConversion(metalnessMap):
            metalnessFP = extractFPImagePNG(metalnessMap, width, height)
            metalnessMapPath = metalnessFP.name
        else:
            metalnessMapPath = extractTexFileName(metalnessMap)

    # missing textures
    if not (metalnessMapPath or roughnessMapPath):
        return None

    tmpImgHex = '{:x}'.format(abs(hash(occlusionMapPath + roughnessMapPath + metalnessMapPath)))

    tmpImg = join(tmpDir, 'occlusion_roughness_metallic_' + tmpImgHex + '.png')

    if not os.path.exists(tmpImg):
        printLog('INFO', 'Generating PBR texture: ' + os.path.basename(tmpImg))

        painter = QPainter() # canvas
        width = 2
        height = 2
        pbrMaps = {'r': occlusionMapPath, 'g': roughnessMapPath, 'b': metalnessMapPath}

        for channel in pbrMaps:
            pbrMap = pbrMaps[channel]
            if pbrMap:
                texTmp = QImage(pbrMap)
                texTmp = texTmp.convertToFormat(QImage.Format_RGB32)

                width = height = max(width, int(math.pow(2, math.ceil(math.log(max(texTmp.width(), texTmp.height()), 2)))))
                painter.begin(texTmp)
                painter.setCompositionMode(QPainter.CompositionMode_Multiply)
                painter.fillRect(0, 0, texTmp.width(), texTmp.height(), QColor(255 * int(channel == 'r'),
                                                                               255 * int(channel == 'g'),
                                                                               255 * int(channel == 'b'),
                                                                               255))
                painter.end()

                pbrMaps[channel] = texTmp
            else:
                pbrMaps[channel] = None

        ormTex = QImage(width, height, QImage.Format_RGB32)

        painter.begin(ormTex)
        painter.setCompositionMode(QPainter.CompositionMode_Plus)
        painter.fillRect(0, 0, width, height, QColor(0, 0, 0, 255))
        for channel in pbrMaps:
            pbrMap = pbrMaps[channel]
            if pbrMap:
                pbrMap = pbrMap.scaled(width, height, QtCore.Qt.IgnoreAspectRatio)
                painter.drawImage(0, 0, pbrMap)
            else:
                painter.fillRect(0, 0, width, height, QColor(255 * int(channel == 'r'),
                                                             255 * int(channel == 'g'),
                                                             255 * int(channel == 'b'),
                                                             255))
        painter.end()

        ormTex.save(tmpImg)

    ormBitmapTex = rt.Bitmaptexture()
    ormBitmapTex.name = name
    ormBitmapTex.filename = tmpImg

    # explicitly set linear color space
    rt.save(ormBitmapTex.bitmap, gamma=1)
    ormBitmapTex.reload()

    # cleanup

    if occlusionFP:
        occlusionFP.close()
        os.unlink(occlusionFP.name)
    if roughnessFP:
        roughnessFP.close()
        os.unlink(roughnessFP.name)
    if metalnessFP:
        metalnessFP.close()
        os.unlink(metalnessFP.name)

    return ormBitmapTex

def createBaseAlphaTex(name, alphaMap, tmpDir, baseColor = [1, 1, 1], baseColorMap = None):
    """
    Create/merge base and alpha BitmapTex
    """

    alphaMapPath = ''
    alphaFP = None
    if isCompatibleTex(alphaMap):
        if texNeedsConversion(alphaMap):
            alphaFP = extractFPImagePNG(alphaMap, width, height)
            alphaMapPath = alphaFP.name
        else:
            alphaMapPath = extractTexFileName(alphaMap)

    baseColorMapPath = ''
    baseFP = None
    if isCompatibleTex(baseColorMap):
        if texNeedsConversion(baseColorMap):
            baseFP = extractFPImagePNG(baseColorMap, width, height)
            baseColorMapPath = baseFP.name
        else:
            baseColorMapPath = extractTexFileName(baseColorMap)

    # missing textures
    if not alphaMapPath or (baseColorMap and (not baseColorMapPath)):
        return None

    tmpImgHex = '{:x}'.format(abs(hash(alphaMapPath)))

    tmpImg = join(tmpDir, 'base_alpha_' + tmpImgHex + '.png')

    if not os.path.exists(tmpImg):
        printLog('INFO', 'Generating base-alpha texture: ' + os.path.basename(tmpImg))

        width = 2
        height = 2
        colorPixel = QColor()

        alphaTex = QImage(alphaMapPath)
        alphaTex = alphaTex.convertToFormat(QImage.Format_ARGB32)

        if baseColorMapPath:
            baseTex = QImage(baseColorMapPath)
            baseTex = baseTex.convertToFormat(QImage.Format_RGB32)

            height = max([baseTex.width(), baseTex.height(), alphaTex.width(), alphaTex.height()])
            width = height = max(width, int(math.pow(2, math.ceil(math.log(height, 2)))))

            baseTex = baseTex.scaled(width, height, QtCore.Qt.IgnoreAspectRatio)
            alphaTex = alphaTex.scaled(width, height, QtCore.Qt.IgnoreAspectRatio)

            baseUCharPtr = baseTex.bits()
            alphaUCharPtr = alphaTex.bits()

            i = 0
            for y in range(height):
                for x in range(width):
                    dCol = struct.unpack('I', baseUCharPtr[i:i+4])[0]
                    aCol = struct.unpack('I', alphaUCharPtr[i:i+4])[0]
                    alpha = (qRed(aCol) + qGreen(aCol) + qBlue(aCol)) / 3
                    colorPixel.setRgb(qRed(dCol), qGreen(dCol), qBlue(dCol), alpha)
                    alphaUCharPtr[i:i+4] = struct.pack('I', colorPixel.rgba())
                    i+=4

        else:
            height = max(alphaTex.width(), alphaTex.height())
            width = height = max(width, int(math.pow(2, math.ceil(math.log(height, 2)))))
            alphaTex = alphaTex.scaled(width, height, QtCore.Qt.IgnoreAspectRatio)

            alphaUCharPtr = alphaTex.bits()

            i = 0
            for y in range(height):
                for x in range(width):
                    aCol = struct.unpack('I', alphaUCharPtr[i:i+4])[0]
                    alpha = (qRed(aCol) + qGreen(aCol) + qBlue(aCol)) / 3
                    colorPixel.setRgb(baseColor[0], baseColor[1], baseColor[2], alpha)
                    alphaUCharPtr[i:i+4] = struct.pack('I', colorPixel.rgba())
                    i+=4

        alphaTex.save(tmpImg)

    baseAlphaBitmapTex = rt.Bitmaptexture()
    baseAlphaBitmapTex.name = name
    baseAlphaBitmapTex.filename = tmpImg

    # cleanup

    if baseFP:
        baseFP.close()
        os.unlink(baseFP.name)
    if alphaFP:
        alphaFP.close()
        os.unlink(alphaFP.name)

    return baseAlphaBitmapTex

def extractImageExportedURI(exportSettings, bitMapTex):

    texPath = extractTexFileName(bitMapTex)

    uriName, uriExt = os.path.splitext(os.path.basename(texPath))

    if maxUtils.imgNeedsCompression(bitMapTex):
        if uriExt == '.hdr':
            uriExt = '.hdr.xz'
        else:
            uriExt = '.ktx2'

    elif not isCompatibleImagePath(texPath):
        uriExt = '.png'

    uriCache = exportSettings['uri_cache']

    uniqueURI = uriName + uriExt

    # NOTE: allow non-unique URLs

    if texPath != CONVERTED_MAP_NAME:
        return uniqueURI

    i = 0

    while uniqueURI in uriCache['uri']:

        index = uriCache['uri'].index(uniqueURI)
        if uriCache['obj'][index] == bitMapTex:
            break

        i += 1
        uniqueURI = uriName + '_' + utils.integerToMaxSuffix(i) + uriExt

    return uniqueURI

def extractActiveViewport():
    """Extract viewport index which fit the required type, taking active viewport as priority"""

    activeViewIdx = rt.viewport.activeViewport

    viewports = [activeViewIdx]

    for i in range(1, rt.viewport.numViews+1):
        viewports.append(i)

    for viewIdx in viewports:
        viewType = rt.viewport.getType(index=viewIdx)

        if viewType in [rt.Name('view_iso_user'), rt.Name('view_persp_user'), rt.Name('view_camera')]:
            return viewIdx

    return activeViewIdx

def extractGroupNames(mNode):
    names = []
    parent = mNode.parent
    if parent and parent != rt.rootNode:
        if rt.isGroupHead(parent):
            names.append(parent.name)
        names = names + extractGroupNames(parent)

    return names

def extractSelectionSetNames(mNode):
    return [selSet.name for selSet in rt.selectionSets if mNode in selSet]

def isPbrMaterial(mMat, forceGltfCompat=False):
    """Check if the material can be exported using glTF-based PBR format"""

    if maxUtils.isUsdPreviewSurfaceMaterial(mMat) or maxUtils.isGLTFMaterial(mMat):
        return True

    if not maxUtils.isPhysicalMaterial(mMat):
        return False

    if (forceGltfCompat or hasattr(mMat, 'V3DMaterialData') and
            extractCustomProp(mMat, 'V3DMaterialData', 'gltfCompat')):
        return True
    else:
        return False

def extractConstraints(gltf, mNode):

    constraints = maxcpp.extractConstraints(getPtr(mNode))

    constraintsNeedFixCameraAndLightRotation = [
        'copyRotation', 'copyTransforms', 'dampedTrack', 'lockedTrack',
        'trackTo', 'childOf', 'motionPath'
    ]

    # replace target pointers by indices
    for c in constraints:
        if 'target' in c:
            c['target'] = getNodeIndex(gltf, c['target'])

            if c['type'] in constraintsNeedFixCameraAndLightRotation:
                c['fixCameraLightRotation'] = True

    baseObj = mNode.baseObject

    if maxUtils.mNodeHasFixOrthoZoom(mNode):
        constraints.append({
            'name': 'Fix Ortho Zoom',
            'mute': False,
            'type': 'fixOrthoZoom',
            'target': getNodeIndex(gltf, getPtr(mNode.parent))
        })

    if maxUtils.mNodeHasCanvasFitParams(mNode):
        constraints.append({
            'name': 'Canvas Fit',
            'mute': False,
            'type': 'canvasFit',
            'target': getNodeIndex(gltf, getPtr(mNode.parent)),
            'edgeH': extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasFitX').upper(),
            'edgeV': extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasFitY').upper(),
            'fitShape': extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasFitShape').upper(),
            'offset': extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasFitOffset')
        })

    if extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasBreakEnabled'):
        constraints.append({
            'name': 'Canvas Visibility Breakpoints',
            'mute': False,
            'type': 'canvasBreakpoints',
            'minWidth': extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasBreakMinWidth'),
            'maxWidth': extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasBreakMaxWidth'),
            'minHeight': extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasBreakMinHeight'),
            'maxHeight': extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasBreakMaxHeight'),
            'orientation': extractCustomProp(baseObj, 'V3DAdvRenderData', 'canvasBreakOrientation').upper()
        })

    return constraints

def extractTriMeshFromNode(node):
    triMesh = getattr(node, 'mesh', None)
    if not triMesh:
        printLog('INFO', 'Converting to mesh: ' + node.name)

        triMesh = rt.snapshotAsMesh(node)

    return triMesh

def extractAnimatableController(obj, nameOrNames):

    subAnim = obj

    # handle string values
    if isinstance(nameOrNames, str):
        nameOrNames = (nameOrNames, )

    for name in nameOrNames:
        for i in range(1, subAnim.numSubs+1):
            # do case-insensitive comparison to fix naming issues in different max versions
            if str(rt.getSubAnim(subAnim, i)).replace('SubAnim:', '').lower() == name.lower():
                subAnim = rt.getSubAnim(subAnim, i)
                break

    return subAnim.controller

def extractWorldStateObj(node):
    """Get object from Max node (INode)"""
    return node.EvalWorldState().Getobj()

def extractGradientRampData(rampMap):

    getRef = rt.refs.getReference

    gradRef = getRef(getRef(rampMap, 4), 1)

    interpRef = getRef(getRef(gradRef, 4), 1)
    interpMode = interpRef.value

    keyRefs = getRef(gradRef, 2)
    keyCount = rt.refs.getNumRefs(keyRefs)

    def getKeyColor(i):
        keyRef = getRef(keyRefs, i+1)
        colRef = getRef(getRef(keyRef, 1), 1)
        return colRef.value

    def getKeyPos(i):
        keyRef = getRef(keyRefs, i+1)
        posRef = getRef(getRef(keyRef, 2), 1)
        return posRef.value

    def getKeyInterp(i):
        keyRef = getRef(keyRefs, i+1)
        interpRef = getRef(getRef(keyRef, 5), 1)
        return interpRef.value

    keyIndices = list(range(keyCount))
    keyIndices.sort(key=lambda i: getKeyPos(i))
    keyPositions = [getKeyPos(i) for i in keyIndices]
    keyColors = [getKeyColor(i) for i in keyIndices]

    if interpMode == GRADRAMP_INTERP_CUSTOM:
        # custom interpolation modes are started with 0, which we reserve as
        # indication of the custom mode itself
        keyInterpolations = [getKeyInterp(i) + 1 for i in keyIndices]
    else:
        keyInterpolations = [interpMode for i in keyIndices]

    rampData = []

    posFromIdx = 0
    posFrom = keyPositions[0]
    posTo = keyPositions[1]
    for i in range(RAMP_DATA_SIZE):
        samplePos = i / (RAMP_DATA_SIZE - 1)

        while samplePos > posTo:
            posFromIdx += 1
            posFrom = posTo
            posTo = keyPositions[posFromIdx + 1]

        t = (samplePos - posFrom) / (posTo - posFrom)
        colFrom = keyColors[posFromIdx]
        colTo = keyColors[posFromIdx + 1]
        keyInterp = keyInterpolations[posFromIdx]

        # easing curve empirical approximations
        if keyInterp == GRADRAMP_INTERP_EASE_IN:
            coeff = math.pow(t, 1.57)
        elif keyInterp == GRADRAMP_INTERP_EASE_IN_OUT:
            coeff = math.pow(t, 3) - 1.44 * math.pow(t, 2) + 1.44 * t
        elif keyInterp == GRADRAMP_INTERP_EASE_OUT:
            coeff = math.pow(t, 0.65)
        elif keyInterp == GRADRAMP_INTERP_LINEAR:
            coeff = t
        elif keyInterp == GRADRAMP_INTERP_SOLID:
            coeff = 0
        else:
            coeff = 0

        r = colFrom.X * (1 - coeff) + colTo.X * coeff
        g = colFrom.Y * (1 - coeff) + colTo.Y * coeff
        b = colFrom.Z * (1 - coeff) + colTo.Z * coeff

        rampData.append(r)
        rampData.append(g)
        rampData.append(b)

    return rampData

def extractCustomProps(mNode):
    """Extract and parse user-defined options"""

    buf = rt.getUserPropBuffer(mNode)

    props = {}
    hasProps = False
    for propStr in str(buf).splitlines():
        propStr = propStr.strip()
        if propStr:
            propArr = [i.strip().strip('"') for i in propStr.split('=')]
            props[propArr[0]] = propArr[1] if len(propArr) > 1 else ''
            hasProps = True

    if hasProps:
        return props
    else:
        return None
