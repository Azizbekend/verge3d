/*
 * Verge3D macros and functions
 * History:
 * 2018-27-04: Moved all functions into methods of a single struct. Namespaced all calls to Verge3D.-Shawn Olson
 */

include "reflectionCubemap.ms"
include "reflectionPlane.ms"
include "clippingPlane.ms"

macroScript exportGLTF
category: "Verge3D"
tooltip: "Export glTF..."
(
    ::v3dManager.exportGLTF()
)

macroScript reexportAll
category: "Verge3D"
tooltip: "Reexport All..."
(
    ::v3dManager.reexportAll()
)


macroScript sneakPeek
category: "Verge3D"
tooltip: "Sneak Peek..."
(
    ::v3dManager.sneakPeek()
)

macroScript runAppManager
category: "Verge3D"
tooltip: "Run App Manager..."
(
    ::v3dManager.runAppManager()
)

macroScript showExportSettings
category: "Verge3D"
tooltip: "Export Settings..."
(
    ::v3dManager.showExportSettings()
)

macroScript autoAssignAttrs
category: "Verge3D"
tooltip: "Auto-Assign Verge3D Params"
(
    on isChecked do (
        ::v3dManager != undefined and ::v3dManager.autoAssignAttrs == true
    )
    on execute do (
        if ::v3dManager  != undefined then (
            if ::v3dManager.autoAssignAttrs == true then (
                ::v3dManager.turnOff()
            ) else (
                ::v3dManager.turnOn()
            )
        )
    )
)

macroScript runUserManual
category: "Verge3D"
tooltip: "User Manual"
(
    ::v3dManager.runUserManual()
)

IBL_ENV_MODE_TYPES = #("PMREM", "PROBE_CUBEMAP", "PROBE", "NONE")
IBL_ENV_MODE_DESC = #("PMREM (slow)", "Light Probe + Cubemap (deprecated)",
        "Light Probe (fast)", "None (fastest)")

SHADOW_FILTERING_TYPES = #("BASIC", "BILINEAR", "PCF", "PCFSOFT", "PCFPOISSON",
        "ESM")
SHADOW_FILTERING_DESC = #("Basic", "Bilinear", "PCF", "PCF (Bilinear)",
        "PCF (Poisson Disk)", "ESM")

::v3dExportSettingsDataCA = attributes V3DExportSettingsData attribID: #(0x688ae226, 0x4b807b38)
(
    fn isESMEnabled shadowFilteringTypeUI = (
        return SHADOW_FILTERING_TYPES[shadowFilteringTypeUI.selection] == "ESM"
    )

    parameters main rollout:params
    (
        copyright type:#string ui:copyrightUI default:""
        bakeText type:#boolean ui:bakeTextUI default:false
        lzmaEnabled type:#boolean ui:lzmaEnabledUI default:false
        aaMethod type:#string default:"Auto"
        useHDR type:#boolean ui:useHdrUI default:false
        useOIT type:#boolean ui:useOitUI default:false
        pmremMaxTileSize type:#string ui:pmremMaxTileSizeUI default:"256"
        iblEnvironmentMode type:#string ui:iblEnvironmentModeUI default:IBL_ENV_MODE_TYPES[1]

        shadowFilteringType type:#string ui:shadowFilteringTypeUI \
                default:"PCFPOISSON"
        esmDistanceScale type:#float ui:esmDistanceScaleUI default:0.3

        outlineEnabled type:#boolean ui:outlineEnabledUI default:false
        edgeStrength type:#float ui:edgeStrengthUI default:3
        edgeGlow type:#float ui:edgeGlowUI default:0
        edgeThickness type:#float ui:edgeThicknessUI default:1
        pulsePeriod type:#float ui:pulsePeriodUI default:0
        visibleEdgeColor type:#color ui:visibleEdgeColorUI default:[255,255,255]
        hiddenEdgeColor type:#color ui:hiddenEdgeColorUI default:[25,25,25]
        renderHiddenEdge type:#boolean ui:renderHiddenEdgeUI default:true

        animExport type:#boolean ui:animExportUI default:true
        animUsePlaybackRange type:#boolean ui:animUsePlaybackRangeUI default:false
        animStartWithZero type:#boolean ui:animStartWithZeroUI default:true

        optimizeAttrs type:#boolean ui:optimizeAttrsUI default:true

        aoEnabled type:#boolean ui:aoEnabledUI default:false
        aoDistance type:#worldUnits ui:aoDistanceUI default:2
        aoFactor type:#float ui:aoFactorUI default:1
        aoTracePrecision type:#float ui:aoTracePrecisionUI default:0.25
        aoBentNormals type:#boolean ui:aoBentNormalsUI default:false

        /*
         * The scene, which was created with Verge3d 3.9.0, and then
         * opened and saved again without Verge3d installed, causes
         * "Error Loading ParamBlock2" error. So register it last to avoid it.
        */
        compressTextures type:#boolean ui:compressTexturesUI default:false
    )


    rollout params "Verge3D Export Settings" width:270
    (

        dotNetControl tb "TabControl" width:270 height:24 pos:[0,0]

        group "Common Settings" (
            edittext copyrightUI "Scene Copyright:"
            checkbox bakeTextUI "Bake Text"
            checkbox lzmaEnabledUI "Enable LZMA Compression"
            checkbox compressTexturesUI "Compress Textures"
            dropdownlist aaMethodUI "Anti-Aliasing" items:#("Auto", "MSAA 4x", "MSAA 8x", "MSAA 16x", "FXAA", "None")
            on aaMethodUI selected i do aaMethod = aaMethodUI.items[i]
            checkbox useHdrUI "Use HDR Rendering"
            checkbox useOitUI "Order-Independent Transparency"
            dropdownlist pmremMaxTileSizeUI "Environment Map Size" items:#("256", "512", "1024")
            on pmremMaxTileSizeUI selected i do pmremMaxTileSize = pmremMaxTileSizeUI.items[i]

            dropdownlist iblEnvironmentModeUI "IBL Environment Mode" \
                    items:IBL_ENV_MODE_DESC \
                    tooltip:"Preferred method of rendering the scene environment."
            on iblEnvironmentModeUI selected i do iblEnvironmentMode = IBL_ENV_MODE_TYPES[i]

            checkbox optimizeAttrsUI "Optimize Mesh Attributes"
        )

        label lbl1 offset:[0,-320] visible:on

        group "Shadows" (
            dropdownlist shadowFilteringTypeUI "Shadow Filtering" \
                    items:SHADOW_FILTERING_DESC tooltip:"Shadow Filtering Mode."
            spinner esmDistanceScaleUI "ESM Distance Scale" \
                    range: [0, 100, 0.3] type: #float scale: 0.01 tooltip: ("Scale " \
                    + "factor for adjusting soft shadows to scenes of various " \
                    + "scales. It's generally useful to decrease this value " \
                    + "for larger scenes, especially if shadows still look " \
                    + "sharp no matter how big the blur radius is set.") \
                    enabled: (isESMEnabled shadowFilteringTypeUI)

            on shadowFilteringTypeUI selected i do (
                shadowFilteringType = SHADOW_FILTERING_TYPES[i]
                esmDistanceScaleUI.enabled = isESMEnabled shadowFilteringTypeUI
            )
        )

        label lbl2 offset:[0,-110] visible:on

        group "Animation" (

            checkbox animExportUI "Export Animations"
            checkbox animUsePlaybackRangeUI "Export within playback range"
            checkbox animStartWithZeroUI "Keyframes start with 0"

            on animExportUI changed arg do (
                animUsePlaybackRangeUI.enabled = arg
                animStartWithZeroUI.enabled = arg
            )

        )

        label lbl3 offset:[0,-102] visible:on

        group "Ambient Occlusion" (
            checkbox aoEnabledUI "Enabled"

            spinner aoDistanceUI "Distance" range: [0, 1000000, 1] type: #worldUnits enabled:false
            spinner aoFactorUI "Factor" range: [0, 100, 1]  type: #float enabled:false
            spinner aoTracePrecisionUI "Trace Precision" range: [0, 1, 0.25]  type: #float enabled:false
            checkbox aoBentNormalsUI "Bent Normals"

            on aoEnabledUI changed arg do (
                aoDistanceUI.enabled = arg
                aoFactorUI.enabled = arg
                aoTracePrecisionUI.enabled = arg
                aoBentNormalsUI.enabled = arg
            )
        )

        label lbl4 offset:[0,-146] visible:on

        group "Outlining Effect" (
            checkbox outlineEnabledUI "Enabled"

            spinner edgeStrengthUI "Edge Strength" range: [0, 100, 3] type: #float enabled:false
            spinner edgeGlowUI "Edge Glow" range: [0, 100, 0]  type: #float enabled:false
            spinner edgeThicknessUI "Edge Thickness" range: [0, 100, 1]  type: #float enabled:false
            spinner pulsePeriodUI "Pulse Period" range: [0, 1000, 0]  type: #float enabled:false

            colorpicker visibleEdgeColorUI "Visible Edge Color:" color:[0,0,0] modal:false enabled:false
            colorpicker hiddenEdgeColorUI "Hidden Edge Color:" color:[0,0,0] modal:false enabled:false
            checkbox renderHiddenEdgeUI "Render Hidden Edge"

            on outlineEnabledUI changed arg do (
                edgeStrengthUI.enabled = arg
                edgeGlowUI.enabled = arg
                edgeThicknessUI.enabled = arg
                pulsePeriodUI.enabled = arg
                visibleEdgeColorUI.enabled = arg
                hiddenEdgeColorUI.enabled = arg
                renderHiddenEdgeUI.enabled = arg
            )
        )


        fn visibleFromGroup name state = (
            setstate = off
            out = off
            for c in params.controls while not out do (
                if iskindof c GroupStartControl and c.text == name do setstate = on
                if setstate do c.visible = state
                if iskindof c GroupEndControl and c.text == name do out = on
            )
        )

        fn visiblePage page:0 = (
            for c in params.controls where c != tb do c.visible = off
            case page of (
                0: visibleFromGroup "Common Settings" on
                1: visibleFromGroup "Shadows" on
                2: visibleFromGroup "Animation" on
                3: visibleFromGroup "Ambient Occlusion" on
                4: visibleFromGroup "Outlining Effect" on
            )
        )

        on tb selected s e do (
            visiblePage page:e.TabPageIndex
        )

        on params open do (
            tb.itemSize = dotNetObject "System.Drawing.Size" (40*GetUIScaleFactor()) (24*GetUIScaleFactor())

            tb.TabPages.add "Common"
            tb.TabPages.add "Shadows"
            tb.TabPages.add "Animation"
            tb.TabPages.add "AO"
            tb.TabPages.add "Outlining"
            visiblePage page:0

            aaMethodUI.selection = findItem aaMethodUI.items aaMethod
            pmremMaxTileSizeUI.selection = findItem pmremMaxTileSizeUI.items pmremMaxTileSize
            iblEnvironmentModeUI.selection = findItem IBL_ENV_MODE_TYPES iblEnvironmentMode
            shadowFilteringTypeUI.selection = findItem \
                    SHADOW_FILTERING_TYPES shadowFilteringType
            esmDistanceScaleUI.enabled = isESMEnabled shadowFilteringTypeUI

            arg = animExportUI.checked
            animUsePlaybackRangeUI.enabled = arg
            animStartWithZeroUI.enabled = arg

            arg = aoEnabledUI.checked
            aoDistanceUI.enabled = arg
            aoFactorUI.enabled = arg
            aoTracePrecisionUI.enabled = arg
            aoBentNormalsUI.enabled = arg

            arg = outlineEnabledUI.checked
            edgeStrengthUI.enabled = arg
            edgeGlowUI.enabled = arg
            edgeThicknessUI.enabled = arg
            pulsePeriodUI.enabled = arg
            visibleEdgeColorUI.enabled = arg
            hiddenEdgeColorUI.enabled = arg
            renderHiddenEdgeUI.enabled = arg
        )
    )
)

::v3dMeshDataCA = attributes V3DMeshData attribID: #(0x53d74d27, 0x2c36214e)
(
    parameters main rollout:params
    (
        renderOrder type:#integer ui:renderOrderUI default:0
        frustumCulling type:#boolean ui:frustumCullingUI default:true
    )

    rollout params "Verge3D Rendering Params" rolledUp:true
    (
        label renderOrderLabelUI "Rendering Order:" across:2 align:#right
        spinner renderOrderUI "" range: [-1000000, 1000000, 0] type: #integer align:#left fieldWidth:67
                toolTip:("The rendering-order index. The smaller the index, the " +
                "earlier the object will be rendered. Useful for sorting " +
                "transparent objects.")
        checkbox frustumCullingUI "Frustum Culling" align:#center offset:[-30, 0]
                toolTip:"Enable frustum culling optimization technique"

        on params open do (
            if isProperty params #autoLayoutOnResize then params.autoLayoutOnResize = true
        )
    )
)

::v3dLightDataCA = attributes V3DLightData attribID: #(0x73ddeaa7, 0x17f07fab)
(
    parameters main rollout:params
    (
        shadowBias type:#float ui:shadowBiasUI default:1
        esmExponent type:#float ui:esmExponentUI default:2.5
    )

    rollout params "Verge3D Light Params" rolledUp:true
    (
        label shadowBiasLabelUI "Shadow Bias:" across:2 align:#right
        spinner shadowBiasUI "" range: [0, 100, 1] type: #float align:#left fieldWidth:67
                toolTip: "Additional shadow bias to reduce self-shadow artifacts"

        label esmExponentLabelUI "ESM Bias:" across:2 align:#right
        spinner esmExponentUI "" range: [1, 10000, 1] type: #float align:#left fieldWidth:67
                toolTip: "Exponential Shadow Map bias. Helps reducing light leaking artifacts."

        on params open do (
            if isProperty params #autoLayoutOnResize then params.autoLayoutOnResize = true
        )
    )
)

::v3dLineDataCA = attributes V3DLineData attribID: #(0x66a555fb, 0x17853f40)
(  
    parameters main rollout:params
    (
        enableLineRendering type:#boolean ui:enableLineRenderingUI default:false
        lineColor type:#color ui:lineColorUI default:[255,255,255]
        lineWidth type:#float ui:lineWidthUI default:1.0
    )

    rollout params "Verge3D Line Params" rolledUp:true
    (
        checkbox enableLineRenderingUI "Enable Line Rendering" align:#center offset:[-15, 0]

        label lineColorLabelUI "Line Color:" across:2 align:#right
        colorpicker lineColorUI "" align:#left
                color:[255,255,255] modal:false enabled:enableLineRenderingUI.checked
        
        label lineWidthLabelUI "Line Width (px):" across:2 align:#right
        spinner lineWidthUI "" range: [0, 1000000, 1] type: #float
                align:#left fieldWidth:67 toolTip: "Line width in pixels"
                enabled: enableLineRenderingUI.checked
        
        fn setVisibility arg = (
            lineColorLabelUI.enabled = arg
            lineColorUI.enabled = arg
            lineWidthLabelUI.enabled = arg
            lineWidthUI.enabled = arg
        )
        on enableLineRenderingUI changed arg do (
            setVisibility(arg)   
        )
        on params open do (
            if isProperty params #autoLayoutOnResize then params.autoLayoutOnResize = true
            setVisibility(enableLineRenderingUI.checked)
        )
    )
)

::v3dCameraDataCA = attributes V3DCameraData attribID: #(0x688aef96, 0x4b807b38)
(
    
    /*
     *   NOTE: The .pos attribute cannot be changed by script after the control has been created, so create 
     *   a control with "offset" flag and center alignment. Have to set the position of the controls 
     *   from the center of rollout's layout, due to a bug in maxscript with rollout's layout size.
    */

    parameters main rollout:params
    (
        viewportFitType type:#string default:"Vertical"

        panningEnabled type:#boolean ui:panningEnabledUI default:true

        rotateSpeed type:#float ui:rotateSpeedUI default:1
        moveSpeed type:#float ui:moveSpeedUI default:1

        minDist type:#worldUnits ui:minDistUI default:0
        maxDist type:#worldUnits ui:maxDistUI default:5000
        minZoom type:#worldUnits ui:minZoomUI default:0.01
        maxZoom type:#worldUnits ui:maxZoomUI default:100
        minAngle type:#angle ui:minAngleUI default:0
        maxAngle type:#angle ui:maxAngleUI default:180
        minAzimuthAngle type:#angle ui:minAzimuthAngleUI default:0
        maxAzimuthAngle type:#angle ui:maxAzimuthAngleUI default:360

        fpsEnabled type:#boolean ui:fpsEnabledUI default:false
        fpsCollisionMaterial type:#material ui:fpsCollisionMaterialUI
        fpsGazeLevel type:#worldUnits ui:fpsGazeLevelUI default:1.8
        fpsStoryHeight type:#worldUnits ui:fpsStoryHeightUI default:5
        enablePointerLock type:#boolean ui:enablePointerLockUI default:false

        disableControls type:#boolean ui:disableControlsUI default:false
    )

    rollout params "Verge3D Camera Params" rolledUp:true
    (
        label viewportFitTypeLabelUI "Viewport Fit: " across:2 align:#right 
        dropdownlist viewportFitTypeUI items:#("Vertical", "Horizontal") align:#left
                offset:[2, 0] toolTip:"How to fit image inside the viewport" 

        checkbox panningEnabledUI "Allow Panning" align:#center offset:[-34, 0]
        
        label rotateSpeedLabelUI "Rot. Speed:" across:2 align:#right
        spinner rotateSpeedUI "" range: [-100, 100, 1] type: #float fieldWidth:67
                toolTip: "Camera rotation speed factor" align:#left
        label moveSpeedLabelUI "Mov. Speed:" across:2 align:#right
        spinner moveSpeedUI "" range: [-100, 100, 1] type: #float fieldWidth:67
                toolTip: "Camera movement speed factor" align:#left

        groupBox groupOrbitCameraLimits "Targeted Camera Limits" align:#center width:162
            label minDistLabelUI "Persp Min Dist:" across:2 align:#right offset:[20, 0]
            spinner minDistUI "" range: [0, 1000000, 0] type: #worldUnits offset:[20, 0]
                    fieldWidth:42 align:#left 
            label maxDistLabelUI "Persp Max Dist:" across:2 align:#right offset:[20, 0]
            spinner maxDistUI "" range: [0, 1000000, 1000] type: #worldUnits offset:[20, 0]
                    fieldWidth:42 align:#left 
            label minZoomLabelUI "Ortho Min Zoom:" across:2 align:#right offset:[20, 0]
            spinner minZoomUI "" range: [0, 10000, 0.01] type: #worldUnits offset:[20, 0]
                    fieldWidth:40 align:#left 
            label maxZoomLabelUI "Ortho Max Zoom:" across:2 align:#right offset:[20, 0]
            spinner maxZoomUI "" range: [0, 10000, 100]  type: #worldUnits offset:[20, 0]
                    fieldWidth:40 align:#left 
            label minAngleLabelUI "Min Vertical Angle:" across:2 align:#right offset:[20, 0]
            spinner minAngleUI "" range: [0, 180, 0]  type: #float offset:[20, 0]
                    fieldWidth:37 align:#left 
            label maxAngleLabelUI "Max Vertical Angle:" across:2 align:#right offset:[20, 0]
            spinner maxAngleUI "" range: [0, 180, 180]  type: #float offset:[20, 0]
                    fieldWidth:37 align:#left 
            label minAzimuthAngleLabelUI "Min Horizon Angle:" across:2 align:#right offset:[20, 0]
            spinner minAzimuthAngleUI "" range: [-360, 360, 0] offset:[20, 0] 
                    type: #float fieldWidth:37 align:#left 
                    toolTip: "Set min/max range to 0-360 to disable horizontal limits" 
            label maxAzimuthAngleLabelUI "Max Horizon Angle:" across:2 align:#right offset:[20, 0]
            spinner maxAzimuthAngleUI "" range: [-360, 360, 360] offset:[20, 0]
                    type: #float fieldWidth:37 align:#left
                    toolTip: "Set min/max range to 0-360 to disable horizontal limits" 

        groupBox groupFirstPerson "First-Person" align:#center width:162 offset:[0, 5]

            checkbox fpsEnabledUI "Enabled" align:#center offset:[-48, 0]

            label fpsCollMatLabel "Collision Material:" 
            materialbutton fpsCollisionMaterialUI "Pick Material" width:100 

            button fpsCollisionMaterialClearUI "Clear" width:100 align:#center
            on fpsCollisionMaterialClearUI pressed do (
                fpsCollisionMaterialUI.material = undefined
            )

            label fpsGazeLevelLabelUI "Gaze Level:" across:2 align:#right
            spinner fpsGazeLevelUI "" range: [0, 1000000, 0] type: #worldUnits 
                    fieldWidth:67 align:#left
            label fpsStoryHeightLabelUI "Story Height:" across:2 align:#right
            spinner fpsStoryHeightUI "" range: [0, 1000000, 0] type: #worldUnits 
                    fieldWidth:67 align:#left
            checkbox enablePointerLockUI "Enable Pointer Lock" align:#center offset:[-22, 0]

        checkbox disableControlsUI "Disable Controls" align:#center offset:[-29, 5]

        on viewportFitTypeUI selected i do viewportFitType = viewportFitTypeUI.items[i]

        fn setVisibility = (
            -- restore viewport fit dropdown from string param
            viewportFitTypeUI.selection = findItem viewportFitTypeUI.items viewportFitType

            all = not disableControlsUI.checked
            fps = fpsEnabledUI.checked

            panningEnabledUI.enabled = all
            rotateSpeedUI.enabled = all
            rotateSpeedLabelUI.enabled = rotateSpeedUI.enabled
            moveSpeedUI.enabled = all
            moveSpeedLabelUI.enabled = moveSpeedUI.enabled

            minDistUI.enabled = all and not fps
            minDistLabelUI.enabled = minDistUI.enabled
            maxDistUI.enabled = all and not fps
            maxDistLabelUI.enabled = maxDistUI.enabled
            minZoomUI.enabled = all and not fps
            minZoomLabelUI.enabled = minZoomUI.enabled
            maxZoomUI.enabled = all and not fps
            maxZoomLabelUI.enabled = maxZoomUI.enabled
            minAngleUI.enabled = all and not fps
            minAngleLabelUI.enabled = minAngleUI.enabled
            maxAngleUI.enabled = all and not fps
            maxAngleLabelUI.enabled = maxAngleUI.enabled
            minAzimuthAngleUI.enabled = all and not fps
            minAzimuthAngleLabelUI.enabled = minAzimuthAngleUI.enabled
            maxAzimuthAngleUI.enabled = all and not fps
            maxAzimuthAngleLabelUI.enabled = maxAzimuthAngleUI.enabled

            fpsEnabledUI.enabled = all
            fpsCollMatLabel.enabled = all and fps
            fpsCollisionMaterialUI.enabled = all and fps
            fpsCollisionMaterialClearUI.enabled = all and fps
            fpsGazeLevelUI.enabled = all and fps
            fpsGazeLevelLabelUI.enabled = fpsGazeLevelUI.enabled
            fpsStoryHeightUI.enabled = all and fps
            fpsStoryHeightLabelUI.enabled = fpsStoryHeightUI.enabled
            enablePointerLockUI.enabled = all and fps
        )

        -- hack, resizing some UI
        fn updateUI = (
            groupOrbitCameraLimits.height = groupFirstPerson.pos[2] - groupOrbitCameraLimits.pos[2] - 5
            groupFirstPerson.height = disableControlsUI.pos[2] - groupFirstPerson.pos[2] - 5
            viewportFitTypeUI.width = 76
        )

        on params open do (
            if isProperty params #autoLayoutOnResize then params.autoLayoutOnResize = true
            updateUI()            
            setVisibility()
        )
        on fpsEnabledUI changed arg do (
            setVisibility()
        )
        on disableControlsUI changed arg do (
            setVisibility()
        )
        on params resized s do (
            if params.open then updateUI()
        )
    )
)

::v3dAnimDataCA = attributes V3DAnimData attribID: #(0x688aef98, 0x4b807b38)
(
    parameters main rollout:params
    (
        animAuto type:#boolean ui:animAutoUI default:true
        animLoop type:#string default:"Repeat"

        animRepeatInfinite type:#boolean ui:animRepeatInfiniteUI default:true
        animRepeatCount type:#float ui:animRepeatCountUI default:1
        animOffset type:#float ui:animOffsetUI default:0

        animUseCustomRange type:#boolean ui:animUseCustomRangeUI default:false
        animCustomRangeFrom type:#integer ui:animCustomRangeFromUI default:0
        animCustomRangeTo type:#integer ui:animCustomRangeToUI default:100

        animSkeletonRoot type:#boolean ui:animSkeletonRootUI default:false
    )

    rollout params "Verge3D Animation Params" rolledUp:true
    (
        checkbox animAutoUI "Auto Start" align:#center offset:[-43, 0] toolTip:"Auto start animation" 

        label animLoopLabelUI "Loop Mode: " across:2 align:#right
        dropdownlist animLoopUI items:#("Repeat", "Ping Pong", "Once") align:#left
                offset:[2, 0] toolTip:"Animation looping mode"

        checkbox animRepeatInfiniteUI "Repeat Infinitely" align:#center offset:[-29, 0]

        label animRepeatCountLabelUI "Repeat Count: " across:2 align:#right
        spinner animRepeatCountUI "" range: [-1000000, 1000000, 1] type: #float 
                enabled:false fieldWidth:67 align:#left
        label animOffsetLabelUI "Offset: " across:2 align:#right
        spinner animOffsetUI "" range: [-1000000, 1000000, 0] type: #float fieldWidth:67 
                align:#left toolTip:"Animation offset in frames"

        on animLoopUI selected i do animLoop = animLoopUI.items[i]

        on animRepeatInfiniteUI changed arg do (
            animRepeatCountLabelUI.enabled = not arg
            animRepeatCountUI.enabled = not arg
        )

        checkbox animUseCustomRangeUI "Custom Frame Range" align:#center offset:[-16, 0]

        label animCustomRangeFromLabelUI "From: " across:2 align:#right
        spinner animCustomRangeFromUI "" range: [-1000000, 1000000, 0] type: #integer 
                fieldWidth:67 align:#left
        label animCustomRangeToLabelUI "To: " across:2 align:#right
        spinner animCustomRangeToUI "" range: [-1000000, 1000000, 100] type: #integer 
                fieldWidth:67 align:#left
        checkbox animSkeletonRootUI "Skeleton Root" align:#center offset:[-34, 0] 
                toolTip:"Make this object a root of the skeleton"

        -- hack, resizing some UI
        fn updateUI = (
            animLoopUI.width = 76
        )
        fn setVisibility arg = ( 
            animCustomRangeFromLabelUI.enabled = arg
            animCustomRangeFromUI.enabled = arg
            animCustomRangeToLabelUI.enabled = arg
            animCustomRangeToUI.enabled = arg
        )

        on params open do (
            if isProperty params #autoLayoutOnResize then params.autoLayoutOnResize = true
            updateUI()

            -- restore loop dropdown from string param
            animLoopUI.selection = findItem animLoopUI.items animLoop

            animRepeatCountLabelUI.enabled = not animRepeatInfiniteUI.checked
            animRepeatCountUI.enabled = not animRepeatInfiniteUI.checked

            arg = animUseCustomRangeUI.checked
            setVisibility(arg)
        )

        on animUseCustomRangeUI changed arg do (
            setVisibility(arg)
        )
        
        on params resized s do (
            if params.open then updateUI()
        )
    )
)

::v3dAdvRenderDataCA = attributes V3DAdvRenderData attribID: #(0x506500d0, 0x60483322)
(
    parameters main rollout:params
    (
        canvasFitX type:#string default:"None"
        canvasFitY type:#string default:"None"
        canvasFitShape type:#string default:"Box"
        canvasFitOffset type:#worldUnits ui:canvasFitOffsetUI default:0

        canvasBreakEnabled type:#boolean ui:canvasBreakEnabledUI default:false
        canvasBreakMinWidth type:#integer ui:canvasBreakMinWidthUI default:0
        canvasBreakMaxWidth type:#integer ui:canvasBreakMaxWidthUI default:10000
        canvasBreakMinHeight type:#integer ui:canvasBreakMinHeightUI default:0
        canvasBreakMaxHeight type:#integer ui:canvasBreakMaxHeightUI default:10000
        canvasBreakOrientation type:#string default:"All"

        fixOrthoZoom type:#boolean ui:fixOrthoZoomUI default:false
        hidpiCompositing type:#boolean ui:hidpiCompositingUI default:false
    )

    rollout params "Verge3D Adv. Rendering Params" rolledUp:true
    (

        groupBox groupFitToCameraEdge "Fit To Camera Edge" align:#center width:162
            
            label canvasFitXLabelUI "Horizontal: " across:2 align:#right
            dropdownlist canvasFitXUI items:#("None", "Left", "Right", "Stretch") align:#left
                    offset:[2, 0] toolTip:"Horizontal canvas edge to fit object to"  

            label canvasFitYLabelUI "Vertical: " across:2 align:#right
            dropdownlist canvasFitYUI items:#("None", "Top", "Bottom", "Stretch") align:#left
                    offset:[2, 0] toolTip:"Vertical canvas edge to fit object to"

            label canvasFitShapeLabelUI "Shape: " across:2 align:#right
            dropdownlist canvasFitShapeUI items:#("Box", "Sphere", "Point") align:#left offset:[2, 0]
                    toolTip:"Canvas fit shape"

            label canvasFitOffsetLabelUI "Fit Offset: " across:2 align:#right
            spinner canvasFitOffsetUI "" range:[0, 10000, 0] type:#worldUnits fieldWidth:67 
                    align:#left toolTip:"Canvas fit offset"
        
            on canvasFitXUI selected i do canvasFitX = canvasFitXUI.items[i]
            on canvasFitYUI selected i do canvasFitY = canvasFitYUI.items[i]
            on canvasFitShapeUI selected i do canvasFitShape = canvasFitShapeUI.items[i]

        groupBox groupVisibilityBreakpoints "Visibility Breakpoints" align:#center width:162 offset:[0, 5]

            checkbox canvasBreakEnabledUI "Enabled" align:#center offset:[-48, 0]
                    toolTip:"Enable breakpoints to affect object visibility depending on canvas size and orientation"
            label canvasBreakMinWidthLabelUI "Min Width: " across:2 align:#right
            spinner canvasBreakMinWidthUI "" range:[0, 10000, 0] type:#integer fieldWidth:67 
                    align:#left toolTip:"Minimum canvas width the object stay visible"
            label canvasBreakMaxWidthLabelUI "Max Width: " across:2 align:#right
            spinner canvasBreakMaxWidthUI "" range:[0, 10000, 10000] type:#integer fieldWidth:67 
                    align:#left toolTip:"Maximum canvas width the object stay visible"
            label canvasBreakMinHeightLabelUI "Min Height: " across:2 align:#right
            spinner canvasBreakMinHeightUI "" range:[0, 10000, 0] type:#integer fieldWidth:67 
                    align:#left toolTip:"Minimum canvas height the object stay visible"
            label canvasBreakMaxHeightLabelUI "Max Height: " across:2 align:#right
            spinner canvasBreakMaxHeightUI "" range:[0, 10000, 10000] type:#integer fieldWidth:67 
                    align:#left toolTip:"Maximum canvas height the object stay visible"
            label canvasBreakOrientationLabelUI "Orientation: " across:2 align:#right
            dropdownlist canvasBreakOrientationUI items:#("All", "Landscape", "Portrait") align:#left
                    offset:[2, 0] toolTip:"Screen orientation the object stay visible"

            on canvasBreakOrientationUI selected i do canvasBreakOrientation = canvasBreakOrientationUI.items[i]
        
        checkbox hidpiCompositingUI "HiDPI Compositing"  align:#center offset:[-24, 0]
                toolTip:"Render this object (and its children) using the separate HiDPI (Retina) compositing pass"
        
        checkbox fixOrthoZoomUI "Fix Ortho Zoom"  align:#center offset:[-31, 0]
                toolTip:"Apply inverse orthographic camera zoom as scaling factor for this object"

        fn setBreakUIVisibility = (
            local brk = canvasBreakEnabledUI.checked
            canvasBreakMinWidthLabelUI.enabled = brk
            canvasBreakMinWidthUI.enabled = brk
            canvasBreakMaxWidthLabelUI.enabled = brk
            canvasBreakMaxWidthUI.enabled = brk
            canvasBreakMinHeightLabelUI.enabled = brk
            canvasBreakMinHeightUI.enabled = brk
            canvasBreakMaxHeightLabelUI.enabled = brk
            canvasBreakMaxHeightUI.enabled = brk
            canvasBreakOrientationLabelUI.enabled = brk
            canvasBreakOrientationUI.enabled = brk
        )

        on canvasBreakEnabledUI changed arg do (
            setBreakUIVisibility()
        )

        -- hack, resizing some UI
        fn updateUI = (
            groupFitToCameraEdge.height = groupVisibilityBreakpoints.pos[2] - groupFitToCameraEdge.pos[2] - 5
            groupVisibilityBreakpoints.height = hidpiCompositingUI.pos[2] - groupVisibilityBreakpoints.pos[2] - 5
            canvasFitXUI.width = 76
            canvasFitYUI.width = 76
            canvasFitShapeUI.width = 76
            canvasBreakOrientationUI.width = 76
        )

        on params open do (
            if isProperty params #autoLayoutOnResize then params.autoLayoutOnResize = true
            updateUI()

            local canvasFitEnabled = false
            -- search for any camera node referencing this object
            for node in (refs.dependentNodes (custAttributes.getOwner this)) do (
                if superClassOf node.parent == Camera do (
                    canvasFitEnabled = true
                )
            )

            canvasFitXLabelUI.enabled = canvasFitEnabled
            canvasFitXUI.enabled = canvasFitEnabled
            canvasFitYLabelUI.enabled = canvasFitEnabled
            canvasFitYUI.enabled = canvasFitEnabled
            canvasFitShapeLabelUI.enabled = canvasFitEnabled
            canvasFitShapeUI.enabled = canvasFitEnabled
            canvasFitOffsetLabelUI.enabled = canvasFitEnabled
            canvasFitOffsetUI.enabled = canvasFitEnabled

            setBreakUIVisibility()

            canvasFitXUI.selection = findItem canvasFitXUI.items canvasFitX
            canvasFitYUI.selection = findItem canvasFitYUI.items canvasFitY
            canvasFitShapeUI.selection = findItem canvasFitShapeUI.items canvasFitShape
            canvasBreakOrientationUI.selection = findItem canvasBreakOrientationUI.items canvasBreakOrientation
        )

        on params resized s do (
            if params.open then updateUI()
        )
    )
)

::v3dStandMaterialDataCA = attributes V3DMaterialData attribID: #(0x357fe514, 0x2cc4b377)
(
    parameters main rollout:params
    (
        alphaMode type:#string ui:alphaModeUI default:"Auto"
        depthWrite type:#boolean ui:depthWriteUI default:true
        depthTest type:#boolean ui:depthTestUI default:true
        dithering type:#boolean ui:ditheringUI default:false
    )

    rollout params "Verge3D Material Params" rolledUp:true
    (
        label alphaModeLabel "Alpha Mode:" offset:[0, 3] across:2 align: #left
        dropdownlist alphaModeUI items:#("Auto", "Opaque", "Blend", "Add", "Mask", "Coverage") align: #left

        on alphaModeUI selected i do alphaMode = alphaModeUI.items[i]
        -- restore dropdown UI from string param
        on params open do alphaModeUI.selection = findItem alphaModeUI.items alphaMode

        checkbox depthWriteUI "Depth Write"
        checkbox depthTestUI "Depth Test"

        checkbox ditheringUI "Dithering"
    )
)

::v3dPhysMaterialDataCA = attributes V3DMaterialData attribID: #(0x17010bf1, 0x298f2957)
(
    parameters main rollout:params
    (
        alphaMode type:#string ui:alphaModeUI default:"Auto"
        twoSided type:#boolean ui:twoSidedUI default:false
        depthWrite type:#boolean ui:depthWriteUI default:true
        depthTest type:#boolean ui:depthTestUI default:true
        dithering type:#boolean ui:ditheringUI default:false
        gltfCompat type:#boolean ui:gltfCompatUI default:false
    )

    rollout params "Verge3D Material Params" rolledUp:true
    (
        label alphaModeLabel "Alpha Mode:" offset:[0, 3] across:2 align: #left
        dropdownlist alphaModeUI items:#("Auto", "Opaque", "Blend", "Add", "Mask", "Coverage") align: #left

        on alphaModeUI selected i do alphaMode = alphaModeUI.items[i]
        -- restore dropdown UI from string param
        on params open do alphaModeUI.selection = findItem alphaModeUI.items alphaMode

        checkbox twoSidedUI "2-Sided"
        checkbox depthWriteUI "Depth Write"
        checkbox depthTestUI "Depth Test"
        checkbox ditheringUI "Dithering"
        checkbox gltfCompatUI "glTF 2.0 compatible"
    )
)

-- Basic materials such as Blend, Shellac etc

::v3dMaterialDataCA = attributes V3DMaterialData attribID: #(0x7d9c569a, 0x5a125a04)
(
    parameters main rollout:params
    (
        alphaMode type:#string ui:alphaModeUI default:"Auto"
        twoSided type:#boolean ui:twoSidedUI default:false
        depthWrite type:#boolean ui:depthWriteUI default:true
        depthTest type:#boolean ui:depthTestUI default:true
        dithering type:#boolean ui:ditheringUI default:false
    )

    rollout params "Verge3D Material Params" rolledUp:true
    (
        label alphaModeLabel "Alpha Mode:" offset:[0, 3] across:2 align: #left
        dropdownlist alphaModeUI items:#("Auto", "Opaque", "Blend", "Add", "Mask", "Coverage") align: #left

        on alphaModeUI selected i do alphaMode = alphaModeUI.items[i]
        -- restore dropdown UI from string param
        on params open do alphaModeUI.selection = findItem alphaModeUI.items alphaMode

        checkbox twoSidedUI "2-Sided"
        checkbox depthWriteUI "Depth Write"
        checkbox depthTestUI "Depth Test"
        checkbox ditheringUI "Dithering"
    )
)

::v3dGradientRampDataCA = attributes V3DGradientRampData attribID: #(0x356baa71, 0x25f03857)
(
    parameters main rollout:params
    (
        bakeToBitmap type:#boolean ui:bakeToBitmapUI default:false
    )

    rollout params "Verge3D Gradient Ramp Params" rolledUp:true
    (
        checkbox bakeToBitmapUI "Bake To Bitmap"
    )
)

::v3dTextureDataCA = attributes V3DTextureData attribID: #(0x5517886e, 0x435e9135)
(
    parameters main rollout:params
    (
        anisotropy type:#string ui:anisotropyUI default:"1"
        compressionMethod type:#string ui:compressionMethodUI default:"AUTO"
    )

    rollout params "Verge3D Texture Params" rolledUp:true
    (
        label anisotropyLabel "Anisotropic filtering:" offset:[-10, 3] across:2 align: #right
        dropdownlist anisotropyUI items:#("1", "2", "4", "8", "16") align: #right

        label compressionMethodLabel "Compression Method:" offset:[-10, 3] across:2 align: #right
        dropdownlist compressionMethodUI items:#("AUTO", "UASTC", "ETC1S", "DISABLE") align: #right

        on anisotropyUI selected i do anisotropy = anisotropyUI.items[i]
        on compressionMethodUI selected i do compressionMethod = compressionMethodUI.items[i]

        -- restore dropdown UI from string param
        on params open do (
            anisotropyUI.selection = findItem anisotropyUI.items anisotropy
            compressionMethodUI.selection = findItem compressionMethodUI.items compressionMethod
        )
    )
)

/*
If you are not supporting older versions of Max, then you can consider making some properties and methods private or protected. But that will break older versions.
*/
struct V3DManagerStruct (
    V3D_MENU_CONTEXT = 0xb6346199,

    autoAssignAttrs = true,--stores whether Verge3d is on or off
    settingsFile = "$userscripts/Verge3D/settings.ini", --file to store settings
    debug = false, --when on, print information to MAXScript listener
    hasInitiated = false, --has the python importer run?
    smeClock = dotNetObject "System.Windows.Forms.Timer",

    fn importPython = (

        if not hasInitiated then (
            pyPath = getThisScriptFilename()
            pyPath = substituteString pyPath "verge3d.ms" ".."
            pyPath = substituteString pyPath "\\" "/"
            python.Execute "import os, sys"
            python.Execute ("sys.path.append(os.path.normpath('" + pyPath + "'))")
            python.Execute "import max_plugin"

            pluginUtils = python.Import("pluginUtils")
            debug = pluginUtils.debug
        )

        if hasInitiated and debug then (
            format "V3D-INFO: Reloading Verge3D modules\n"
            -- COMPAT: < 2021
            python.Execute ("if sys.version_info[0] > 2:
    import importlib
else:
    import imp as importlib")
            python.Execute "importlib.reload(max_plugin)"
        )
    ),

    fn exportGLTF = (
        importPython()
        try
            python.Execute "max_plugin.exportGLTF()"
        catch (
            python.Execute "max_plugin.exportCleanup()"
            throw()
        )
    ),

    fn reexportAll = (
        importPython()
        try
            python.Execute "max_plugin.reexportAll()"
        catch (
            python.Execute "max_plugin.exportCleanup()"
            throw()
        )
    ),

    fn runAppManager = (
        importPython()
        python.Execute "max_plugin.runAppManager()"
    ),

    fn runUserManual = (
        importPython()
        python.Execute "max_plugin.runUserManual()"
    ),

    fn addCustomRootNodeAttributes = (
        if not isProperty rootNode #V3DExportSettingsData then (
            if debug == true then format "V3D-INFO: Assigning export settings on root node\n"
            custAttributes.add rootNode ::v3dExportSettingsDataCA
        )
    ),

    fn showExportSettings = (
        addCustomRootNodeAttributes()
        createdialog rootNode.V3DExportSettingsData.params
    ),

    fn sneakPeek = (
        importPython()
        try
            python.Execute "max_plugin.sneakPeek()"
        catch (
            python.Execute "max_plugin.exportCleanup()"
            throw()
        )
    ),

    fn regMenu = (
        if debug == true then format "V3D-INFO: Registering Verge3D menu\n"

        local oldMenu = menuMan.findMenu "Verge3D"
        if oldMenu != undefined then menuMan.unregisterMenu oldMenu

        local mainMenuBar = menuMan.getMainMenuBar()
        local subMenu = menuMan.createMenu "Verge3D"

        local exportItem = menuMan.createActionItem "exportGLTF" "Verge3D"
        local sneakPeekItem = menuMan.createActionItem "sneakPeek" "Verge3D"
        local appMgrItem = menuMan.createActionItem "runAppManager" "Verge3D"
        local expSetItem = menuMan.createActionItem "showExportSettings" "Verge3D"
        local turnOnItem = menuMan.createActionItem "autoAssignAttrs" "Verge3D"
        local runUserManItem = menuMan.createActionItem "runUserManual" "Verge3D"

        local sepItem = menuMan.createSeparatorItem()

        subMenu.addItem exportItem (subMenu.numItems() + 1)
        subMenu.addItem sepItem (subMenu.numItems() + 1)
        subMenu.addItem sneakPeekItem (subMenu.numItems() + 1)
        subMenu.addItem appMgrItem (subMenu.numItems() + 1)
        subMenu.addItem sepItem (subMenu.numItems() + 1)
        subMenu.addItem expSetItem (subMenu.numItems() + 1)
        subMenu.addItem turnOnItem (subMenu.numItems() + 1)
        subMenu.addItem sepItem (subMenu.numItems() + 1)
        subMenu.addItem runUserManItem (subMenu.numItems() + 1)

        subMenuItem = menuMan.createSubMenuItem "Verge3D" subMenu
        /* just before Help menu */
        subMenuIndex = mainMenuBar.numItems()
        mainMenuBar.addItem subMenuItem subMenuIndex

        menuMan.updateMenuBar()
    ),

    fn addCustomAttrs node = (
        if SuperClassOf node == GeometryClass and not isProperty node #V3DMeshData then (
            custAttributes.add node ::v3dMeshDataCA
        )

        if SuperClassOf node == camera and not isProperty node #V3DCameraData then (
            custAttributes.add node ::v3dCameraDataCA
        )

        if SuperClassOf node == light and not isProperty node #V3DLightData then (
            custAttributes.add node ::v3dLightDataCA
        )
        
        if (SuperClassOf node == GeometryClass or SuperClassOf node == shape) and
                not isProperty node #V3DLineData then (
            custAttributes.add node ::v3dLineDataCA
        )
        
        if not isProperty node #V3DAnimData then (
            custAttributes.add node ::v3dAnimDataCA
        )

        if not isProperty node #V3DAdvRenderData then (
            custAttributes.add node ::v3dAdvRenderDataCA
        )
    ),

    fn addCustomMaterialAttrs mat = (
        -- NOTE: can be called many times

        if classOf mat == Standardmaterial do (
            if not isProperty mat #V3DMaterialData then (
                custAttributes.add mat ::v3dStandMaterialDataCA
            )
        )

        if classOf mat == PhysicalMaterial do (
            -- backwards compatibility
            alphaMode
            if custAttributes.get mat ::v3dStandMaterialDataCA != undefined do (
                alphaMode = mat.alphaMode
                custAttributes.delete mat ::v3dStandMaterialDataCA
            )

            -- use 'if not isProperty mat #V3DMaterialData then (' after
            -- the 'backwards compatibility' lines will be dropped
            if custAttributes.get mat ::v3dPhysMaterialDataCA == undefined do (
                custAttributes.add mat ::v3dPhysMaterialDataCA
            )

            -- backwards compatibility
            if alphaMode != undefined do (
                mat.alphaMode = alphaMode
            )
        )

        if classOf mat == Blend or
                classOf mat == Shellac or
                classOf mat == ArnoldMapToMtl or
                classOf mat as string == "ai_standard_surface" or
                classOf mat as string == "ai_lambert" or
                classOf mat as string == "ai_mix_shader" or
                classOf mat as string == "ai_two_sided" or
                classOf mat as string == "MaxUsdPreviewSurface" do (
            if not isProperty mat #V3DMaterialData then (
                custAttributes.add mat ::v3dMaterialDataCA
            )
        )
    ),

    fn addCustomTextureAttrs tex = (
        -- NOTE: can be called many times

        if classOf tex == BitmapTexture do (
            if not isProperty tex #V3DTextureData then (
                custAttributes.add tex ::v3dTextureDataCA
            )
        )
    ),

    fn addCustomGradientRampAttrs texRamp = (
        -- NOTE: can be called many times

        if classOf texRamp == Gradient_Ramp do (
            if not isProperty texRamp #V3DGradientRampData then (
                custAttributes.add texRamp ::v3dGradientRampDataCA
            )
        )
    ),

    fn addCustomAttributesTo nodes = (
        for node in nodes do (
            addCustomAttrs(node)
        )
    ),
    fn addCustomAttrsSel = (
        addCustomAttributesTo selection
    ),
    fn removeCustomAttrsFrom nodes = (
        for node in nodes do (
            custAttributes.delete node ::v3dMeshDataCA
            custAttributes.delete node ::v3dCameraDataCA
            custAttributes.delete node ::v3dLightDataCA
            custAttributes.delete node ::v3dLineDataCA
            custAttributes.delete node ::v3dAnimDataCA
            custAttributes.delete node ::v3dAdvRenderDataCA
        )
    ),
    fn removeCustomAttrsSel = (
        removeCustomAttrsFrom selection
    ),

    fn cleanupScene = (
        removeCustomAttrsFrom $*

        custAttributes.delete rootNode ::v3dExportSettingsDataCA

        for mat in sceneMaterials do (
            custAttributes.delete mat ::v3dPhysMaterialDataCA
            custAttributes.delete mat ::v3dStandMaterialDataCA
            custAttributes.delete mat ::v3dMaterialDataCA

            if mat.numSubs > 0 then (
                for i = 1 to mat.numSubs do (
                    -- fixes crash with missing subs
                    if mat[i] != undefined do (
                        custAttributes.delete mat[i] ::v3dTextureDataCA
                        custAttributes.delete mat[i] ::v3dGradientRampDataCA
                    )
                )
            )
        )

        ::v3dReflectionCubemapAPI.cleanupScene()
        ::v3dClippingPlaneAPI.cleanupScene()
    ),

    fn nodeCreatedCallback = (
        node = callbacks.notificationParam()
        addCustomAttrs(node)
    ),

    fn matRefCallback = (
        mat = callbacks.notificationParam()
        addCustomMaterialAttrs(mat)
    ),

    fn traverseSlateMaterialEditor cb = (
        editor = trackViewNodes[#SME]
        if editor != undefined do (
            for i = 1 to editor.numSubs do (
                view = editor[i]
                for j = 1 to view.numSubs do (
                    node = view[j]
                    cb editor view node i j
                )
            )

        )
    ),

    fn addedTextureNodesHandler = (
        fn lastNodeCb editor view node viewIdx nodeIdx = (
            /**
             * it seems to be enough to just process the last node, if a new
             * node was added this should be it
             */
            if nodeIdx == view.numSubs and isProperty node "reference" and node.reference != undefined do (
                addCustomTextureAttrs(node.reference)
                addCustomGradientRampAttrs(node.reference)
            )
        )
        traverseSlateMaterialEditor(lastNodeCb)
    ),

    fn smeClockCb = (
        /**
         * Slate Material Editor can be unavailable as a track view node in some
         * cases (e.g. in the #systemPostReset callback), that's why this function
         * should be used as a timer callback to finally obtain the editor.
         */
        if trackViewNodes[#SME] != undefined then (
            dotnet.removeEventHandler smeClock "tick" smeClockCb
            smeClock.stop()

            /**
             * Watching name changes in the Slate Material Editor is a more
             * consistent way to detect added nodes than relying onto
             * NodeEventCallback or some other node/material callbacks.
             *
             * Also the "name" change event suits better than the others, because
             * apparently it's more guaranteed that the reference from SME's
             * Node Interface to the actual node already exists at the moment,
             * despite such attributes as "parameters" or "subAnimStructure".
             * See: https://help.autodesk.com/view/3DSMAX/2020/ENU/?guid=GUID-513285B3-DBF6-471E-B587-B5BE14D4D875
             */
            when name trackViewNodes[#SME] change id:#smeNameChanged do (
                addedTextureNodesHandler()
            )
        )
    ),

    fn removeCustomChangeHandlers = (
        deleteAllChangeHandlers id:#smeNameChanged
        dotnet.removeEventHandler smeClock "tick" smeClockCb
        smeClock.stop()
    ),

    fn regCustomChangeHandlers = (
        removeCustomChangeHandlers()
        dotnet.addEventHandler smeClock "tick" smeClockCb
        smeClock.interval = 100
        smeClock.start()
    ),

    fn maxStartupCallback = (
        if not hasInitiated then (
            importPython()
            python.Execute "max_plugin.init()"
            hasInitiated = true
            -- TODO: refactor menu registration condition
            if menuMan.registerMenuContext V3D_MENU_CONTEXT then (
                regMenu()
            )
            callbacks.removeScripts id:#v3dMaxStartupCallback
            if ::v3dManager.autoAssignAttrs == true then (
                addCustomRootNodeAttributes()
            )
        )
    ),

    fn systemPostNewCallback = (
        addCustomRootNodeAttributes()
        regCustomChangeHandlers()
    ),

    fn systemPostResetCallback = (
        addCustomRootNodeAttributes()
        regCustomChangeHandlers()
    ),

    fn addCustomAttributesOnFileOpen = (
        for node in objects do (
            addCustomAttrs(node)
        )

        /**
         * Materials don't need to be processed, because #mtlRefAdded is
         * triggered after opening a file anyway.
         */

        fn allNodesCb editor view node viewIdx nodeIdx = (
            if isProperty node "reference" and node.reference != undefined do (
                addCustomTextureAttrs(node.reference)
                addCustomGradientRampAttrs(node.reference)
            )
        )
        traverseSlateMaterialEditor(allNodesCb)
    ),

    fn addCustomAttributesOnFileMergeImport = (
        for node in objects do (
            addCustomAttrs(node)
        )

        /**
         * Materials don't need to be processed, because #mtlRefAdded is
         * triggered after merging a file or importing FBX anyway.
         *
         * Textures don't need to be processed, because SME triggers change
         * handlers for every added node when merging files or importing FBX.
         */
    ),

    fn filePostOpenProcessCallback = (
        addCustomRootNodeAttributes()
        regCustomChangeHandlers()
        addCustomAttributesOnFileOpen()
    ),

    fn filePostImportCallback = (
        addCustomAttributesOnFileMergeImport()
    ),

    fn filePostMergeCallback = (
        addCustomAttributesOnFileMergeImport()
    ),

    /*
     * Remember to namespace callbacks... since you don't want to accidentally
     * remove a callback from other scripts.
     */
    fn removeCallbacks = (
        callbacks.removeScripts id:#v3dSystemPostNewCallback
        callbacks.removeScripts id:#v3dSystemPostResetCallback
        callbacks.removeScripts id:#v3dFilePostOpenProcessCallback
        callbacks.removeScripts id:#v3dFilePostImportCallback
        callbacks.removeScripts id:#v3dFilePostMergeCallback
        callbacks.removeScripts id:#v3dMatRefCallback
        callbacks.removeScripts id:#v3dNodeCreatedCallback

        gc light:true
    ),

    fn startCallbacks = (
        removeCallbacks()
        callbacks.addScript #systemPostNew "::v3dManager.systemPostNewCallback()" id:#v3dSystemPostNewCallback
        callbacks.addScript #systemPostReset "::v3dManager.systemPostResetCallback()" id:#v3dSystemPostResetCallback
        callbacks.addScript #filePostOpenProcess "::v3dManager.filePostOpenProcessCallback()" id:#v3dFilePostOpenProcessCallback
        callbacks.addScript #postImport "::v3dManager.filePostImportCallback()" id:#v3dFilePostImportCallback

        /**
         * NOTE: #filePostMergeProcess isn't fired for the Import->Merge...
         * operation for some reason. Although, #filePostMerge works fine.
         */
        callbacks.addScript #filePostMerge "::v3dManager.filePostMergeCallback()" id:#v3dFilePostMergeCallback

        callbacks.addScript #mtlRefAdded "::v3dManager.matRefCallback()" id:#v3dMatRefCallback
        callbacks.addScript #nodeCreated "::v3dManager.nodeCreatedCallback()" id:#v3dNodeCreatedCallback
    ),

    fn santize = (
        if autoAssignAttrs == undefined then autoAssignAttrs = true
        autoAssignAttrs = autoAssignAttrs as BooleanClass
    ),
    fn saveSettings = (
        local iniExists = doesFileExist settingsFile
        if (not iniExists) then (
            local settingpath = ((symbolicPaths.getPathValue "$userScripts") + "/Verge3D")
            if (makeDir settingpath all:true == false ) then (
                iniExists = false
                local message =  "V3D-ERROR: Verge3D config directory doesn't exist and could not be created. Settings not saved."
                if not IsNetServer() then messagebox message else format "%\n" message
            ) else iniExists = true
        )
        if iniExists then (
            setINISetting settingsFile "Verge3D" "autoAssignAttrs" (autoAssignAttrs as String)
        )
    ),
    fn loadSettings = (
        local tmpState = getINISetting settingsFile "Verge3D" "autoAssignAttrs"
        try (
            if tmpState != undefined and tempState != "" then tmpState = tmpState as BooleanClass
        ) catch (
            format "V3D-ERROR: Problem loading Verge3D settings: %\n" (getCurrentException())
        )
        if classof tmpState == BooleanClass then autoAssignAttrs = tmpState
    ),
    fn turnOn = (
        autoAssignAttrs = true
        saveSettings()
        startCallbacks()
        regCustomChangeHandlers()
    ),
    fn turnOff = (
        autoAssignAttrs = false
        saveSettings()
        removeCallbacks()
        removeCustomChangeHandlers()
    ),
    fn init = (
        loadSettings()
        callbacks.removeScripts id:#v3dMaxStartupCallback
        callbacks.addScript #postSystemStartup "::v3dManager.maxStartupCallback()" id:#v3dMaxStartupCallback
        if autoAssignAttrs == true then turnOn()
    )
)

::v3dManager = V3DManagerStruct()
::v3dManager.init()

utility Verge3D "Verge3D"
(
    label workflowTools "Workflow Tools:"

    button exportBtn "Export glTF" width:150 height:25
    on exportBtn pressed do macros.run "Verge3D" "exportGLTF"

    button sneakPeekBtn "Sneak Peek" width:150 height:25
    on sneakPeekBtn pressed do macros.run "Verge3D" "sneakPeek"

    button appMgrBtn "Run App Manager" width:150 height:25
    on appMgrBtn pressed do macros.run "Verge3D" "runAppManager"

    button expSetBtn "Export Settings" width:150 height:25
    on expSetBtn pressed do macros.run "Verge3D" "showExportSettings"

    label objectTools "Object Tools:"

    button addCustomAttrsBtn "Add Verge3D Params" width:150 height:25
    on addCustomAttrsBtn pressed do ::v3dManager.addCustomAttrsSel()

    button removeCustomAttrsBtn "Remove Verge3D Params" width:150 height:25
    on removeCustomAttrsBtn pressed do ::v3dManager.removeCustomAttrsSel()

    label additionalTools "Additional Tools:"

    button registerMenuBtn "Register Verge3D Menu" width:150 height:25
    on registerMenuBtn pressed do ::v3dManager.regMenu()

    button cleanupSceneBtn "Cleanup Scene" width:150 height:25
    on cleanupSceneBtn pressed do ::v3dManager.cleanupScene()

    button reexportAllBtn "Reexport All glTF Files" width:150 height:25
    on reexportAllBtn pressed do macros.run "Verge3D" "reexportAll"
)
