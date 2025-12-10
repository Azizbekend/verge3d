/**
 * V3DReflectionCubemap helper object.
 */

plugin simpleManipulator V3DReflectionCubemap
    name:"ReflCubemap"
    category:"Verge3D"
    classID:#(0x3665fb25, 0x51b8242b)
    version:1 (

    local ComboBoxStyle = dotNetClass "System.Windows.Forms.ComboBoxStyle"

    local NUM_LINE_SEGMENTS_CIRCLE = 30
    local COL_LIGHT_BLUE = [0.4, 0.671, 0.976]
    local VOLUME_TYPE_ENUM = #("ELIPSOID", "BOX")
    local VOLUME_TYPE_ENUM_UI = #("Sphere", "Box")
    local MAX_VALUE = 1.0e10

    /**
     * Get the plugin node which the UI element is currently referring to
     * (basically, the plugin node which is currently selected).
     */
    fn getCurrNode = (
        (refs.dependentNodes this)[1]
    )

    /**
     * Check if the current plugin node is selected. Can be used to detect
     * situations when the plugin's UI is working with an unselected plugin node,
     * which may cause issues or be a result of a bug.
     */
    fn isCurrNodeSelected = (
        currNode = this.getCurrNode()
        currSelected = for obj in selection where obj == currNode collect obj
        currSelected.count > 0
    )

    fn createEllipsoidGizmo radius = (
        local gizmo = manip.makeGizmoShape()

        local lineXY = #(), lineXZ = #(), lineYZ = #()
        for i = 1 to NUM_LINE_SEGMENTS_CIRCLE do (
            local t = 360 * (i - 1) / (NUM_LINE_SEGMENTS_CIRCLE - 1)
            local rSint = radius * sin t, rCost = radius * cos t
            lineXY[i] = [rCost, rSint, 0]
            lineXZ[i] = [rCost, 0, rSint]
            lineYZ[i] = [0, rCost, rSint]
        )

        gizmo.startNewLine()
        for i = 1 to lineXY.count do gizmo.addPoint lineXY[i]
        gizmo.startNewLine()
        for i = 1 to lineXZ.count do gizmo.addPoint lineXZ[i]
        gizmo.startNewLine()
        for i = 1 to lineYZ.count do gizmo.addPoint lineYZ[i]

        gizmo
    )

    fn createBoxGizmo halfEdgeSize = (
        local gizmo = manip.makeGizmoShape()

        -- bottom square
        gizmo.startNewLine()
        gizmo.addPoint [halfEdgeSize, -halfEdgeSize, -halfEdgeSize]
        gizmo.addPoint [halfEdgeSize, halfEdgeSize, -halfEdgeSize]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, -halfEdgeSize, -halfEdgeSize]
        gizmo.addPoint [-halfEdgeSize, halfEdgeSize, -halfEdgeSize]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, -halfEdgeSize, -halfEdgeSize]
        gizmo.addPoint [halfEdgeSize, -halfEdgeSize, -halfEdgeSize]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, halfEdgeSize, -halfEdgeSize]
        gizmo.addPoint [halfEdgeSize, halfEdgeSize, -halfEdgeSize]

        -- top square
        gizmo.startNewLine()
        gizmo.addPoint [halfEdgeSize, -halfEdgeSize, halfEdgeSize]
        gizmo.addPoint [halfEdgeSize, halfEdgeSize, halfEdgeSize]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, -halfEdgeSize, halfEdgeSize]
        gizmo.addPoint [-halfEdgeSize, halfEdgeSize, halfEdgeSize]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, -halfEdgeSize, halfEdgeSize]
        gizmo.addPoint [halfEdgeSize, -halfEdgeSize, halfEdgeSize]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, halfEdgeSize, halfEdgeSize]
        gizmo.addPoint [halfEdgeSize, halfEdgeSize, halfEdgeSize]

        -- side lines
        gizmo.startNewLine()
        gizmo.addPoint [halfEdgeSize, -halfEdgeSize, -halfEdgeSize]
        gizmo.addPoint [halfEdgeSize, -halfEdgeSize, halfEdgeSize]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, -halfEdgeSize, -halfEdgeSize]
        gizmo.addPoint [-halfEdgeSize, -halfEdgeSize, halfEdgeSize]

        gizmo.startNewLine()
        gizmo.addPoint [halfEdgeSize, halfEdgeSize, -halfEdgeSize]
        gizmo.addPoint [halfEdgeSize, halfEdgeSize, halfEdgeSize]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, halfEdgeSize, -halfEdgeSize]
        gizmo.addPoint [-halfEdgeSize, halfEdgeSize, halfEdgeSize]

        gizmo
    )

    fn populateSetList uiSetElem = (
        uiSetElem.items.clear()
        uiSetElem.items.add ""
        uiSetElem.items.addrange (for s in selectionSets collect s.name)
    )

    fn updateSetListSelectedOption uiSetElem optionToSelect = (
        this.populateSetList(uiSetElem)
        uiSetElem.SelectedIndex = uiSetElem.FindStringExact optionToSelect
    )

    fn onSetListSet uiSetElem val = (
        /**
         * The UI element may not be initialized yet, just checking if a random
         * dotNetControl property exists.
         *
         * NOTE: Don't do any UI updates for the node if it's not selected. This
         * fixes the bug when merging in a scene with the plugin nodes in it
         * causes the currently selected plugin node in the current scene to
         * reset its *Set properties due to incorrect update events for the UI
         * elements.
         */
        if (hasProperty uiSetElem "enabled" and this.isCurrNodeSelected()) do (
            this.updateSetListSelectedOption uiSetElem val
        )
        val
    )

    parameters main rollout:params (

        influenceType type:#string ui:influenceTypeUI

        influenceDistance type:#float default:1 ui:influenceDistanceUI
        intensity type:#float default:1 ui:intensityUI
        clipStart type:#float default:0.1 ui:clipStartUI
        clipEnd type:#float default:100 ui:clipEndUI

        visibilitySet type:#string default:"" ui:visibilitySetUI
        on visibilitySet set val do this.onSetListSet this.params.visibilitySetUI val

        visibilitySetInv type:#boolean default:false ui:visibilitySetInvUI

        useCustomParallax type:#boolean default:false ui:useCustomParallaxUI
        parallaxType type:#string ui:parallaxTypeUI
        parallaxDistance type:#float default:1 ui:parallaxDistanceUI

        useCustomInfluence type:#boolean default:false ui:useCustomInfluenceUI
        influenceSet type:#string default:"" ui:influenceSetUI
        on influenceSet set val do this.onSetListSet this.params.influenceSetUI val

        influenceSetInv type:#boolean default:false ui:influenceSetInvUI
    )

    rollout params "Parameters" width:800 (

        group "General" (
            label influenceTypeLabelUI "Influence Type" align:#center
            dropdownlist influenceTypeUI items:VOLUME_TYPE_ENUM_UI
                    tooltip:"The type of the influence volume."
            on influenceTypeUI selected i do influenceType = VOLUME_TYPE_ENUM[i]

            label influenceDistanceLabelUI "Influence Distance" align:#center
            spinner influenceDistanceUI range:[0, MAX_VALUE, 1] type:#worldunits
                    align:#center width:100
                    tooltip:"The size of the influence volume."

            spinner intensityUI "Intensity" range:[0, MAX_VALUE, 1]
                    tooltip:"The intensity of the probe's environment map."

            spinner clipStartUI "Clipping Start" range: [0, MAX_VALUE, 0.1]
                    type:#worldunits
                    tooltip:("Objects located closer than this value won't " \
                    + "appear on the probe's environment map.")

            spinner clipEndUI "Clipping End" range: [0, MAX_VALUE, 100]
                    type:#worldunits
                    tooltip:("Objects located further than this value won't " \
                    + "appear on the probe's environment map.")
        )

        group "Visibility Selection Set" (
            dotNetControl visibilitySetUI "System.Windows.Forms.ComboBox" height:22
            on visibilitySetUI DropDown do (
                this.updateSetListSelectedOption visibilitySetUI visibilitySet
            )
            on visibilitySetUI SelectedIndexChanged do (
                visibilitySet = visibilitySetUI.SelectedItem
            )

            checkbox visibilitySetInvUI "Invert Visibility" align:#center
                    tooltip:("If Visibility Set is specified then this option " \
                    + "makes it so only objects that don't belong to the set " \
                    + "are rendered into the environment map.")
        )

        group "Custom Parallax" (
            checkbox useCustomParallaxUI "Enable Custom Parallax"
                    tooltip:"Enable Custom Parallax options."

            label parallaxTypeLabelUI "Parallax Type" align:#center
            dropdownlist parallaxTypeUI items:VOLUME_TYPE_ENUM_UI
                    tooltip:"The type of the parallax volume."
            on parallaxTypeUI selected i do parallaxType = VOLUME_TYPE_ENUM[i]

            label parallaxDistanceLabelUI "Parallax Distance" align:#center
            spinner parallaxDistanceUI range:[0, MAX_VALUE, 1] type:#worldunits
                    align:#center width:100
                    tooltip:"The size of the parallax volume."

            on useCustomParallaxUI changed arg do (
                parallaxTypeLabelUI.enabled = arg
                parallaxTypeUI.enabled = arg
                parallaxDistanceLabelUI.enabled = arg
                parallaxDistanceUI.enabled = arg
            )
        )

        group "Custom Influence" (
            checkbox useCustomInfluenceUI "Enable Custom Influence"
                    tooltip:("This group of options is used for selecting " \
                    + "objects that will be affected by this probe. The " \
                    + "Influence Selection Set option (if set to a non-empty " \
                    + "value) will be used instead of the Influence Type and " \
                    + "Influence Distance settings.")

            label influenceSetLabelUI "Influence Selection Set" align:#center
            dotNetControl influenceSetUI "System.Windows.Forms.ComboBox" height:22
            on influenceSetUI DropDown do (
                this.updateSetListSelectedOption influenceSetUI influenceSet
            )
            on influenceSetUI SelectedIndexChanged do (
                influenceSet = influenceSetUI.SelectedItem
            )

            checkbox influenceSetInvUI "Invert Influence" align:#center
                    tooltip:("If Influence Set is specified then this option " \
                    + "makes it so only objects that don't belong to the set " \
                    + "are affected by this probe.")

            on useCustomInfluenceUI changed arg do (
                influenceSetLabelUI.enabled = arg
                influenceSetUI.enabled = arg
                influenceSetInvUI.enabled = arg
            )
        )

        on params open do (
            visibilitySetUI.DropDownStyle = ComboBoxStyle.DropDownList
            this.updateSetListSelectedOption visibilitySetUI visibilitySet

            influenceTypeUI.selection = findItem VOLUME_TYPE_ENUM influenceType
            parallaxTypeUI.selection = findItem VOLUME_TYPE_ENUM parallaxType

            arg = useCustomParallaxUI.checked
            parallaxTypeLabelUI.enabled = arg
            parallaxTypeUI.enabled = arg
            parallaxDistanceLabelUI.enabled = arg
            parallaxDistanceUI.enabled = arg

            influenceSetUI.DropDownStyle = ComboBoxStyle.DropDownList
            this.updateSetListSelectedOption influenceSetUI influenceSet

            arg = useCustomInfluenceUI.checked
            influenceSetLabelUI.enabled = arg
            influenceSetUI.enabled = arg
            influenceSetInvUI.enabled = arg
        )
    )

    /**
     * Disable manipulating other objects since it's not needed for the
     * reflection cubemap.
     */
    on canManipulate target return off

    tool create (
        on mousePoint click do case click of (
            1: nodeTM.translation = gridPoint
            2: #stop
        )

        on mouseMove click do case click of (
            2: influenceDistance = length gridDist
            3: #stop
        )
    )

    on updateGizmos do (
        this.clearGizmos()

        local gizmo

        case this.influenceType of (
            "ELIPSOID": (
                gizmo = createEllipsoidGizmo influenceDistance
            )
            "BOX": (
                gizmo = createBoxGizmo influenceDistance
            )
        )

        this.addGizmoShape gizmo 0 COL_LIGHT_BLUE [0, 1, 0]
    )

    on create do (
        /**
         * Using the "create" event handler to specify default values for the
         * plugin parameters. Local plugin variables are not available within a
         * parameter block, so can't use them to set default parameter values in
         * a straightforward way.
         */
        influenceType = VOLUME_TYPE_ENUM[1]
        parallaxType = VOLUME_TYPE_ENUM[1]
    )
)

struct V3DReflectionCubemapStruct (

    private
    /**
    * Individual "renamed" and "deleted" events should update the visibilitySet
    * value.
    * Such manipulations with a selection set as copying, cutting, pasting, adding
    * and substracting objects are implemented in Max via deleting and recreating
    * the entire selection set. Need to guard selSetDeletedCallback with the check
    * for the currently modified set to prevent resetting the visibilitySet value.
    */
    currentlyModifiedSet = "",

    public
    fn selSetPreModifyCallback setName = (
        this.currentlyModifiedSet = setName
    ),

    fn selSetPostModifyCallback setName = (
        this.currentlyModifiedSet = ""
    ),

    fn selSetRenamedCallback param = (
        local oldName = param[1], newName = param[2]

        for h in helpers where (classOf h == V3DReflectionCubemap) do (
            if h.visibilitySet == oldName do (
                h.visibilitySet = newName
            )

            if h.influenceSet == oldName do (
                h.influenceSet = newName
            )
        )
    ),

    fn selSetDeletedCallback setName = (
        if this.currentlyModifiedSet != setName do (
            for h in helpers where (classOf h == V3DReflectionCubemap) do (
                if h.visibilitySet == setName do (
                    h.visibilitySet = ""
                )

                if h.influenceSet == setName do (
                    h.influenceSet = ""
                )
            )
        )
    ),

    fn cleanupScene = (
        rcObjects = for h in helpers where classOf h == V3DReflectionCubemap collect h
        undo on (
            delete rcObjects
        )
    ),

    fn init = (
        callbacks.removeScripts id:#v3dReflectionCubemapSelSetPreModifyCallback
        callbacks.addScript #NamedSelSetPreModify \
                "::v3dReflectionCubemapAPI.selSetPreModifyCallback (callbacks.notificationParam())" \
                id:#v3dReflectionCubemapSelSetPreModifyCallback

        callbacks.removeScripts id:#v3dReflectionCubemapSelSetPostModifyCallback
        callbacks.addScript #NamedSelSetPostModify \
                "::v3dReflectionCubemapAPI.selSetPostModifyCallback (callbacks.notificationParam())" \
                id:#v3dReflectionCubemapSelSetPostModifyCallback

        callbacks.removeScripts id:#v3dReflectionCubemapSelSetRenamedCallback
        callbacks.addScript #NamedSelSetRenamed \
                "::v3dReflectionCubemapAPI.selSetRenamedCallback (callbacks.notificationParam())" \
                id:#v3dReflectionCubemapSelSetRenamedCallback

        callbacks.removeScripts id:#v3dReflectionCubemapSelSetDeletedCallback
        callbacks.addScript #NamedSelSetDeleted \
                "::v3dReflectionCubemapAPI.selSetDeletedCallback (callbacks.notificationParam())" \
                id:#v3dReflectionCubemapSelSetDeletedCallback
    )
)

::v3dReflectionCubemapAPI = V3DReflectionCubemapStruct()
::v3dReflectionCubemapAPI.init()
