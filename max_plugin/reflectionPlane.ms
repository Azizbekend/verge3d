/**
 * V3DReflectionPlane helper object.
 */
plugin simpleManipulator V3DReflectionPlane
    name: "ReflPlane"
    category: "Verge3D"
    classID: #(0x62ce3008, 0x37e1738e)
    version: 1 (

    local ComboBoxStyle = dotNetClass "System.Windows.Forms.ComboBoxStyle"

    local COL_LIGHT_BLUE = [0.4, 0.671, 0.976]
    local MAX_VALUE = 1.0e6

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
        planeSizeY type:#float default:0 ui:planeSizeYUI
        planeSizeX type:#float default:0 ui:planeSizeXUI

        influenceDistance type:#float default:1 ui:influenceDistanceUI
        falloff type:#float default:0.5 ui:falloffUI
        clipStart type:#float default:0.1 ui:clipStartUI

        visibilitySet type:#string default:"" ui:visibilitySetUI
        on visibilitySet set val do this.onSetListSet this.params.visibilitySetUI val

        visibilitySetInv type:#boolean default:false ui:visibilitySetInvUI

    )

    rollout params "Parameters" width:800 (

        group "General" (
            spinner planeSizeYUI "Length:" range:[0, MAX_VALUE, 0] type:#worldunits
                    tooltip:"Influence volume length"
            spinner planeSizeXUI "Width:" range:[0, MAX_VALUE, 0] type:#worldunits
                    tooltip:"Influence volume width"
            spinner influenceDistanceUI "Influence Dist:" range:[0, MAX_VALUE, 1] type:#worldunits
                    tooltip:"Influence volume height"
            spinner falloffUI "Falloff:" range:[0, 1, 0.5]
                    tooltip:"Control how fast the probe influence decreases"
            spinner clipStartUI "Clip Offset:" range: [0, MAX_VALUE, 0.1]
                    type:#worldunits
                    tooltip:("Objects located closer than this value won't " \
                    + "appear on the probe's environment map")
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
                    + "are rendered into the environment map")
        )

        on params open do (
            visibilitySetUI.DropDownStyle = ComboBoxStyle.DropDownList
            this.updateSetListSelectedOption visibilitySetUI visibilitySet
        )
    )

    /**
     * Disable manipulating other objects since it's not needed for the reflection plane
     */
    on canManipulate target return off

    tool create (
        on mousePoint click do case click of (
            1: nodeTM.translation = gridPoint
            2: #stop
        )

        on mouseMove click do case click of (
            2: (planeSizeX = abs gridDist.x
                planeSizeY = abs gridDist.y)
            3: #stop
        )
    )

    on updateGizmos do (
        this.clearGizmos()

        local gizmo = manip.makeGizmoShape()

        gizmo.startNewLine()
        gizmo.addPoint [planeSizeX, -planeSizeY, 0]
        gizmo.addPoint [planeSizeX, planeSizeY, 0]

        gizmo.startNewLine()
        gizmo.addPoint [-planeSizeX, -planeSizeY, 0]
        gizmo.addPoint [-planeSizeX, planeSizeY, 0]

        gizmo.startNewLine()
        gizmo.addPoint [-planeSizeX, -planeSizeY, 0]
        gizmo.addPoint [planeSizeX, -planeSizeY, 0]

        gizmo.startNewLine()
        gizmo.addPoint [-planeSizeX, planeSizeY, 0]
        gizmo.addPoint [planeSizeX, planeSizeY, 0]


        gizmo.startNewLine()
        gizmo.addPoint [planeSizeX, -planeSizeY, 0]
        gizmo.addPoint [planeSizeX, -planeSizeY, influenceDistance]

        gizmo.startNewLine()
        gizmo.addPoint [planeSizeX, planeSizeY, 0]
        gizmo.addPoint [planeSizeX, planeSizeY, influenceDistance]

        gizmo.startNewLine()
        gizmo.addPoint [-planeSizeX, planeSizeY, 0]
        gizmo.addPoint [-planeSizeX, planeSizeY, influenceDistance]

        gizmo.startNewLine()
        gizmo.addPoint [-planeSizeX, -planeSizeY, 0]

        gizmo.addPoint [-planeSizeX, -planeSizeY, influenceDistance]

        this.addGizmoShape gizmo 0 COL_LIGHT_BLUE [0,1,0]
    )

    on create do (
        /* nothing */
    )
)

struct V3DReflectionPlaneStruct (

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
        for h in helpers where (classOf h == V3DReflectionPlane and h.visibilitySet == oldName) do (
            h.visibilitySet = newName
        )
    ),

    fn selSetDeletedCallback setName = (
        if this.currentlyModifiedSet != setName do (
            for h in helpers where (classOf h == V3DReflectionPlane and h.visibilitySet == setName) do (
                h.visibilitySet = ""
            )
        )
    ),

    fn cleanupScene = (
        rcObjects = for h in helpers where classOf h == V3DReflectionPlane collect h
        undo on (
            delete rcObjects
        )
    ),

    fn init = (
        callbacks.removeScripts id:#v3dReflectionPlaneSelSetPreModifyCallback
        callbacks.addScript #NamedSelSetPreModify \
                "::v3dReflectionPlaneAPI.selSetPreModifyCallback (callbacks.notificationParam())" \
                id:#v3dReflectionPlaneSelSetPreModifyCallback

        callbacks.removeScripts id:#v3dReflectionPlaneSelSetPostModifyCallback
        callbacks.addScript #NamedSelSetPostModify \
                "::v3dReflectionPlaneAPI.selSetPostModifyCallback (callbacks.notificationParam())" \
                id:#v3dReflectionPlaneSelSetPostModifyCallback

        callbacks.removeScripts id:#v3dReflectionPlaneSelSetRenamedCallback
        callbacks.addScript #NamedSelSetRenamed \
                "::v3dReflectionPlaneAPI.selSetRenamedCallback (callbacks.notificationParam())" \
                id:#v3dReflectionPlaneSelSetRenamedCallback

        callbacks.removeScripts id:#v3dReflectionPlaneSelSetDeletedCallback
        callbacks.addScript #NamedSelSetDeleted \
                "::v3dReflectionPlaneAPI.selSetDeletedCallback (callbacks.notificationParam())" \
                id:#v3dReflectionPlaneSelSetDeletedCallback
    )
)

::v3dReflectionPlaneAPI = V3DReflectionPlaneStruct()
::v3dReflectionPlaneAPI.init()
