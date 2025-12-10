/**
 * V3DClippingPlane helper object.
 */
plugin simpleManipulator V3DClippingPlane
    name: "ClippingPlane"
    category: "Verge3D"
    classID: #(0x62ce3002, 0x37e1737e)
    version: 1 (

    local ComboBoxStyle = dotNetClass "System.Windows.Forms.ComboBoxStyle"

    local COL_LIGHT_BLUE = [0.4, 0.671, 0.976]
    local MAX_VALUE = 1.0e6

    local RENDER_SIDE_TYPES = #("FRONT", "BACK", "DOUBLE")
    local RENDER_SIDE_DESC = #("Front Side", "Back Side", "Double-Sided")

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

    fn populateAffectedObjectsList = (
        local elem = this.params.affectedObjectsUI
        elem.items.clear()
        elem.items.add ""
        elem.items.addrange (for s in selectionSets collect s.name)
    )

    fn updateAffectedObjectsSelectedOption optionToSelect = (
        local elem = this.params.affectedObjectsUI
        this.populateAffectedObjectsList()
        elem.SelectedIndex = elem.FindStringExact optionToSelect
    )

    fn onAffectedObjectsSet val = (
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
        if (hasProperty this.params.affectedObjectsUI "enabled" and this.isCurrNodeSelected()) do (
            this.updateAffectedObjectsSelectedOption val
        )
        val
    )

    parameters main rollout:params (
        affectedObjects type:#string default:"" ui:affectedObjectsUI
        on affectedObjects set val do this.onAffectedObjectsSet val

        negated type:#boolean default:false ui:negatedUI
        clipShadows type:#boolean default:false ui:clipShadowsUI
        unionPlanes type:#boolean default:true ui:unionPlanesUI
        crossSection type:#boolean default:false ui:crossSectionUI
        crossSectionColor type:#point4 ui:crossSectionColorUI default:[0.5, 0, 0, 1]
        crossSectionRenderSide type:#string default:"FRONT" ui:crossSectionRenderSideUI
        crossSectionSize type:#float default:1000 ui:crossSectionSizeUI

        /* internal */
        planeSize type:#float default:0
    )

    rollout params "Parameters" width:800 (

        label affectedObjectsLabelUI "Affected Objects" align:#center
        dotNetControl affectedObjectsUI "System.Windows.Forms.ComboBox" height:22
        on affectedObjectsUI DropDown do (
            this.updateAffectedObjectsSelectedOption affectedObjects
        )
        on affectedObjectsUI SelectedIndexChanged do (
            affectedObjects = affectedObjectsUI.SelectedItem
        )

        checkbox negatedUI "Negated"
                tooltip: "Swap clipped and unclipped sides."
        checkbox clipShadowsUI "Clip Shadows"
                tooltip: "Clip shadows casted from the clipped objects."
        checkbox unionPlanesUI "Union Planes"
                tooltip: "Construct a union from all the clipping planes, affecting the object, not their intersection."
        checkbox crossSectionUI "Cross-Section"
                tooltip: "Fill cross-section between the clipping plane and the affected objects."
        colorpicker crossSectionColorUI "Cross-Section Color:" color:[0.5, 0, 0, 1] alpha:true modal:false enabled:false
                tooltip: "Cross-section diffuse color and opacity."
        dropdownlist crossSectionRenderSideUI "Render Side"
                items:RENDER_SIDE_DESC tooltip: "Cross-section rendering side."
        spinner crossSectionSizeUI "Cross-Sect. Size" range:[0, MAX_VALUE, 1000] type:#worldunits
                tooltip: "Cross-section plane size. Increase if you use larger scene size."

        on crossSectionRenderSideUI selected i do (
            crossSectionRenderSide = RENDER_SIDE_TYPES[i]
        )

        on unionPlanesUI changed arg do (
            crossSecEnabled = crossSectionUI.checked

            crossSectionUI.enabled = arg
            crossSectionColorUI.enabled = arg and crossSecEnabled
            crossSectionRenderSideUI.enabled = arg and crossSecEnabled
            crossSectionSizeUI.enabled = arg and crossSecEnabled
        )

        on crossSectionUI changed arg do (
            uniPlanesEnabled = unionPlanesUI.checked

            crossSectionColorUI.enabled = arg and uniPlanesEnabled
            crossSectionRenderSideUI.enabled = arg and uniPlanesEnabled
            crossSectionSizeUI.enabled = arg and uniPlanesEnabled
        )

        on params open do (
            affectedObjectsUI.DropDownStyle = ComboBoxStyle.DropDownList
            this.updateAffectedObjectsSelectedOption affectedObjects

            crossSectionRenderSideUI.selection = findItem \
                    RENDER_SIDE_TYPES crossSectionRenderSide

            uniPlanesEnabled = unionPlanesUI.checked
            crossSecEnabled = crossSectionUI.checked

            crossSectionUI.enabled = uniPlanesEnabled
            crossSectionColorUI.enabled = uniPlanesEnabled and crossSecEnabled
            crossSectionRenderSideUI.enabled = uniPlanesEnabled and crossSecEnabled
            crossSectionSizeUI.enabled = uniPlanesEnabled and crossSecEnabled
        )
    )

    /**
     * Disable manipulating other objects since it's not needed for the clipping plane
     */
    on canManipulate target return off

    tool create (
        on mousePoint click do case click of (
            1: nodeTM.translation = gridPoint
            2: #stop
        )

        on mouseMove click do case click of (
            2: planeSize = length gridDist
            3: #stop
        )
    )

    on updateGizmos do (
        this.clearGizmos()

        local gizmo = manip.makeGizmoShape()
        local halfEdgeSize = planeSize
        local upDownSize = halfEdgeSize/4

        if negated do upDownSize = -upDownSize

        gizmo.startNewLine()
        gizmo.addPoint [halfEdgeSize, -halfEdgeSize, 0]
        gizmo.addPoint [halfEdgeSize, halfEdgeSize, 0]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, -halfEdgeSize, 0]
        gizmo.addPoint [-halfEdgeSize, halfEdgeSize, 0]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, -halfEdgeSize, 0]
        gizmo.addPoint [halfEdgeSize, -halfEdgeSize, 0]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, halfEdgeSize, 0]
        gizmo.addPoint [halfEdgeSize, halfEdgeSize, 0]


        gizmo.startNewLine()
        gizmo.addPoint [halfEdgeSize, -halfEdgeSize, 0]
        gizmo.addPoint [halfEdgeSize, -halfEdgeSize, upDownSize]

        gizmo.startNewLine()
        gizmo.addPoint [halfEdgeSize, halfEdgeSize, 0]
        gizmo.addPoint [halfEdgeSize, halfEdgeSize, upDownSize]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, halfEdgeSize, 0]
        gizmo.addPoint [-halfEdgeSize, halfEdgeSize, upDownSize]

        gizmo.startNewLine()
        gizmo.addPoint [-halfEdgeSize, -halfEdgeSize, 0]

        gizmo.addPoint [-halfEdgeSize, -halfEdgeSize, upDownSize]

        this.addGizmoShape gizmo 0 COL_LIGHT_BLUE [0,1,0]
    )

    on create do (
        /* nothing */
    )
)

struct V3DClippingPlaneStruct (

    private
    /**
    * Individual "renamed" and "deleted" events should update the affectedObjects
    * value.
    * Such manipulations with a selection set as copying, cutting, pasting, adding
    * and substracting objects are implemented in Max via deleting and recreating
    * the entire selection set. Need to guard selSetDeletedCallback with the check
    * for the currently modified set to prevent resetting the affectedObjects value.
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
        for h in helpers where (classOf h == V3DClippingPlane and h.affectedObjects == oldName) do (
            h.affectedObjects = newName
        )
    ),

    fn selSetDeletedCallback setName = (
        if this.currentlyModifiedSet != setName do (
            for h in helpers where (classOf h == V3DClippingPlane and h.affectedObjects == setName) do (
                h.affectedObjects = ""
            )
        )
    ),

    fn cleanupScene = (
        rcObjects = for h in helpers where classOf h == V3DClippingPlane collect h
        undo on (
            delete rcObjects
        )
    ),

    fn init = (
        callbacks.removeScripts id:#v3dClippingPlaneSelSetPreModifyCallback
        callbacks.addScript #NamedSelSetPreModify \
                "::v3dClippingPlaneAPI.selSetPreModifyCallback (callbacks.notificationParam())" \
                id:#v3dClippingPlaneSelSetPreModifyCallback

        callbacks.removeScripts id:#v3dClippingPlaneSelSetPostModifyCallback
        callbacks.addScript #NamedSelSetPostModify \
                "::v3dClippingPlaneAPI.selSetPostModifyCallback (callbacks.notificationParam())" \
                id:#v3dClippingPlaneSelSetPostModifyCallback

        callbacks.removeScripts id:#v3dClippingPlaneSelSetRenamedCallback
        callbacks.addScript #NamedSelSetRenamed \
                "::v3dClippingPlaneAPI.selSetRenamedCallback (callbacks.notificationParam())" \
                id:#v3dClippingPlaneSelSetRenamedCallback

        callbacks.removeScripts id:#v3dClippingPlaneSelSetDeletedCallback
        callbacks.addScript #NamedSelSetDeleted \
                "::v3dClippingPlaneAPI.selSetDeletedCallback (callbacks.notificationParam())" \
                id:#v3dClippingPlaneSelSetDeletedCallback
    )
)

::v3dClippingPlaneAPI = V3DClippingPlaneStruct()
::v3dClippingPlaneAPI.init()
