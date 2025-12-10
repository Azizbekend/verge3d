<%inherit file="dialog.tpl"/>

<%block name="blkDialogId">diaHelp</%block>

<%block name="blkDialogHeader">
  Support and Documentation
</%block>

<%block name="blkDialogContent">
  <div class="dialog-icon dialog-icon-help"></div>

  <div class="dialog-right-content-short">
    <div class="dialog-text">
      <a href="${manualURL}" target="_blank" title="Open Verge3D user manual" rel="noreferrer">
        <span class="help-link-icon help-link-icon-manual"></span>
        <span class="help-link-text">User Manual</span>
      </a>
    </div>

    <div class="dialog-text">
      <a href="https://www.soft8soft.com/forums" target="_blank" title="Seek for assistance on Soft8Soft forums" rel="noreferrer">
        <span class="help-link-icon help-link-icon-forums"></span>
        <span class="help-link-text">Community Forums</span>
      </a>
    </div>

    <div class="dialog-text">
      <a href="https://www.soft8soft.com/wiki/index.php/Main_Page" target="_blank" title="Open Verge3D Wiki" rel="noreferrer">
        <span class="help-link-icon help-link-icon-webgl-report"></span>
        <span class="help-link-text">Verge3D Wiki</span>
      </a>
    </div>

    <div class="dialog-text">
      <a href="${manualURL.replace('/manual/', '/api/')}" target="_blank" title="Open Verge3D developer API reference" rel="noreferrer">
        <span class="help-link-icon help-link-icon-manual"></span>
        <span class="help-link-text">Developer API Reference</span>
      </a>
    </div>

    <div class="dialog-text">
      <a href="https://cdn.soft8soft.com/demo/examples/index.html" target="_blank" title="Open advanced code examples" rel="noreferrer">
        <span class="help-link-icon help-link-icon-code-examples"></span>
        <span class="help-link-text">Code Examples</span>
      </a>
    </div>
  </div>
</%block>
