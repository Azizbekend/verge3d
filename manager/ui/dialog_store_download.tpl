<%inherit file="dialog.tpl"/>

<%block name="blkDialogId">diaStoreDownload</%block>

<%block name="blkDialogHeader">
  Downloading from Verge3D Demo Store...
</%block>

<%block name="blkDialogContent">
  <div id="downloadPercentageCont">
    <div class="spinner-preloader-cont">
      <div class="spinner-preloader-percentage" id='downloadPercentage'>50%</div>
      <div class="spinner-preloader"></div>
    </div>
    <button class="button" id="cancelStoreDownloading">Cancel</button>
  </div>
</%block>

<%block name="blkDialogScript">
    ${parent.blkDialogScript()}

    cancelStoreDownloading.addEventListener('click', function(event) {
        makeRequest('/store/?req=cancel', null, null);
    });
</%block>
