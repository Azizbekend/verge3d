<%inherit file="dialog.tpl"/>

<%block name="blkDialogId">diaStoreDownloadDone</%block>

<%block name="blkDialogHeader">
  Download Status
</%block>

<%block name="blkDialogContent">
  <div class="dialog-text">
    Operation complete: demo <a href="${manageURL}" class="colored-link">${nameDisp}</a> successfuly downloaded.
  </div>
  <button id="diaStoreDownloadDoneOk" class="button">Ok</button>
</%block>

<%block name="blkDialogScript">
  diaStoreDownloadDoneOk.addEventListener('click', function() {
      destroyDialog("diaStoreDownloadDone");
  });
  diaStoreDownloadDoneClose.addEventListener('click', function() {
      destroyDialog("diaStoreDownloadDone");
  });
</%block>
