<%inherit file="dialog.tpl"/>

<%block name="blkDialogId">diaNetworkDownloadDone</%block>

<%block name="blkDialogHeader">
  Download Status
</%block>

<%block name="blkDialogContent">
  <div class="dialog-text">
    Operation complete: ${numFiles} file(s) successfuly downloaded.
  </div>
  <button id="diaNetworkDownloadDoneOk" class="button">Ok</button>
</%block>

<%block name="blkDialogScript">
  diaNetworkDownloadDoneOk.addEventListener('click', function() {
      destroyDialog("diaNetworkDownloadDone");
  });
  diaNetworkDownloadDoneClose.addEventListener('click', function() {
      destroyDialog("diaNetworkDownloadDone");
  });
</%block>
