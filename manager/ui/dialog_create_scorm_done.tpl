<%inherit file="dialog.tpl"/>

<%block name="blkDialogId">diaCreateScormDone</%block>

<%block name="blkDialogHeader">
  Downloading SCORM Package...
</%block>

<%block name="blkDialogContent">
  <div class="dialog-text-center">
    Your SCORM package is ready. If the download doesn't start immediately, click <a href="${downloadURL}" download id="downloadScormTemplate" class="colored-link">here</a>.
  </div>

  <div class="dialog-text-center">
    Follow <a href="https://www.soft8soft.com/docs/manual/en/introduction/Creating-SCORM-Courses.html" target="_blank" class="colored-link">next steps</a> as described in the User Manual.
  </div>

  <button id="diaCreateScormDoneOk" class="button">Got It!</button>
</%block>

<%block name="blkDialogScript">
  function diaCreateScormDoneCloseHandler() {
      destroyDialog('diaCreateScormDone');
  }

  diaCreateScormDoneClose.addEventListener('click', diaCreateScormDoneCloseHandler);
  diaCreateScormDoneOk.addEventListener('click', diaCreateScormDoneCloseHandler);

  focusElem('diaCreateScormDoneOk');
  document.getElementById('downloadScormTemplate').click();
</%block>
