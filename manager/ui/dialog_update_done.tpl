<%inherit file="dialog.tpl"/>

<%block name="blkDialogId">diaUpdateDone</%block>

<%block name="blkDialogHeader">
  App Updated
</%block>

<%block name="blkDialogContent">
  <div class="dialog-text-center">
    <strong>${title}</strong> has been ${'successfully' if len(mergeConflicts) == 0 else ''} updated.
  </div>
  % if len(mergeConflicts):
    <div class="dialog-text red">
      The following file(s) had merge conflicts and were replaced by template-based files:
    </div>
    <ul><li>${'</li><li>'.join(mergeConflicts)}</li></ul>
    <div class="dialog-text">Check your app and restore these files from backup if anything is wrong.</div>
  % endif
  <button id="diaUpdateDoneOk" class="button">Ok</button>
</%block>

<%block name="blkDialogScript">
  diaUpdateDoneClose.addEventListener('click', function() {
      destroyDialog('diaUpdateDone');
  });
  diaUpdateDoneOk.addEventListener('click', function() {
      destroyDialog('diaUpdateDone');
  });

  focusElem('diaUpdateDoneOk');

  if (appManager.updatedAppIconElem) {

      var icon = appManager.updatedAppIconElem;

      if (icon.classList.contains('app-icon')) {
          icon.classList.remove('app-icon-update');
          icon.classList.add('app-icon-update-inactive');
      } else {
          icon.classList.add('hidden');
      }
  }
</%block>
