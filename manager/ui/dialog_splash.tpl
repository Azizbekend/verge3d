<%inherit file="dialog.tpl"/>

<%block name="blkDialogId">diaSplashScreen</%block>

<%block name="blkDialogHeader">
  Setup Verge3D
</%block>

<%block name="blkDialogContent">
  <form action="" id="settingsForm">
    <div class="splash-title">
      <div class="splash-title-image"></div>
      <div class="splash-title-text">
        <div class="splash-title-text-top">Welcome</div>
        <div class="splash-title-text-bottom">to Verge3D ${version}!</div>
      </div>
      <div class="splash-title-bar"></div>
    </div>

    <div class="splash-settings-container">
      <hr>

      <div class="dialog-text settings-text">
        Applications Folder:
        <input type="text" name="extAppsDirectory" value="${settings['extAppsDirectory']}" class="dialog-wide-input">
        <div id="diaSplashScreenSelectFolder" class="settings-select-folder"></div>
      </div>

      <div style="dialog-text settings-text">
        Select Theme:

        <select name="theme" class="settings-theme">
          <option value="light" ${'selected' if settings['theme'] == 'light' else ''}>Light</option>
          <option value="dark" ${'selected' if settings['theme'] == 'dark' else ''}>Dark</option>
        </select>
      </div>

      <hr>
    </div>

    <input type="submit" id="splashScreenSave" value="Apply" class="button settings-button">
  </form>
</%block>

<%block name="blkDialogScript">
  settingsForm.addEventListener('submit', function(event) {

      var data = {};

      settingsForm.querySelectorAll('input').forEach(function(elem) {
          switch (elem.type) {
              case 'checkbox':
                  data[elem.name] = elem.checked;
                  break;
              case 'text':
                  data[elem.name] = elem.value;
                  break;
              case 'number':
                  data[elem.name] = Number(elem.value);
                  break;
          }
      });

      settingsForm.querySelectorAll('select').forEach(function(elem) {
          elem.querySelectorAll('option').forEach(function(elemOpt) {
              if (elemOpt.selected)
                  data[elem.name] = elemOpt.value;
          });
      });

      var jsonData = JSON.stringify(data);

      makeRequest('/settings/save?splash=1', jsonData, function(response) {

        makeRequest('/restart');

          // wait some time for server to restart
          setTimeout(function() {
              destroyDialog('diaSplashScreen');
              document.location.reload(true);
          }, 1000);
      });

      event.preventDefault();
  });

  diaSplashScreenSelectFolder.addEventListener('click', function() {
      makeRequest('/select_dir', null, function(response) {
          // verify it was not cancelled
          if (response.length)
              document.querySelector('input[name="extAppsDirectory"]').value = response;
      });
  });

  // remove close button
  diaSplashScreenClose.parentElement.removeChild(diaSplashScreenClose);

  focusElem('splashScreenSave');
</%block>
