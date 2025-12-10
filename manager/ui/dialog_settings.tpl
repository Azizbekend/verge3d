<%inherit file="dialog.tpl"/>

<%block name="blkDialogId">diaSettings</%block>

<%block name="blkDialogHeader">
  App Manager Settings
</%block>

<%block name="blkDialogContent">
  <div class="settings-left-panel">
    <div class="settings-left-item settings-left-item-active" id="settingsItemGeneral">General</div>
    <div class="settings-left-item" id="settingsItemUI">UI</div>
    <div class="settings-left-item" id="settingsItemTemplates">Templates</div>
    <div class="settings-left-item" id="settingsItemPuzzles">Puzzles</div>
  </div>

  <form action="" id="settingsForm" class="settings-form">
    <div class="settings-general">
      <div class="dialog-text settings-text">
        Applications Folder:
        <input type="text" name="extAppsDirectory" value="${settings['extAppsDirectory']}" class="dialog-wide-input">
        <div id="diaSettingsSelectFolder" class="settings-select-folder"></div>
      </div>

      <div class="dialog-text settings-text">
        <input type="checkbox" name="checkForUpdates" value="" ${'checked' if settings['checkForUpdates'] else ''} class="settings-checker">Notify about Verge3D updates
      </div>
      <div class="dialog-text settings-text">
        <input type="checkbox" name="uploadSources" value="" ${'checked' if settings['uploadSources'] else ''} class="settings-checker">Upload app sources (models, puzzles, backups)
      </div>
      <div class="dialog-text settings-text">
        Verge3D Network cache age (minutes)
        <input type="number" name="cacheMaxAge" value="${settings['cacheMaxAge']}" min=0 class="dialog-num-input">
      </div>
      <div class="dialog-text settings-text">
        <input type="checkbox" name="externalInterface" value="" ${'checked' if settings['externalInterface'] else ''} class="settings-checker">Enable external server interface
      </div>
      % if 'externalAddress' in settings:
        <div class="dialog-text settings-text">
          <div>Local network address (copy and paste it to address bar):</div>
          <input type="text" name="externalAddress" value="${settings['externalAddress']}" class="external-address">
        </div>
      % else:
        <div style="height: 60px"></div>
      % endif
      <div style="height: 48px"></div>
      <hr>
    </div>

    <div class="settings-ui hidden">
      <div style="dialog-text settings-text">
        Select Theme:
        <select name="theme" class="settings-theme">
          <option value="light" ${'selected' if settings['theme'] == 'light' else ''}>Light</option>
          <option value="dark" ${'selected' if settings['theme'] == 'dark' else ''}>Dark</option>
        </select>
      </div>
      <div class="dialog-text settings-text">
        User Manual URL:
        <input type="text" name="manualURL" value="${settings['manualURL']}" class="dialog-wider-input">
      </div>
      <div style="height: 185px"></div>
      <hr>
    </div>

    <div class="hidden" id="appTemplatesCont">
      % for template in settings['appTemplates']:
        <div class="app-template-item">
          <input type="text" name="name" value="${template['name']}" class="">
          <textarea name="description" cols="40" rows="2" class="app-template-description-area">${template['description']}</textarea>
          <a class="app-template-delete"></a>
        </div>
      % endfor
      <div class="app-template-item app-template-item-last">
        Click the plus icon to add a new template
        <a class="app-template-new" id="appTemplateNew" title="Append new application template"></a>
      </div>
    </div>

    <div class="settings-puzzles hidden">
      <div class="dialog-text settings-text">
        <input type="checkbox" name="enablePerformanceMode" value="" ${'checked' if settings['enablePerformanceMode'] else ''} class="settings-checker">Enable Performance Mode
      </div>
      <div style="height: 234px"></div>
      <hr>
    </div>

    <input type="submit" id="settingsSave" value="Apply Changes" class="button settings-button">
  </form>
</%block>

<%block name="blkDialogScript">
    diaSettingsClose.addEventListener('click', function() {
        destroyDialog('diaSettings');
    });

    const settingsElems = document.querySelectorAll('.settings-left-panel > div');
    const settingsFormParts = document.querySelectorAll('.settings-form > div');

    settingsElems.forEach(elem => {
        elem.addEventListener('click', event => {
            for (let i = 0; i < settingsElems.length; i++) {
                if (settingsElems[i] == event.target) {
                    settingsElems[i].classList.add('settings-left-item-active');
                    settingsFormParts[i].classList.remove('hidden');
                } else {
                    settingsElems[i].classList.remove('settings-left-item-active');
                    settingsFormParts[i].classList.add('hidden');
                }
            }
        });
    });

    settingsForm.addEventListener('submit', function(event) {

        const data = {};
        const appTemplates = [];

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

        settingsForm.querySelectorAll('div.app-template-item').forEach(function(elem) {
            const appTemplateItem = elem.querySelector('[name="name"]');
            if (appTemplateItem && appTemplateItem.value) {
                appTemplates.push({
                    'name': appTemplateItem.value,
                    'description': elem.querySelector('[name="description"]').value
                });
            }
        });
        data['appTemplates'] = appTemplates;

        settingsForm.querySelectorAll('select').forEach(function(elem) {
            elem.querySelectorAll('option').forEach(function(elemOpt) {
                if (elemOpt.selected)
                    data[elem.name] = elemOpt.value;
            });
        });

        const jsonData = JSON.stringify(data);

        makeRequest('/settings/save', jsonData, function(response) {

          makeRequest('/restart');

            // wait some time for server to restart
            setTimeout(function() {
                destroyDialog('diaSettings');
                document.location.reload(true);
            }, 1000);
        });

        event.preventDefault();
    });

    diaSettingsSelectFolder.addEventListener('click', function() {
        makeRequest('/select_dir', null, function(response) {
            // verify it was not cancelled
            if (response.length)
                document.querySelector('input[name="extAppsDirectory"]').value = response;
        });
    });

    appTemplateNew.addEventListener('click', function(event) {
        appTemplatesCont.lastElementChild.insertAdjacentHTML('beforebegin', '\
            <div class="app-template-item">\
              <input type="text" name="name" value="My App Template" class="">\
              <textarea name="description" cols="40" rows="2" class="app-template-description-area">This is my new application template</textarea>\
              <a class="app-template-delete"></a>\
            </div>\
        ');

        event.preventDefault();
    });

    settingsForm.addEventListener('click', function(event) {
        const elem = event.target
        if (elem.classList.contains('app-template-delete')) {
            appTemplatesCont.removeChild(elem.parentElement);
        }
    });

  focusElem('settingsSave');
</%block>
