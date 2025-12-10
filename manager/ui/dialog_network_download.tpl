<%inherit file="dialog.tpl"/>

<%block name="blkDialogId">diaNetworkDownload</%block>

<%block name="blkDialogHeader">
  Downloading from Verge3D Network...
</%block>

<%!
  import os
%>

<%block name="blkDialogContent">
  <div class="dialog-right-content-short" id="downloadConfirmCont">
    <div class="dialog-text dialog-text-delete">
      You are about to download <span id="numFilesToDownload">0</span> files.
      Beware: this will overwrite them locally!
    </div>
    <div class="two-buttons">
      <div><button class="button delete-button" id="downloadYes">Proceed</button></div>
      <div><button class="button" id="downloadNo">Cancel</button></div>
    </div>
  </div>

  <div id="downloadPercentageCont">
    <div class="spinner-preloader-cont">
      <div class="spinner-preloader-percentage" id='downloadPercentage'>0%</div>
      <div class="spinner-preloader"></div>
    </div>
    <button class="button" id="cancelDownloading">Cancel</button>
  </div>

  <div class="dialog-right-content-short" id="downloadErrorCont">
    <div class="dialog-text">
      No files selected for download!
    </div>
    <button id="downloadOk" class="button">Got it!</button>
  </div>
</%block>

<%block name="blkDialogScript">
    ${parent.blkDialogScript()}
    var downloadKeys = [];

    function switchDiaNetworkMode(mode) {
        if (mode == 'confirm') {
            downloadConfirmCont.style.display = 'block';
            downloadPercentageCont.style.display = 'none';
            downloadErrorCont.style.display = 'none';
        } else if (mode == 'preloader') {
            downloadConfirmCont.style.display = 'none';
            downloadPercentageCont.style.display = 'block';
            downloadErrorCont.style.display = 'none';
        } else if (mode == 'error') {
            downloadConfirmCont.style.display = 'none';
            downloadPercentageCont.style.display = 'none';
            downloadErrorCont.style.display = 'block';
        }
    }

    diaNetworkDownload.addEventListener('dialogopen', function() {
        downloadKeys.length = 0;
        var checkers = document.getElementsByClassName('netcheckbox');

        for (var i = 0; i < checkers.length; i++) {
            var checker = checkers[i];

            if (checker.checked)
                downloadKeys.push(checker.value);
        }

        if (downloadKeys.length) {
            switchDiaNetworkMode('confirm');
            focusElem('downloadNo');
            document.getElementById('numFilesToDownload').innerText = downloadKeys.length;
        } else {
            switchDiaNetworkMode('error');
        }
    });

    downloadYes.addEventListener('click', function() {
        var url = '/storage/net/?req=download&key=' + downloadKeys.join('&key=');

        var percentageTimer = null;

        makeRequest(url, null, function(response) {
            clearTimeout(percentageTimer);

            closeDialog('diaNetworkDownload');

            var dia = appendDialog(response);
            openDialog(dia);
        });

        function percentageListener() {
            var percentage = document.getElementById('downloadPercentage');
            percentage.innerHTML = Math.round(this.responseText || 0) + '%';

            percentageTimer = window.setTimeout(function() {
                var req = new XMLHttpRequest();
                req.addEventListener('load', percentageListener);
                req.open('GET', '/storage/net/?req=progress');
                req.send();
            }, 300)
        }
    
        percentageListener();
        switchDiaNetworkMode('preloader');

    });

    downloadNo.addEventListener('click', function() {
        closeDialog('diaNetworkDownload');
    });

    downloadOk.addEventListener('click', function() {
        closeDialog('diaNetworkDownload');
    });

    cancelDownloading.addEventListener('click', function(event) {
        makeRequest('/storage/net/?req=cancel', null, null);
    });

</%block>
