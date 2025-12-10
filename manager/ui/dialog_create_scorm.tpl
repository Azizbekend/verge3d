<%inherit file="dialog.tpl"/>

<%block name="blkDialogId">diaCreateScorm</%block>

<%block name="blkDialogHeader">
  Create SCORM Course Package
</%block>

<%!
  import os
%>

<%block name="blkDialogContent">
  <form action="" class="TODO" id="createScormForm">

    <input type="hidden" name="app" value="${app['name']}">

    <div class="dialog-text">Course Name:</div>
    <span title="SCORM course name"><input type="text" name="courseName" value="${app['title']}" class="dialog-semi-wide-input"></span>

    <div class="dialog-text">Course ID:</div>
    <span title="SCORM manifest ID in reverse domain name notation"><input type="text" id="courseID" name="courseID" value="com.example.${app['title'].replace(' ', '')}" class="dialog-semi-wide-input"></span>

    <div class="dialog-text">Default Item Title:</div>
    <span title="Title of the default course item"><input type="text" name="defaultItemTitle" value="Run App" class="dialog-semi-wide-input"></span>

    <input type="submit" value="Create Course" class="button">

  </form>

  <div class="spinner-preloader-cont" id="creatingScormPercentageCont">
    <div class="spinner-preloader"></div>
    <div class="dialog-text-creating-native-app">Creating the App...</div>
  </div>

</%block>

<%block name="blkDialogScript">

    function switchScormDiaMode(mode) {
        if (mode == 'form') {
            createScormForm.style.display = 'block';
            creatingScormPercentageCont.style.display = 'none';
        } else {
            createScormForm.style.display = 'none';
            creatingScormPercentageCont.style.display = 'block';
        }
    }
    switchScormDiaMode('form');

    diaCreateScormClose.addEventListener('click', function() {
        switchScormDiaMode('form');
        closeDialog('diaCreateScorm');
    });

    createScormForm.addEventListener('submit', function(event) {
        var scormFormData = new FormData(createScormForm);

        makeRequest('/create_scorm/', scormFormData, function(response) {
            switchScormDiaMode('form');
            closeDialog('diaCreateScorm');

            var dia = appendDialog(response);
            openDialog(dia);
        });

        switchScormDiaMode('preloader');
        event.preventDefault();
    });

</%block>
