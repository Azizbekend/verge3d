(function () {

    var DIALOG_ICONS_MAP = {
        'diaNewApp': 'toolbar-icon-new-app',
        'diaAppTemplates': 'toolbar-icon-templates',
        'diaSettings': 'toolbar-icon-settings',
        'diaLicense': 'toolbar-icon-license',
        'diaLicenseKey': 'toolbar-icon-license',
        'diaHelp': 'toolbar-icon-help',
        'diaAbout': 'toolbar-icon-about',
    }

    var VERSION_URL = 'https://www.soft8soft.com/verge3d-' + appManager.package.toLowerCase() + '-latest.json?' + Date.now();

    function parseVersion(text) {
        version = [0, 0, 0, 0];

        dotSplit = text.split('.');
        preSplit = dotSplit[2].split('-pre');

        version[0] = +dotSplit[0];
        version[1] = +dotSplit[1];
        version[2] = +preSplit[0];

        if (preSplit.length > 1)
            version[3] = +preSplit[1];
        else
            // stable release is more recent than pre release
            version[3] = 1000;

        return version;
    }

    /**
     *  1 v1 > v2
     *  0 v1 = v2
     * -1 v1 < v2
     */
    function compareVersions(v1, v2) {

        for (var i = 0; i < 4; i++) {
            var sign = Math.sign(v1[i] - v2[i]);
            if (sign)
                return sign;
        }

        return 0;
    }

    function checkForUpdates() {
        function reqListener() {
            if (this.status == 200) {
                var info = JSON.parse(this.responseText);

                var v1 = parseVersion(info.version);
                var v2 = parseVersion(appManager.version);

                if (compareVersions(v1, v2) > 0) {
                    newVersionAvailDot.classList.remove('hidden');

                    newVersionPostfix.classList.remove('hidden');
                    newVersionPostfix.innerHTML = '(<a href="' + info.release_notes +
                        '" target="_blank" >update ' + info.version + ' available!)</a>'
                }
            }
        }

        var req = new XMLHttpRequest();
        req.addEventListener('load', reqListener);
        req.open('GET', VERSION_URL);
        req.send();
    }

    function updateFilter() {

        // escaping special characters
        const regExp = new RegExp(filterInput.value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');

        var items = document.getElementsByClassName('filterable');

        for (var i = 0; i < items.length; i++) {
            var item = items[i];

            var filterResults = item.textContent.match(regExp);

            if (filterResults && filterResults.length > 0) {

                item.classList.remove('hidden');

            } else {

                item.classList.add('hidden');

            }

        }

    }

    function applyTableHeightHack() {
        var bannerHeight = document.querySelector('.main-panel div.banner').offsetHeight;
        var tableHeight = document.querySelector('.main-panel table').offsetHeight;

        document.querySelectorAll('.main-panel table tbody').forEach(function (tableBody) {

            var rowHeight = window.innerHeight - bannerHeight - tableHeight;

            if (rowHeight > 0) {
                tableBody.insertAdjacentHTML('beforeend', '<tr style="height:' + rowHeight + 'px"></tr>');
            }

        });
    }

    function init() {

        window.addEventListener('load', function () {
            if (appManager.checkForUpdates)
                checkForUpdates();

            const filterInput = document.getElementById('filterInput');
            if (filterInput)
                filterInput.addEventListener('input', function () {
                    updateFilter();
                });

            handleDialogEscape('diaAbout', false);
            handleDialogEscape('diaAppTemplates', true);
            handleDialogEscape('diaDeleteConfirm', true);
            handleDialogEscape('diaDelete', true);
            handleDialogEscape('diaError', true);
            handleDialogEscape('diaHelp', false);
            handleDialogEscape('diaLicense', false);
            handleDialogEscape('diaLicenseKey', false);
            handleDialogEscape('diaLicenseKeyDone', true);
            handleDialogEscape('diaNetworkDelete', true);
            handleDialogEscape('diaNetworkError', true);
            handleDialogEscape('diaNewApp', false);
            handleDialogEscape('diaNewAppCreated', true);
            handleDialogEscape('diaPublishing', false);
            handleDialogEscape('diaPublished', true);
            handleDialogEscape('diaQrCode', false);
            handleDialogEscape('diaSettings', true);
            handleDialogEscape('diaUpdate', true);
            handleDialogEscape('diaUpdateDone', true);

            applyTableHeightHack();

            makeRequest('/settings/do_show_splash', null, function (response) {
                if (response == '1')
                    makeRequest('/settings/splash_screen', null, function (response) {
                        var dia = appendDialog(response);
                        openDialog(dia);
                    });
            });

        });

        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/manager/js/sw.js', { scope: '/' }).then(() => {
                navigator.serviceWorker.ready.then((worker) => {
                    //worker.sync.register('syncdata');
                });
            }).catch((err) => {
                console.log(err)
            });
        }
    }

    window.capitalizeFirstLetters = function (str) {
        return str.toLowerCase().replace(/^\w|\s\w/g, function (letter) {
            return letter.toUpperCase();
        });
    }

    window.toggleCollapsible = function (elem) {
        switch (elem.className) {
            case 'app-collapsed':
                elem.className = 'app-expanded';
                elem.parentElement.parentElement.querySelectorAll('.app-icon-cut').forEach(function (cutElem) {
                    cutElem.classList.remove('hidden');
                });
                break;
            case 'app-expanded':
                elem.className = 'app-collapsed';
                elem.parentElement.parentElement.querySelectorAll('.app-icon-cut').forEach(function (cutElem) {
                    cutElem.classList.add('hidden');
                });
                break;
            default:
                break;
        }
    }

    window.focusElem = function (id) {
        setTimeout(function () {
            document.getElementById(id).focus();
        }, 0);
    }

    window.openDialog = function (id) {
        document.getElementById(id).style.display = 'flex';

        setActiveToolbar(DIALOG_ICONS_MAP[id]);

        var event = new Event('dialogopen');
        document.getElementById(id).dispatchEvent(event);
    }

    window.closeDialog = function (id) {
        document.getElementById(id).style.display = 'none';
        setActiveToolbar();

        var event = new Event('dialogclose');
        document.getElementById(id).dispatchEvent(event);
    }

    window.isDialogOpen = function (id) {
        var elem = document.getElementById(id);
        return (elem && elem.style.display !== 'none')
    }

    window.appendDialog = function (code) {
        document.body.insertAdjacentHTML('beforeend', code);
        var dia = document.body.lastElementChild;
        var scriptElem = dia.getElementsByTagName('script')[0];
        if (scriptElem)
            eval(scriptElem.innerHTML);
        else
            console.error('Missing dialog script, check your template for errors!');
        return dia.id;
    }

    window.destroyDialog = function (id) {
        document.getElementById(id).parentNode.removeChild(document.getElementById(id));
        setActiveToolbar();
    }

    window.makeRequest = function (url, data, callback) {
        var req = new XMLHttpRequest();

        if (data) {
            req.open('POST', url);
            req.send(data);
        } else {
            req.open('GET', url);
            req.send();
        }

        if (callback) {
            req.addEventListener('load', function () {
                callback(this.responseText);
            });
        }
    }

    window.openFile = function (url) {
        var req = new XMLHttpRequest();
        req.open('GET', '/open/?filepath=' + url);
        req.send();
    }

    window.deleteApp = function (app) {
        makeRequest('/delete_confirm/?app=' + app, null, function (response) {
            var dia = appendDialog(response);
            openDialog(dia);
        });
    }

    window.updateApp = function (app, appIconElem) {

        // save to change class without reloading the page
        appManager.updatedAppIconElem = appIconElem || null;

        makeRequest('/update_app_info/?app=' + app, null, function (response) {
            var dia = appendDialog(response);
            openDialog(dia);
        });
    }

    window.publishApp = function (app, isZip) {

        var percentageTimer = null;

        function reqListener(response) {
            closeDialog('diaPublishing');
            clearTimeout(percentageTimer);

            var dia = appendDialog(response);
            openDialog(dia);
        }

        function percentageListener() {
            var percentage = document.getElementById('publishingPercentage');
            percentage.innerHTML = Math.round(this.responseText || 0) + '%';

            percentageTimer = window.setTimeout(function () {
                var req = new XMLHttpRequest();
                req.addEventListener('load', percentageListener);
                req.open('GET', '/storage/net/?req=progress');
                req.send();
            }, 300)
        }

        openDialog('diaPublishing')
        percentageListener();

        makeRequest('/storage/net/?req=upload&app=' + app + '&zip=' + (isZip ? '1' : '0'), null, reqListener);
    }

    window.createQRCode = function (url) {
        var typeNumber = 0;
        var errorCorrectionLevel = 'L';
        var qr = qrcode(typeNumber, errorCorrectionLevel);
        qr.addData(url);
        qr.make();
        return qr.createDataURL(7);
    }

    window.createNativeApp = function (app) {
        openDialog('diaCreateNativeApp');
    }

    window.createScorm = function (app) {
        openDialog('diaCreateScorm');
    }

    window.openSettingsDialog = function () {
        makeRequest('/settings/', null, function (response) {
            var dia = appendDialog(response);
            openDialog(dia);
        });
    }

    window.copyTextArea = function (id) {
        var elem = document.getElementById(id);
        elem.select();
        document.execCommand('copy');
        elem.selectionStart = 0;
        elem.selectionEnd = 0;
    }

    /**
     * Activates home/network icon if class not specified
     */
    window.setActiveToolbar = function (className) {

        if (!className) {
            if (window.location.pathname.indexOf('/storage/net') > -1)
                className = 'toolbar-icon-network';
            else if (window.location.pathname.indexOf('/store') > -1)
                className = 'toolbar-icon-store';
            else if (window.location.pathname.indexOf('/manage') > -1)
                className = 'reserved-empty-class';
            else
                className = 'toolbar-icon-home';
        }

        document.querySelectorAll('.toolbar a').forEach(function (icon) {
            if (icon.classList.contains(className))
                icon.classList.add('toolbar-icon-active');
            else
                icon.classList.remove('toolbar-icon-active');
        });
    }

    window.handleDialogEscape = function (dialogID, doDestroy) {
        document.addEventListener('keydown', function (event) {
            if (event.keyCode == 27 && isDialogOpen(dialogID)) {
                if (doDestroy)
                    destroyDialog(dialogID);
                else
                    closeDialog(dialogID);
            }
        });
    }

    window.downloadDemo = function (demo) {
        function reqListener(response) {
            closeDialog('diaStoreDownload');
            var dia = appendDialog(response);
            openDialog(dia);
        }

        openDialog('diaStoreDownload')
        makeRequest('/store/?req=download&demo=' + demo, null, reqListener);
    }

    /**
     * Inspired by https://stackoverflow.com/questions/27194359/javascript-pluralize-an-english-string
     */
    String.prototype.plural = function (revert) {

        const plural = {
            '(quiz)$': "$1zes",
            '^(ox)$': "$1en",
            '([m|l])ouse$': "$1ice",
            '(matr|vert|ind)ix|ex$': "$1ices",
            '(x|ch|ss|sh)$': "$1es",
            '([^aeiouy]|qu)y$': "$1ies",
            '(hive)$': "$1s",
            '(?:([^f])fe|([lr])f)$': "$1$2ves",
            '(shea|lea|loa|thie)f$': "$1ves",
            'sis$': "ses",
            '([ti])um$': "$1a",
            '(tomat|potat|ech|her|vet)o$': "$1oes",
            '(bu)s$': "$1ses",
            '(alias)$': "$1es",
            '(octop)us$': "$1i",
            '(ax|test)is$': "$1es",
            '(us)$': "$1es",
            '([^s]+)$': "$1s"
        };

        const singular = {
            '(quiz)zes$': "$1",
            '(matr)ices$': "$1ix",
            '(vert|ind)ices$': "$1ex",
            '^(ox)en$': "$1",
            '(alias)es$': "$1",
            '(octop|vir)i$': "$1us",
            '(cris|ax|test)es$': "$1is",
            '(shoe)s$': "$1",
            '(o)es$': "$1",
            '(bus)es$': "$1",
            '([m|l])ice$': "$1ouse",
            '(x|ch|ss|sh)es$': "$1",
            '(m)ovies$': "$1ovie",
            '(s)eries$': "$1eries",
            '([^aeiouy]|qu)ies$': "$1y",
            '([lr])ves$': "$1f",
            '(tive)s$': "$1",
            '(hive)s$': "$1",
            '(li|wi|kni)ves$': "$1fe",
            '(shea|loa|lea|thie)ves$': "$1f",
            '(^analy)ses$': "$1sis",
            '((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$': "$1$2sis",
            '([ti])a$': "$1um",
            '(n)ews$': "$1ews",
            '(h|bl)ouses$': "$1ouse",
            '(corpse)s$': "$1",
            '(us)es$': "$1",
            's$': ""
        };

        const irregular = {
            'move': 'moves',
            'foot': 'feet',
            'goose': 'geese',
            'sex': 'sexes',
            'child': 'children',
            'man': 'men',
            'tooth': 'teeth',
            'person': 'people'
        };

        const uncountable = [
            'sheep',
            'fish',
            'deer',
            'moose',
            'series',
            'species',
            'money',
            'rice',
            'information',
            'equipment'
        ];

        // save some time in the case that singular and plural are the same
        if (uncountable.indexOf(this.toLowerCase()) >= 0)
            return this;

        // check for irregular forms
        for (let word in irregular) {

            let pattern, replace;

            if (revert) {
                pattern = new RegExp(irregular[word] + '$', 'i');
                replace = word;
            } else {
                pattern = new RegExp(word + '$', 'i');
                replace = irregular[word];
            }

            if (pattern.test(this))
                return this.replace(pattern, replace);
        }

        let array;
        if (revert)
            array = singular;
        else
            array = plural;

        // check for matches using regular expressions
        for (reg in array) {

            const pattern = new RegExp(reg, 'i');

            if (pattern.test(this))
                return this.replace(pattern, array[reg]);
        }

        return this;
    }

    init();

})(); // end of closure



