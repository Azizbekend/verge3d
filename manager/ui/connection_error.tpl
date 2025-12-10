<%namespace name="utils" file="utils.tpl"/>

<html>
<head>
  <title>Verge3D App Manager</title>
  <%include file="head.tpl"/>

  <script>
    // NOTE: disable slow refresh on Windows
    if (['Win32', 'Win64', 'Windows', 'WinCE'].indexOf(window.navigator.platform) == -1)
      setTimeout(function(){ location.reload(); }, 5000);
  </script>
</head>

<body>
  <div class="main-panel">

    <div class="banner"></div>

    <table class="network-directory">
      <thead>
        <tr>
          <th colspan=2>
            App Manager Connection Error
          </th>
        </tr>
      </thead>

      <tbody>
        <tr>
          <td colspan=2>
            <div class="error-message">
              <p>Could not connect to the local server!</p>
              <p>Please launch App Manager from the modelling suite, or via desktop icon / Start menu.</p>
            </div>
          </td>
        </tr>
      </tbody>
      <tfoot><tr><td colspan=2>© Soft8Soft 2023</td></tr></tfoot>
    </table>

  </div>

</body>
</html>
