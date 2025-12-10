<%namespace name="utils" file="utils.tpl"/>

<html>
<head>
  <title>Verge3D App Manager</title>
  <%include file="head.tpl"/>
</head>

<body>
  <div class="main-panel">

    <div class="banner"></div>

    <table class="network-directory">
      <thead>
        <tr>
          <th colspan=2>
            App Manager Error
          </th>
        </tr>
      </thead>

      <tbody>
        <tr>
          <td colspan=2>
            <div class="error-message">
              ${utils.parseError(message)}
            </div>
          </td>
        </tr>
      </tbody>
      <tfoot><tr><td colspan=2>© Soft8Soft 2023</td></tr></tfoot>
    </table>

  </div>

  <%include file="toolbar_error.tpl"/>

</body>
</html>
