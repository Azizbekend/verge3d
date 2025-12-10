<html>
<head>
  <title>Verge3D Network</title>
  <%include file="head.tpl"/>
  <script src="/manager/js/network.js"></script>
</head>

<%

FILE_CLASSES_MAP = {
    'net-file-icon-archive': ['zip', 'xz', 'tar', 'rar', '7z'],
    'net-file-icon-blender': ['blend'],
    'net-file-icon-code': ['html', 'js', 'css', 'json', 'xml', 'ts'],
    'net-file-icon-font': ['ttf', 'woff', 'woff2'],
    'net-file-icon-gltf': ['gltf', 'glb', 'bin'],
    'net-file-icon-image': ['jpg', 'jpeg', 'hdr', 'png', 'bmp', 'tiff', 'gif', 'ico'],
    'net-file-icon-max': ['max', '3ds'],
    'net-file-icon-maya': ['ma', 'mb'],
    'net-file-icon-sound': ['mp3', 'wav', 'ogg'],
    'net-file-icon-vector': ['svg']
}

import os.path
    
def fileIconClass(file):
    fileClass = 'net-file-icon-other'

    if file['isDir']:
        fileClass = 'net-file-icon-folder'
    else:
        fileExt = os.path.splitext(file['name'])[1].lstrip('.')
        for cl, extlist in FILE_CLASSES_MAP.items():
            for ext in extlist:
                if ext == fileExt:
                    fileClass = cl 
                    break

    return 'net-file-icon ' + fileClass

def humansize(size):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""

    B = float(size)
    KB = float(1024)
    MB = float(KB ** 2)
    GB = float(KB ** 3)
    TB = float(KB ** 4)

    if B < KB:
        return '{0:.0f}'.format(B)
    elif KB <= B < MB:
        return '{0:.1f} K'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.1f} M'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.1f} G'.format(B / GB)
    elif TB <= B:
        return '{0:.1f} T'.format(B / TB)

%>

<body>
  <div class="main-panel">

    <div class="banner banner-network"></div>

    <%include file="search.tpl"/>

    <table class="network-directory">
      <thead>
        <tr>
          <th colspan=2>
            directory
          </th>
          <th>size</th>
          <th>date &amp; time</th>
        </tr>
      </thead>

      <tbody>
        % for file in filesViewInfo:
          <tr class="filterable">
            <td>
              % if file['isDir']:
                <span title="Select entire directory">
                  <input type="checkbox" value="${file['key']}" class="netcheckboxdir" onClick="toggleDir(this)">
                </span>
              % else:
                <span title="Select file">
                  <input type="checkbox" value="${file['key']}" class="netcheckbox">
                </span>
              % endif
            </td>
            <td>
              <span style="margin-left: ${20 * file['indent']}px">
                <span class="${fileIconClass(file)}"></span>
                % if file['isDir']:
                  ${file['name']}/
                % else:
                  <a href="${file['url']}">${file['name']}</a>
                % endif
              </span>
            </td>
            <td>
              ${humansize(file['size'])}
            </td>
            <td>
              ${file['date']}
            </td>
          </tr>
        % endfor

        % if len(filesViewInfo) == 0:
        <tr>
          <td colspan=4>
            <div class="network-empty">
              Your Verge3D Network directory is empty. Try to upload something!
            </div>
          </td>
        </tr>
        % endif
      </tbody>
      <tfoot><tr><td colspan=4>© Soft8Soft 2023</td></tr></tfoot>
    </table>

  </div>

  <%include file="toolbar.tpl"/>
  <%include file="toolbar_network.tpl"/>
  <%include file="dialog_network_download.tpl"/>

</body>
</html>
