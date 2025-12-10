<%inherit file="dialog.tpl"/>

<%block name="blkDialogId">diaAbout</%block>

<%block name="blkDialogHeader">
  About Verge3D
</%block>

<%block name="blkDialogContent">
  <div class="dialog-icon dialog-icon-about"></div>

  <div class="dialog-right-content-long">

    <div class="dialog-about-line">Verge3D ${'Ultimate' if package == 'ALL' else 'for ' + package.title()}</div>
    <div class="dialog-about-line">Version:
      <span class="dialog-about-indent">${licenseInfo['releaseVersion']}</span>
      <span class="red hidden" title="Check out the new cool features right away!" id="newVersionPostfix"></span>
    </div>
    <div class="dialog-about-line">
      <span>Follow Us:</span>
      <span class="dialog-about-indent">
        <a href="https://twitter.com/soft8soft" target="_blank" title="Follow us on Twitter!" rel="noreferrer" alt="" class="share-icon share-icon-twitter"></a>
        <a href="https://www.facebook.com/soft8soft/" target="_blank" title="Follow us on Facebook!" rel="noreferrer" alt="" class="share-icon share-icon-facebook"></a>
        <a href="https://www.linkedin.com/company/soft8soft" target="_blank" title="Follow us on LinkedIn!" rel="noreferrer" alt="" class="share-icon share-icon-linkedin"></a>
        <a href="https://www.youtube.com/c/soft8soft" target="_blank" title="Follow us on YouTube!" rel="noreferrer" alt="" class="share-icon share-icon-youtube"></a>
        <a href="https://www.reddit.com/r/RealVerge3D/" target="_blank" title="Follow us on Reddit!" rel="noreferrer" alt="" class="share-icon share-icon-reddit"></a>
        <a href="https://t.me/soft8soft" target="_blank" title="Follow us on Telegram!" rel="noreferrer" alt="" class="share-icon share-icon-telegram"></a>
      </span>
    </div>
    <div class="dialog-about-line">
      Copyright: Soft8Soft 2023. All rights reserved.
    </div>
    <div class="dialog-about-line">
      By using this software you agree with all terms and conditions of <a href="https://www.soft8soft.com/verge3d-end-user-license-agreement/" target="_blank" rel="noreferrer" class="colored-link">Verge3D End-User License Agreement</a>.
    </div>
  </div>

</%block>
