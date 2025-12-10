<div class="toolbar-store">
  <a href="javascript:void(0);" class="toolbar-store-icon toolbar-store-icon-all" title="Display all assets"></a>
  <a href="javascript:void(0);" class="toolbar-store-icon toolbar-store-icon-demos" title="Display demos"></a>
  <a href="javascript:void(0);" class="toolbar-store-icon toolbar-store-icon-tutorials" title="Display tutorials"></a>
  <a href="javascript:void(0);" class="toolbar-store-icon toolbar-store-icon-libraries" title="Display libraries"></a>
</div>

<script>

function setActiveToolbarStore(className) {
    document.querySelectorAll('.toolbar-store a').forEach(function(icon) {
        if (icon.classList.contains(className))
            icon.classList.add('toolbar-icon-active');
        else
            icon.classList.remove('toolbar-icon-active');
    });
}

function switchStoreFilter(e) {
    const icon = e.target;
    const iconClass = icon.classList[1];

    setActiveToolbarStore(iconClass);

    const items = document.getElementsByClassName('filterable');

    const matchedType = iconClass.replace('toolbar-store-icon-', '');

    for (let i = 0; i < items.length; i++) {
        const item = items[i];

        const filterResult = item.querySelector('.asset-store-item-type');

        if (matchedType == 'all' || (filterResult && filterResult.textContent.plural().toLowerCase() == matchedType)) {

            item.classList.remove('hidden');

        } else {

            item.classList.add('hidden');

        }

    }
}

document.querySelectorAll('.toolbar-store a').forEach(function(icon) {
    icon.addEventListener('click', switchStoreFilter);
});

setActiveToolbarStore('toolbar-store-icon-all');

document.body.classList.add('always-scroll');

</script>

