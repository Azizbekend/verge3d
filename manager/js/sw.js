const CACHE = 'v3d-app-manager-cache-v1';

const CONN_ERR_URL = '/connection_error';

const CONN_ERR_ASSETS = [
    '/manager/css/fonts.css',
    '/manager/css/common.css',
    '/manager/js/common.js',
    '/manager/js/qrcode.js',
    '/manager/img/banner.svg',
    '/manager/fonts/Rubik-Regular.woff',
    '/manager/img/favicons/manifest.json',
    '/manager/img/favicons/favicon-32x32.png'
];

// low case
const IGNORE_RESOURCES = [
    '.js',
    '.css',
    '.png',
    '.jpg',
    '.jpeg',
    '.svg',
    '.webp',
    '.hdr',
    '.json',
    '.xml',
    '.gltf',
    '.bin',
    '.glb',
    '.mp3',
    '.wav',
    '.mp4',
    '.webm',
    '.xz',
    '.ktx2',
    '.wasm',
    '.ttf',
    '.woff',
    '.csv',
    '.plug', // plugins
    '.block', // plugins
    '/settings/do_show_splash',
    '/ping',
    '/stop',
    '/restart',
    '/reset',
    '/select_dir',
    '/get_preview_dir'
];

self.addEventListener('install', (event) => {

    event.waitUntil(caches.open(CACHE).then(cache => {
        return cache.addAll([CONN_ERR_URL].concat(CONN_ERR_ASSETS));
    }));

    console.log('Verge3D App Manager worker installed');

});

async function handleDisconnection(request) {

    try {

        const responseFromNetwork = await fetch(request);
        return responseFromNetwork;

    } catch (error) {

        let fallbackResponse = await caches.match(request);
        if (fallbackResponse) {
            return fallbackResponse;
        }

        fallbackResponse = await caches.match(CONN_ERR_URL);
        if (fallbackResponse) {
            return new Response(fallbackResponse.body, {
                status: 404,
                headers: fallbackResponse.headers
            });
        }

        return new Response('Network error happened', {
            status: 408,
            headers: { 'Content-Type': 'text/plain' }
        });

    }

}

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    const pathname = url.pathname;

    if (url.port != '8668')
        return;

    // do not ignore assets required for error page
    for (let i = 0; i < CONN_ERR_ASSETS.length; i++) {
        if (pathname == CONN_ERR_ASSETS[i]) {
            event.respondWith(handleDisconnection(event.request));
            return;
        }
    }

    for (let i = 0; i < IGNORE_RESOURCES.length; i++)
        if (pathname.toLowerCase().endsWith(IGNORE_RESOURCES[i]))
            return;

    event.respondWith(handleDisconnection(event.request));
});
