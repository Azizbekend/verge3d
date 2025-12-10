'use strict';

const CONTAINER_ID = 'v3d-container';
const SCENE_URL = 'template.gltf';

const preloader = new v3d.SimplePreloader({ container: CONTAINER_ID });
const app = new v3d.App(CONTAINER_ID, null, preloader);

app.loadScene(SCENE_URL, () => {
    app.enableControls();
    app.run();
    runCode();
});

function runCode() {
    // add your code here, e.g. console.log('Hello, World!');
}
