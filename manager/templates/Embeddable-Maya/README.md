# Verge3D React.js Application Example

1) Create a React.js application via the Create React App utility:

    ```
    npx create-react-app react-app-example
    ```

2) Enter the `react-app-example` directory and install the `verge3d` npm package:

    ```
    cd react-app-example
    npm i verge3d
    ```

3) Copy the following files from Verge3D distribution into your app:

    * the contents of Verge3D's `manager/templates/Embeddable/public` directory
    to `react-app-example/public`

    * the contents of Verge3D's `manager/templates/Embeddable/src` directory to
    `react-app-example/src`

    or use the commands below:

    <u>Linux/macOS</u>

    *Change `V3D_PATH` and `MY_PATH` to where the Verge3D distribution and your
    React application are located respectively.*

    ```sh
    V3D_PATH=path/to/v3d/distribution
    MY_PATH=path/to/my/react/app
    cp -r $V3D_PATH/manager/templates/Embeddable/public/* $MY_PATH/public/
    cp -r $V3D_PATH/manager/templates/Embeddable/src/* $MY_PATH/src/
    ```

    <u>Windows(PowerShell)</u>

    *Change `V3D_PATH` and `MY_PATH` to where the Verge3D distribution and your
    React application are located respectively.*

    ```powershell
    $V3D_PATH = "path\to\v3d\distribution"
    $MY_PATH = "path\to\my\react\app"
    Copy-Item -Path "$V3D_PATH\manager\templates\Embeddable\public\*" -Destination "$MY_PATH\public" -Recurse
    Copy-Item -Path "$V3D_PATH\manager\templates\Embeddable\src\*" -Destination "$MY_PATH\src" -Recurse
    ```

4) Create a file called `react-app-example/src/V3DApp.js` with the following content:

    ```js
    import React from 'react';

    import { createApp } from './v3dApp/app';
    import './v3dApp/app.css';

    class V3DApp extends React.Component {
      #app = null;
      #PL = null;

      #uuid = window.crypto.randomUUID();
      #containerId = `v3d-container-${this.#uuid}`;
      #fsButtonId = `fullscreen-button-${this.#uuid}`;
      #sceneURL = 'v3dApp/app.gltf';

      async loadApp() {
        ({ app: this.#app, PL: this.#PL } = await createApp({
          containerId: this.#containerId,
          fsButtonId: this.#fsButtonId,
          sceneURL: this.#sceneURL,
        }));
      }

      disposeApp() {
        this.#app?.dispose();
        this.#app = null;

        // dispose Puzzles' visual logic
        this.#PL?.dispose();
        this.#PL = null;
      }

      reloadApp() {
        this.disposeApp();
        this.loadApp();
      }

      componentDidMount() {
        this.loadApp();
      }

      componentWillUnmount() {
        this.disposeApp();
      }

      render() {
        return <div id={this.#containerId}>
          <div
            id={this.#fsButtonId}
            className="fullscreen-button fullscreen-open"
            title="Toggle fullscreen mode"
          ></div>
        </div>;
      }
    }

    export default V3DApp;
    ```

5) Replace the contents of `react-app-example/src/index.js` with the following code:

    ```js
    import React from 'react';
    import ReactDOM from 'react-dom/client';

    import V3DApp from './V3DApp';

    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(<V3DApp/>);
    ```

6) Run the development server by executing the following command in the
`react-app-example` directory:

    ```
    npm start
    ```

    By default the application now should be available at http://localhost:3000/.


# Verge3D Vue.js Application Example

1) Create a Vue.js application via Vite:

    ```
    npm create vite@latest vue-app-example -- --template vue
    ```

2) Enter the `vue-app-example` directory and install the `verge3d` npm package:

    ```
    cd vue-app-example
    npm i verge3d
    ```

3) Copy the following files from Verge3D distribution into your app:

    * the contents of Verge3D's `manager/templates/Embeddable/public` directory
    to `vue-app-example/public`

    * the contents of Verge3D's `manager/templates/Embeddable/src` directory to
    `vue-app-example/src`

    or use the commands below:

    <u>Linux/macOS</u>

    *Change `V3D_PATH` and `MY_PATH` to where the Verge3D distribution and your
    Vue application are located respectively.*

    ```sh
    V3D_PATH=path/to/v3d/distribution
    MY_PATH=path/to/my/vue/app
    cp -r $V3D_PATH/manager/templates/Embeddable/public/* $MY_PATH/public/
    cp -r $V3D_PATH/manager/templates/Embeddable/src/* $MY_PATH/src/
    ```

    <u>Windows(PowerShell)</u>

    *Change `V3D_PATH` and `MY_PATH` to where the Verge3D distribution and your
    Vue application are located respectively.*

    ```powershell
    $V3D_PATH = "path\to\v3d\distribution"
    $MY_PATH = "path\to\my\vue\app"
    Copy-Item -Path "$V3D_PATH\manager\templates\Embeddable\public\*" -Destination "$MY_PATH\public" -Recurse
    Copy-Item -Path "$V3D_PATH\manager\templates\Embeddable\src\*" -Destination "$MY_PATH\src" -Recurse
    ```

4) Create a file `vue-app-example/src/components/V3DApp.vue` containing the following
code:

    <u>Vue 3</u>

    ```js
    <template>
      <div :id="containerId">
        <div
          :id="fsButtonId"
          class="fullscreen-button fullscreen-open"
          title="Toggle fullscreen mode"
        ></div>
      </div>
    </template>

    <script>
    import { createApp } from '../v3dApp/app';

    export default {
      name: 'V3DApp',

      created() {
        this.app = null;
        this.PL = null,

        this.uuid = window.crypto.randomUUID();
        this.containerId = `v3d-container-${this.uuid}`;
        this.fsButtonId = `fullscreen-button-${this.uuid}`;
        this.sceneURL = 'v3dApp/app.gltf';

        this.loadApp = async function() {
          ({ app: this.app, PL: this.PL } = await createApp({
            containerId: this.containerId,
            fsButtonId: this.fsButtonId,
            sceneURL: this.sceneURL,
          }));
        }

        this.disposeApp = function() {
          this.app?.dispose();
          this.app = null;

          // dispose Puzzles' visual logic
          this.PL?.dispose();
          this.PL = null;
        }

        this.reloadApp = function() {
          this.disposeApp();
          this.loadApp();
        }
      },

      mounted() {
        this.loadApp();
      },

      beforeUnmount() {
        this.disposeApp();
      },
    }
    </script>

    <style>
    @import '../v3dApp/app.css';
    </style>
    ```

    <u>Vue 2</u>

    ```js
    <template>
      <div :id="containerId">
        <div
          :id="fsButtonId"
          class="fullscreen-button fullscreen-open"
          title="Toggle fullscreen mode"
        ></div>
      </div>
    </template>

    <script>
    import { createApp } from '../v3dApp/app';

    export default {
      name: 'V3DApp',

      created() {
        this.app = null;
        this.PL = null,

        this.uuid = window.crypto.randomUUID();
        this.containerId = `v3d-container-${this.uuid}`;
        this.fsButtonId = `fullscreen-button-${this.uuid}`;
        this.sceneURL = 'v3dApp/app.gltf';

        this.loadApp = async function() {
          ({ app: this.app, PL: this.PL } = await createApp({
            containerId: this.containerId,
            fsButtonId: this.fsButtonId,
            sceneURL: this.sceneURL,
          }));
        }

        this.disposeApp = function() {
          this.app?.dispose();
          this.app = null;

          // dispose Puzzles' visual logic
          this.PL?.dispose();
          this.PL = null;
        }

        this.reloadApp = function() {
          this.disposeApp();
          this.loadApp();
        }
      },

      mounted() {
        this.loadApp();
      },

      beforeDestroy() {
        this.disposeApp();
      },
    }
    </script>

    <style>
    @import '../v3dApp/app.css';
    </style>
    ```

5) Replace the contents of `vue-app-example/src/App.vue` with the following code:

    ```js
    <template>
      <V3DApp></V3DApp>
    </template>

    <script>
    import V3DApp from './components/V3DApp.vue';

    export default {
      name: 'App',
      components: {
        V3DApp,
      },
    }
    </script>
    ```

6) Run the development server by executing the following command in the
`vue-app-example` directory:

    ```
    npm run dev
    ```

    By default the application now should be available at http://localhost:5173/.
