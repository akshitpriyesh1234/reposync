Frontend artifact
=================


UI architecture
---------------


The Voxel51 webapp is integarted with the python backend, adding data to this front-end is done by using the Python APIs provided by voxel fiftyone.
The inteded way of adding additional features to the frontend is by creating a plugin that the host application can dynamically load.

.. image:: /figs/ui-arch.png
   :alt: ui-arch

The fiftyone webapp and individual plugins are bundled separately, a script tag is created and attached to the DOM tree for loading the plugin on demand.

::

   const script = document.createElement("script");
   script.type = "application/javascript";
   script.src = url;
   script.async = true;
   document.head.prepend(script);

.. note::
   Plugin hot reload: Needless to say, this approach prevents any hot reload or HMR from working, hence,
   :code:`vite build --watch`
   can be used as a slower workaround.

Getting plugins specific data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each dataset can have plugins data that can be retrived using the usePluginSettings hook.

::

   import * as fop from "@fiftyone/plugins";
   const { data_key } = fop.usePluginSettings("plugin-name", {data_key: null});


Using Fiftyone app state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Recoil atoms defined in the host app can be used in plugins as long
:code:`vite-plugin-external` is configured with :code:`@fiftyone/state` and the recoil instance.

vite.config.ts
::

   plugins: [
   react(),
   isPluginBuild ? viteExternalsPlugin({
      react: 'React',
      'react-dom': 'ReactDOM',
      'recoil': 'recoil',
      '@fiftyone/state': '__fos__'
   }) : undefined
   ],


.. note::

    Fiftyone plugins is documented extensively `here <https://docs.voxel51.com/plugins/api/fiftyone.plugins.html>`_

Addons
~~~~~~

Plugins can export objects to the global scope, thus existing plugins can be extended with new features.

For this, the entry point to plugin node module should export Objects to the Global scope.

::

   document['TemplateContainer'] = TemplateContainer;

thus, enabling other plugins to get the reference to these Objects.
::

   if(document['TemplateContainer']) const TemplateContainer = document['TemplateContainer'];

.. note::

    * Packages that can cause inconsistencies and dependency conflicts should be avoided in these plugins.
    * It is preferable to externalize dependencies to keep the bundle size manageable.
