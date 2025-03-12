Artifacts
=========

Our XAI voxel51 plugins have two major deliverables, backend and
frontend.

Frontend artifacts
------------------
**Frontend artifacts** are the UI plugins located in the Jfrog
artifactory
`repo <https://artifactory.boschdevcloud.com/ui/repos/tree/General/xai-pmt-cs-plugins-local>`__.
If you use the teams version of voxel51, you need to upload the plugins
as shown in the figure below.

.. image:: ../figs/plugins_cloud.png
   :alt: Upload plugins on Voxel51 Teams

To load the plugins to your fiftyone deployment, e.g., on
localhost:5151, start the fiftyone app and pass the following command if
you want to use for example the Ford Dat3 UI plugins:

::

   fiftyone plugins download https://UserID:APIKEY@artifactory.boschdevcloud.com/artifactory/xai-pmt-plugins-cs-local/v3d-fo-plugin_v0.3.zip

To upload locally downloaded plugins to your fiftyone deployment, pass
the following command if you want to use for example the Ford Dat3 UI
plugins:

::

   import fiftyone.management as fom
   fom.upload_plugin("Path_To_Plugin_zip/v3d-fo-plugin_vx.x.zip", overwrite=True)

Backend artifacts
------------------
**Backend artifacts** are the fields that are necessary in your db for
xai plugins to work. For the ease of a user, we provide the pip package
to get these fields written to your database. The backend artifacts are
also available on Jfrog artifactory
`repo <https://artifactory.boschdevcloud.com/ui/repos/tree/General/xai-pmt-cs-local>`__.
To gain access to both artifacts, please apply for security group name
XAIOnePMTUsersCs. You can replace pXXX with your desired python version
p38, p39, p310, p311.

::

   pip install --extra-index-url https://UserID:APIKEY@artifactory.boschdevcloud.com/artifactory/api/pypi/xai-pmt-cs-local/simple xaipostprocess-0.7-cpXXX-cpXXX-linux_x86_64.whl
