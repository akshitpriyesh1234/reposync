Operations
===========

CI/CD
-----

**Continuous integration (Backend artifacts)**
Please refer to the figure modular_build_backend pipeline below.
For each PR raised to our backend artifactory master, the PR pipeline is auto triggered. The PR pipeline needs two inputs, a) voxel version to use for performing the test,
b) The plugin version (.zip) to use for the UI testing. The latter is required because there are some minimal UI tests that should pass to merge to master, although your contribution is only to the backend artifact.
In short, the UI test uses your new changes in the backend artifact to make sure no frontend components are breaking.
After the unit test and the UI tests, the FOSS scans/Copyright scans and keyword scans are performed.
Finally the pip artifact is published to the artifactory, if and only if it is the CI pipeline, i.e., the only build is performed in the PR pipeline.
Prior to the publishing, the PR description is taken from the user and added to the CHANGELOG.md and a new version is also incremented.
The decision if its a major, minor or a patch version increment is based on the user input in the PR description.
After the Unit tests passed, the sphix documentation for the artifact is built and pushed to the dev storage account for a PR pipeline.
Likewise, during the CI build, after the pip wheels are built, the approved sphinx documentation is built and push to the prod storage folder.
It is from this prod folder the documentation web app is rendered.

.. drawio-image:: figs/pipeline_be.drawio.png
   :export-scale: 150

**Continuous integration (Frontend artifacts)**
Please refer to the figure modular_build_frontend pipeline below.
In the case of our CI/CD pipeline for deployment of the frontend artifacts, the pipeline needs two inputs.
a) The voxel version to use for building the plugin zip artifact, b) The version with with the created artifact should be tagged with.
In this case as of today the CHANGELOG.md is expected to be manually added to the file by the user.
The PR pipeline will stop at the point of building the plugins and addons. However, the CI pipeline will in addition to build and pack the artifacts (incl. addons), will also push to the artifactory.

.. drawio-image:: figs/pipeline_fe.drawio.png
   :export-scale: 150

**Sphinx doccumentation**
During the CI build you will notice that there are some addition steps that is not detailed on the above pipeline.

1) What is createdocsinfra parameter in the CI pipeline ?

This is a parameter that is needed if you want to deploy the sphinx documentation web app infra from scratch.
Setting it to yes will spawn an app service plan, a corresponding app service with the docs docker from xaimasteracr.azurecr.io/docs, mount the html files
from xaioperations, and set the azure active directory authentication for the app service too.
However, this is a one time trigger and does not have to be triggered often.

.. drawio-image:: figs/docs4xai.drawio.png
   :export-scale: 100
   :align: center

2) How are the new built docs (html files) rendered in the web app ?

The PR pipeline will build the sphinx documentation and push to the xaioperations blob storage in dev folder.
But the dev folder is not served by the app service.
Once the PR i approved the same docs are built and pushed to the prod folder of xaioperations blob storage.
The docker image that is served on the app service will render all the html files that are present in the
prod folder (which are built by the CI pipeline after the PR is approved). In short the app service will merely host a static web app.
By keeping the files on an external storage keeps the docker image lean and app service faster.


UI Test
--------
The UI tests are performed using selenium docker. To perform UI tests three fundamental blocks are necessary,
a) The selenium docker, b) The mongo docker c) The test script, d) The built plugin zip artifact.
The agent (Azure pipeline) will initially take in your recent changes (can be backend or frontend) and builds it.
As shown in the figure below the entire test is performed on the Azure build agent.
The blob storage with the unit test dataset is mounted to the azure agent.
The necessary environment is set, i.e., setting up of conda environment and installing the necessary packages.
The selenium and mongo dockers are pulled and started.
Now the environment is available on the build agent with database and selenium docker, fiftyone is installed and the plugin is built.
In the case of UI test being triggered by our backend artifact repo, the plugins are just pulled from the artifactory and the backend
artifact is tested. That is the new change in the backend artifact should not break anything in the frontend.
However, in the case that the UI test is triggered by our frontend repo, the plugins are built using your recent changes that you raised a PR for.
It is for this reason, we denote in the below figure we mention this as Build/upload plugin before starting the app.
With is being set, the fiftyone app is started and let run silently in the localhost of the agent.
The test script is now run on the UI that is being hosted in localhost. When there is a test case fail or crash,
the browser console logs are zipped and the corresponding screenshot is taken and provided to the user as an artifact.
The selenium docker is using chromium drivers but if required a user can change this to firefox too.

.. drawio-image:: figs/UItesting.drawio.png
   :export-scale: 100
   :align: center

UI test cases/coverage
~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table:: Table Title
   :file: general/v3d_ui_tcs.csv
   :header-rows: 1
   :class: longtable

XAI @ Voxel51 Teams
--------------------
Currently OnePMT is having 2 licenses (1 Admin and 1 Member) at the Dyper51 deployment of Voxel51.
Now if one of the non-licenced user wants to write a new dataset for their use-case. They can use our voxel51_db_update job.
Currently we limit the access rights to this job only to XAI team members (at EDL/BGSW/Sunnyvale) because this is not a service we offer, its merely
a provisioning we made for our team to exploit the licenses to an optimal level.

.. drawio-image:: figs/teamsV51.drawio.png
   :export-scale: 120
   :align: center


As a user, you use our AML template stepwise.py, to submit the job. You as a user need to only change what importer script you want to run.
For example below, you only need to change the contents of :code:`bash run_main.sh`. For example, in
our case we want to import datasets from the blob store and then run the xai operations, i.e., gt_dt, slice teller and dsconfig.
In this case you can collect these steps as a bash script and save it in run_main.sh. The advantages are:

- You can run several python files in one bash file.
- You can also choose to install any custom packages, i.e., pip install -r requirements.txt or even apt packages that is mandatory for your python script to run.

Our kind request is to use the NC4as-T4-v3-vnet compute for GPU because its relatively cheaper.
::

   pipeline_env_check = ScriptRunConfig(
    source_directory='your/work/folder/',
    command=[
        'az login --identity && \
             export FIFTYONE_API_URI=$(az keyvault secret show --name APIURIPRIVATE --vault-name xaipmtamlwspac4589713775 --query value -o tsv) && \
             export FIFTYONE_API_KEY=$(az keyvault secret show --name APIKEY --vault-name xaipmtamlwspac4589713775 --query value -o tsv) && \
             export ARTIFACTORYTOKEN=$(az keyvault secret show --name ARTIFACTORYTOKEN --vault-name xaipmtamlwspac4589713775 --query value -o tsv) && \
             bash run_main.sh', input_data,
    ],
    compute_target='NC4as-T4-v3-vnet',
    environment=env,
   )

The aml cluster uses the fiftyone teams docker image from the our registry xaimasteracr.
Now if you want to use the newer version of voxel51 teams, you need to trigger our pipeline modular_build_backend/xai-teams-env-backend with the new version as a parameter.
This will pick up the install key from variable groups, build the docker with fiftyone teams version of your choice, push to the ACR and register the image on AML.
So for the next job you submit, the newest environment is used by default.
The cluster talks to the DYPER51 deployment of voxel51 though the private endpoint. This private endpoint is created by between our subnet to DYPER51.
Likewise, our blob container xaiops in storage account xaioperations is already mounted to DYPER51 AKS cluster. As this datastore is already
registered to the AML workspace, you can access the data from the datastore in your job directly.
