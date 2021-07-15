
# Setting up

We need some packages, a pip freeze can be found there
with the exact versions used (Personally, I do not really
believe in this, replicating runs 100% is sometimes hard,
and if a second run with slightly different version results
in a huge difference, this should be investigated instead
of swept under the rug).
I tried to fix all random numbers etc so that runs should 
be deterministic, but did not explicitly test for this.

To install all packages:

$ pip install -r requirements.txt 

It *might* be possible that pymrmre needs to  be installed last. 
I installed it afterwards and needed to install other packages
before pymrmre.

If you want to use a virtual environment for this,
you can execute something like this, in case of python 3.6

$ virtualenv -p /usr/bin/python3.6 /data/radCV/venv
$ source /data/radCV/venv/bin/activate

We also need the scikit-feature package from github.
The version is probably not important, as the package has
not been updated since 2 years, so any version from 2021 will do.
Unfotunately, I had to modify a small thing to get rid of 
warnings, so the version to use is below ./3rd/ .
To install, change to that directory and execute

$ python setup.py install



# Experiment

The experiment is then started with ./startExperiment.py
It will write all the artifacts into /data/radCV/mlruns
One can change this path by changing the TrackingPath
variable at the beginning of the file.
Also, it uses 24 cores for running, this can be changed
at the very bottom of the file.

Experiments already executed will not execute a second time.
One my machine I had during development several strange
crashed, I believe these stem from race conditions, so to
avoid to restart everything, I implemented a simple check.

The mlflow ui can be started by

$ mlflow ui --backend-store-uri file:///data/radCV/mlruns

It can be used to either track the experiments or to
just look at the metrics.

**Note**: Because mlflow is only used during storing of the results,
the timing shown in mlflow is not the training time!


# Evaluation

Evaluation code is unfortunately rather messy.
Some extra packages needs to be installed, e.g.
cm-super, dvipng packages are needed for plotting.
(Unfortunately no requirements.txt available)

Evaluation needs access to the whole mlruns folders,
because it needs to recompute some of the results,
which were not computed during the experiment.
The path can be found in at the beginning in
TrackingPath = "/data/radCV/mlruns"

If the experiment is not re-executed, the
mlruns needs to be _exactly_ at this place, else artifacts
will not be found, as these seem to be hardlinked
in the meta.yml files in the mlrun folders.



