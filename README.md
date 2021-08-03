# Language Identification for Text

This repository defines experiments that compare performance of different approaches to language identification for text, and demonstrates an number of reasonable general choices and practices.  It began as a fresh fork of <https://github.com/TomLippincott/new-project.git>, so that repository may be a useful starting-point for similar projects.

## Initial setup

The following commands only need to be run once.  First, clone the LID experiments repository, and change directory into it:

```
git clone https://github.com/TomLippincott/lid.git
cd lid
```

Then, create and activate a Python 3 virtual environment:

```
python3 -m venv local
source local/bin/activate
```

Install the required packages into the virtual environment:

```
pip install -r requirements.txt
```

Download and unpack the data sets:

```
wget -O - http://www.logical-space.org/img/data.tgz | tar xpz
```

Copy the default configuration into place:

```
cp custom.py.example custom.py
```

At this point you should be able to invoke a dry run of the predefined experiments:

```
scons -n
```

This will print out a long list of commands that SCons would run (see below for *actually*
running them).  Finally, you can exit the virtual environment (or simply end the terminal session etc):

```
deactivate
```

## Basic use

In the previous section, running `scons -n` printed what the build system would do, without
actually doing it.  Note that what it's printing are *simple shell commands*: you could 
copy-paste them in sequence and get the same results as running the actual build system.  
However, generating them programmatically helps manage the complexity and scale, and because 
SCons understands the dependencies between the steps, and tracks changes to the scripts, much 
effort is saved and many errors avoided.  Moreover, as explained in the [Advanced use](#advanced)
section, it becomes trivial to run on a compute grid to exploit the massive parallelism of
most experimental pipelines.

For the moment though, you should be able to simply invoke `scons`, and depending on your
computer, the experiments should fully run in under an hour.  Afterwards, try running it
again: you should see a message that everything is "up to date".  If you examine the `work/` 
directory, you'll see all the files produced by the experiment: until one of them disappears
or has a newer timestamp than those derived from it, or one of the scripts changes, SCons
knows everything is consistent.  Try deleting something (maybe a model file) and run `scons -n`:
you'll see that SCons now wants to rebuild the missing file and everything that depended on
it.

Using this code base/general approach will often follow a simple pattern (step 4 is where 
the actual work happens):

1. Start terminal session in the `lid` directory
2. Activate the virtual environment with `source local/bin/activate`
3. Open various files in your editor
4. Go back and forth between modifying files, running `scons`, pushing changes to Github...
5. Exit the virtual environment and hopefully use the results in a paper

You may spend weeks or months in step 4 in the same terminal session, locally on a laptop or 
remotely (kept alive via `tmux`), focused on research but with the assurance that what you're
doing is preserved, portable, replicable, and scalable.

## In depth explanation

The `SConstruct` file is heavily documented and describes the experimental pipeline such that the `scons` command can determine the order in which to invoke scripts to go from the data downloaded earlier all the way to a PDF containing tables and visualizations!  The structure of the pipeline and implementation of each step are mostly decoupled: at the start of the project, we laid out the abstract build steps in their entirety:

1. **Preprocess a raw dataset to a common JSON format**
2. Randomly split a JSON dataset into train/dev/test
3. **Train a model on train/dev inputs using a parameter dictionary**
4. **Apply a model on test input**
5. Evaluate model output
6. Generate figures from model output
7. Write out a report based on 1-6

Step 1 needs to be implemented once for dataset.  Steps 3 and 4 need to be implemented for each model type.  In the `SConstruct` file, these rules are defined by iterating over the `DATASETS` and `MODELS` variables and applying a simple naming convention for the associated script name.  Therefore, it's unnecessary to modify the `SConstruct` file to add a new dataset or model type: entries just need to be added to those variables, and the corresponding scripts put in place.

### Variables and customization

The variables defined in `SConstruct` are overridden by the contents of `custom.py`, providing a nice way to quickly move between laptop and grid, CPU and GPU, etc.

## <a name="advanced"></a> Advanced use

Beyond providing organization and implicit documentation, a huge advantage of this approach is how easy it is to scale up experiments to run on a grid and/or with GPU hardware (right now there are still some hard-coded aspects regarding the HLTCOE grid, but that can be fixed easily).  At the top of the `custom.py` file there are two variables set to `False`: `USE_GPU` and `USE_GRID`.  When `USE_GPU=True`, the system checks if a command has the `can_use_gpu` property set to `True`, in which case it sets some appropriate flags and (on the grid) makes a few choices about how and where the command will be run.  `USE_GRID=True` is more interesting: it translates each command into a corresponding call to `qsub` and, critically, *maps the build-dependency structure into the grid system's job scheduler*.  Instead of running the *experiment's commands*, SCons runs the *submit commands* corresponding to them, and then returns to the terminal prompt.  Don't be fooled!  Run `qstat` and you should see that all those submitted commands are queued up or running.

## References and further information

[SCons build system](http://www.scons.org)
