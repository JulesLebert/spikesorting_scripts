# Installation

Download by using

```bash
    git clone https://github.com/JulesLebert/spikesorting_scripts.git
```

Install all the dependencies by using cd to this directory and then use

```bash
    pip install -e .
```

All the spike sorters have to be installed separately (see https://spikeinterface.readthedocs.io/en/0.13.0/sortersinfo.html)

# Running spikesorting on concatenated recordings

Edit the parameters of the spikesorting in concatenated_recordings_params.json

Navigate in scripts/ and run

```bash
    python spikesorting_concatenated_NP.py json_files/concatenated_recordings_params.json
```

The jobs folder contains job files to run on the UCL cluster myriad (https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/)


# Installation of pykilosort on myriad

I highly recommand to use conda for pykilosort on myriad as cupy can be complicated to install without

If a python module is unload, unload it by using

```bash   
    module unload python
```

And load python with conda using

```bash
    module load python/miniconda3/4.10.3
    source $UCL_CONDA_PATH/etc/profile.d/conda.sh
```

Create a new conda environment

```bash
    conda create --name ss_pykilosort python=3.9
    conda activate ss_pykilosort
```

And install dependencies

```bash
    conda install cupy
    pip install phylib pypandoc
    # recommended
    git clone --branch ibl_prod https://github.com/int-brain-lab/pykilosort

    cd pykilosort
    pip install -r requirements.txt
    pip install -e .
```

Example of job running pykilosort in `jobs/run_single_NP_pykilosort.sh`