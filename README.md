# Installation

Download by using

`git clone https://github.com/JulesLebert/spikesorting_scripts.git`

Install all the dependencies by using cd to this directory and then use

`pip install -e .`

All the spike sorters have to be installed separately

# Running spikesorting on concatenated recordings

Edit the paramaters of the spikesorting in concatenated_recordings_params.json

Navigate in scripts/ and run

`python spikesorting_concatenated_NP.py json_files/concatenated_recordings_params.json`

The jobs folder contains job files to run on the UCL cluster myriad (https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/)
