# DNA-Deep Learning simulation study

## Repository structure
This repository is organized in the following manner:
|Directory| Description|
|---|---|
|'simulation-script'|All msprime related scripts that generate the sequences and scripts on antibiotic resistence|
|'dataset'|All dataset (sequence and label) should be saved in this directory, each sequence set and its corresponding label set should have a unique name, in the form of `sequence-NAME.in` and `label-NAME.in`. Currently includes a script that generate fake data and label. **This directory will be removed for the finalized repository, it's here just to show our pipeline structure**|
|'nn-model-script'|The **only** script that runs all NN models. The difference between different analysis will occurs in arguments and JSON files. We will use the same model script for all our analysis to make our life easier.|
|'model-spec'|Each model (RNN,CNN,Transformer...) will have a .json file with all its pre-defined parameter in it. Each json file will include the same number of entries(RNN model will have entries such as `filterSize` and `strideSize`). If the model does not have a certain parameter, we leave the value empty in the json file (For RNN, we will have `"filerSize": ""`).|
|'output'| Save all output file and scripts for plotting our model performance here. The output file should have the same name as its input sequence/label. The format should be `NAME.out`. The `NAME.out` files will be deleted for the finalized repository, and only visualization scripts will be kept.|
