# Elastic net
 Cross-validated, bootstrapped, and permuted elastic-net penalized regression

## Table of Contents
1. [Project Overview](#Project_Overview)
2. [Setup](#Setup)\
   2.1 [System Requirements](#System_Requirements)\
   2.2 [Installation](#Installation)
3. [Instructions for Use](#Instructions_for_Use)\
   3.1 [Script](#script)\
   3.2 [Function](#function)\
   3.3 [Standalone application](#standalone_application)\
   3.4 [Relevant outputs](#relevant_outputs)
4. [Demo](#Demo)
5. [Contributors](#Contributors)

## 1. Project Overview <a name="Project_Overview"></a>
The code is intended to create a wrapper to easily estimate an elastic-net penalized regression model, and perform bootstrap and/or permutation analysis on it. The code was conceived to provide users with a full set of analyses by setting just a few parameters, without the need of writing code.

## 2. Setup <a name="Setup"></a>
### 2.1 System Requirements <a name="System_Requirements"></a>
The code was tested on the following operating systems:
- Linux
- Mac OSX
- Windows

The code was tested on MATLAB versions 2016b and 2017a. Previous and subsequent versions might differ in some MATLAB built-in employed functions.\

### 2.2 Installation <a name="Installation"></a>
For installation, the repository folder must be downloaded or cloned.\
If the ZIP folder is downloaded, then it will be required to un-zip it, whereas to clone the repository a working github version will be required.  
If a working version of MATLAB is intalled, the *src* folder will have to be added to the MATLAB path, otherwise the standalone app may be used (see section [3. Instructions for Use](#Instructions_for_Use)). The setup process should require only a few minutes.

## 3. Instructions for Use <a name="Instructions_for_Use"></a>
There are three possible ways to run the code:
- the ```elasticnet``` script
- the ```elasticnet``` function
- the ```elasticnet``` standalone app

N.B. To run the script and the function a working MATLAB is needed, whereas for the standalone app only a MATLAB runtime is required.\
N.B. To run the script and the function the MATLAB Statistics and Machine Learning Toolbox and Parallel Computing Toolbox are required in order to run the code.


### 3.1 Script <a name="script"></a>
To run the ```elasticnet``` script some parameters have to be set at the beginning of the script. An explanation of the meaning of the parameters is provided in the script above each respective parameter, and reported here:
- ```seed```: specify the seed for random processes, for reproducibility. Seed can be a scalar or a string (e.g., 'Default'; please refer to the MATLAB function rng). If 'false' is given to seed, no seed will be set.
- ```data_options.DataPath```: string containing the full path to the database.
- ```data_options.Sheet```: string specifying the name of the spreadsheet to be used. Leave empty to load the default sheet (the first one).
- ```data_options.Target```: specify the outcome variable (column number).
- ```data_options.Predictors```: specify the model indipendent variables (column number).
- ```data_options.TargetType```: specify regression type. Type: 'binomial' for logistic regression or 'normal' for linear regression.
- ```data_options.Normalization```: specify normalization type. Type: 'standard' for standardization or 'minmax' for min-max normalization. 
- ```cv_options.KOuter```: specify the number of folds in the outer CV. 
- ```cv_options.KInner```: specify the number of folds in the inner CV.
- ```cv_options.Alpha```: specify the values of the alpha hyper-parameter (trade-off between the L1 and L2 regularizations). Alpha can be scalar or a numeric vector. If a numeric vector is provided, the alpha value will be optimized in the cross-validation.
- ```cv_options.Lambda```: pecify the set of values on which the lambda hyper-parameter will be optimized. If empty, the default MATLAB optimization of [lassoglm](https://it.mathworks.com/help/stats/lassoglm.html) will be performed, otherwise define a sequence of values such as logspace(-5, 5, 100), in order to test the lambda on 100 values from 10e-5 to 10e5.
- ```cv_options.Weighted```: specify if class weights will be assigned to the observations in an inversely proportional fashion to the number of observations in each class following $W_c = \frac{1}{n_c s_c}$ where $n_c$ is the number of classes, and $s_c$ is the number of subjects belonging to the class $c$. This might be useful in the context of classification on imbalanced data. Type: true to assign class weights or false otherwise.
- ```bootstrap_options.ModelDefiner```: specify the model that will be undergo bootstrap. Type 'optimize' to perform the optimization of the lambda and alpha hyper-parameters as specified in the cross-validation settings. Otherwise, set a callable (i.e., a function). Callables are passed by putting @ before the desired function (@function). Some examples of functions that can be used are central tendency mesures(e.g., @mean, @median, or @mode).
- ```bootstrap_options.NResamples```: specify number of bootstrap iterations. 
- ```bootstrap_options.BootstrapType```: specify the method to calculate confidence intervals (CIs). Type: 'norm' for normal CIs, 'per' for percentile CIs, 'cper' for corrected percentile CIs, 'bca' for bias-corrected CIs, or 'stud' for studentized CIs. Check the MATLAB page of [bootci](https://it.mathworks.com/help/stats/bootci.html) for further details on the options.
- ```bootstrap_options.Alpha```: specify the significance level.
- ```bootstrap_options.SE```: specify the number of resmaplings in the inner bootstrap loop for the calculation of the studentized standard error (SE) estimate. This option will be ignored if the BootstrapType is not studentized ('stud'). The MATLAB default for this parameter is 100. Check the MATLAB page of [bootci](https://it.mathworks.com/help/stats/bootci.html) for further details on this option. 

The script is organized in sections in order to let the user choose whether to perform the bootstrap. It should be noted, though, that the bootstrap cannot be run without before running the cross-validation procedure or having the required inputs.\
It is also possible to run a permutation test to assess if the model predictive score is significantly different from a null model, using the following code (note that in the workspace there should be the variables generated by the ```elasticnet``` script):
```matlab
>>> npermutations = 500
>>> [pval, null_stat] = permute_model(X, y, bootstrap_options.Model, npermutations);
```
where ```npermutations``` can be modified in order to decide how many permutations should be performed. The ```X``` and ```y``` are the predictors and target, and the ```bootstrap_options.Model``` is the model that should be permuted. The ```pval``` variable stands for the p-value and the ```null_stat``` is an array containing the scores of the permuted models. The scores are the balance accuracy in the case of dichotomous target and the R<sup>2</sup> for continuous targets.
Once the analyses are done, the following code can be used to save the model and the results in the working directory (see section [3.4 Relevant outputs](#relevant_outputs) for further details):
```matlab
>>> save('model.mat');
>>> writetable(performance, performance.xlsx);
>>> writetable(bootstrap_results, bootstrap_results.xlsx)
```
Note that the cross-validation section will prompt two figures:
- the confusion matrix
- the receiver operating characteristic (ROC) curve (for each fold of the outer layer and their mean)

### 3.2 Function <a name="function"></a>
The ```elasticnet``` function is contained in the *src* folder, so if the folder is in the MATLAB path (see section [2.2 Installation](#Installation)) the funciton can be called from everywhere. To run the funciton some both positional and Name-Value arguments have to be set. 

Positional arguments are:
- ```dataset```: string containing the full path to the database.
- ```target```: specify the outcome variable (column number).
- ```predictors```: specify the model indipendent variables (column number).
- ```target_type```: specify regression type. Type: 'binomial' for logistic regression or 'normal' for linear regression.
- ```normalization_type```: specify normalization type. Type: 'standard' for standardization or 'minmax' for min-max normalization. 
- ```kouter```: specify the number of folds in the outer CV. 
- ```kinner```: specify the number of folds in the inner CV.

Name-Value arguments with their default value are:
- 'Seed', false: specify the seed for random processes, for reproducibility. Seed can be a scalar or a string (e.g., 'Default'; please refer to the MATLAB function rng). If 'false' is given to seed, no seed will be set.
- 'Sheet', []: string specifying the name of the spreadsheet to be used. Leave empty to load the default sheet (the first one).
- 'Alpha', 0.5: specify the values of the alpha hyper-parameter (trade-off between the L1 and L2 regularizations). Alpha can be scalar or a numeric vector. If a numeric vector is provided, the alpha value will be optimized in the cross-validation.
- 'Lambda', []: Specify the set of values on which the lambda hyper-parameter will be optimized. If empty, the default MATLAB optimization of [lassoglm](https://it.mathworks.com/help/stats/lassoglm.html) will be performed, otherwise define a sequence of values such as logspace(-5, 5, 100), in order to test the lambda on 100 values from 10e-5 to 10e5.
- 'Weighted', false: specify if class weights will be assigned to the observations in an inversely proportional fashion to the number of observations in each class following $W_c = \frac{1}{n_c s_c}$ where $n_c$ is the number of classes, and $s_c$ is the number of subjects belonging to the class $c$. This might be useful in the context of classification on imbalanced data. Type: true to assign class weights or false otherwise.
- 'Bootstrap', false: boolean defining whether to perform the bootstrap procedure (true) or not (false).
- 'NResamples', 5000: specify number of bootstrap iterations. This parameter is ignored if 'Bootstrap' is false. 
- 'BootstrapType', 'norm': specify the method to calculate confidence intervals (CIs). Type: 'norm' for normal CIs, 'per' for percentile CIs, 'cper' for corrected percentile CIs, 'bca' for bias-corrected CIs, or 'stud' for studentized CIs. Check the MATLAB page of [bootci](https://it.mathworks.com/help/stats/bootci.html) for further details on the options. This parameter is ignored if 'Bootstrap' is false. 
- 'SE', 100: specify the number of resmaplings in the inner bootstrap loop for the calculation of the studentized standard error (SE) estimate. This option will be ignored if the BootstrapType is not studentized ('stud'). The MATLAB default for this parameter is 100. Check the MATLAB page of [bootci](https://it.mathworks.com/help/stats/bootci.html) for further details on this option. This parameter is ignored if 'Bootstrap' is false. 
- 'Permute', false: boolean defining whether to perform the permutations (true) or not (false).
- 'NIterations', 5000: specify number of permutation iterations. This parameter is ignored if 'Permute' is false. 
- 'ModelDefiner', @median: specify the model that will undergo bootstrap or permutations. Type 'optimize' to perform the	optimization of the lambda and alpha hyper-parameters as specified in the cross-validation settings. Otherwise, set a callable (i.e., a function). Callables are passed by putting	@ before the desired function (@function). Some examples of functions that can be used are central tendency mesures (e.g., @mean, @median, or @mode). This parameter is ignored if 'Bootstrap' and 'Permute' are false. 

Here a very basic example is provided:
```matlab
>>> elsticnet('./demo/synthetic_data_continuous.xlsx', 1, 2:21, 'normal', 'standard', 5, 5, 'Weighted', false)
```
The function will save a *.mat* file containing the relevant outputs of the funciton (see section [3.4 Relevant outputs](#relevant_outputs) for further details). To save the performance metrics in an excel file, the following code can be used:
```matlab
>>> load('elasticnet_results.mat');
>>> writetable(performance, performance.xlsx);
```
Eventually, to save the results of the bootstrap procedure in an excel file, type: 
```matlab
>>> load('elasticnet_results.mat');
>>> writetable(bootstrap_results, bootstrap_results.xlsx)
```
Contrarily to the script (see section [3.1 Script](#script)), the function will not display any figure. However, it will be possible to create them by:
```matlab
>>> load('elasticnet_results.mat');
>>> [~, figures] = metricsnplots(cv_options.RegressionType, cv_results);
```
N.B. When the working directory is not the *src* folder, make sure that the *src* folder is in the MATLAB path, otherwise MATLAB may not be able to find the function.

### 3.3 Standalone application <a name="standalone_application"></a>
To run the standalone application the ```elasticnet``` command should be called, followed by the input parameters. The input parameters are the same of the ```elasticnet``` function (see section [3.2 Function](#function) for details on the input parameters). Note that none of the input parameters will need to be strings, instead it will be suffcient to type directly in the values separated by a space. This also applies to Name-Value arguments, where the name is typed-in as any other value. Some differences, though, are present between the Windows and Linux versions.\
To run on Windows it is sufficient to open the command window by searching for *cmd* in the programs and type the command followed by the input parameters. Here is an example:
```console
path_to_elasticnet_folder> elsticnet ~/elsticnet/demo/synthetic_data_continuous.xlsx 1 2:21 normal standard 5 5 Weighted false
```
N.B. The ```Weighted false``` part is defining a Name-Value argument, but it must be simply enetered as two inputs seprated by a space like all the other paramters.

To run on Linux, instead, the path to the runtime should be provided before the input parameters:
```console
user@username:~$ elasticnet <path_to_runtime> ~/elsticnet/demo/synthetic_data_continuous.xlsx 1 2:21 normal standard 5 5 Weighted false
```
For the Linux version, it is also possible to run the app through the *run_elasticnet.sh* file.\ 
In both cases make sure that the files have the required permissions. If a permission error is raised, try:
```console
user@username:~$ chmod +x run_elasticnet.sh
user@username:~$ chmod +x elasticnet
```
The app will save a *.mat* file containing the relevant outputs (see section [3.4 Relevant outputs](#relevant_outputs) for further details). To save the performance metrics in an excel file, the following code can be used in MATLAB:
```matlab
>>> load('elasticnet_results.mat');
>>> writetable(performance, performance.xlsx);
```
Eventually, to save the results of the bootstrap procedure in an excel file, type: 
```matlab
>>> load('elasticnet_results.mat');
>>> writetable(bootstrap_results, bootstrap_results.xlsx)
```
Contrarily to the script (see section [3.1 Script](#script)), the app will not display any figure. However, it will be possible to create them by:
```matlab
>>> load('elasticnet_results.mat');
>>> [~, figures] = metricsnplots(cv_options.RegressionType, cv_results);
```
N.B. In both the Windows and Linux versions make sure to be in the directory of the app, or to provide the right paths.\
N.B. In the *standalone* folder the *elasticnet.mat* file contains the source file that was binarized to build the app.

### 3.4 Relevant outputs <a name="relevant_outputs"></a>
Relevant outputs are:
- ```data_options```: is a structure array containing the information needed to properly handle the data.
- ```cv_options```: is a structure array containing the parameters passed to the cross-validation procedure and the lassoglm function for elastic-net penalized regression fitting.
- ```cv_results```: is a structure array containing the results of the nested cross-validation procedure (the following data are saved in the structure array over the cross-validations: coefficients, best lambda value, best alpha value, AUC, ROC coordinates, true observations, and predictions).
- ```performance```: is a table containing the accuracy measures of the cross-validated model.
- ```figures```: is a structure array containing the plots derived from the cross-validated model. This output is available only for the ```elastic_net``` script.
- ```bootstrap_options```: is a structure array containing the parameters passed to the bootstrap function and the information about the model that will undergo the bootstrap procedure.
- ```bootstrap_results```: is a table containing the statistics derived from the bootstrap procedure. The mean, median, standard deviation, lower bound of the confidence intervals, upper bound of the confidence intervals, and variable inclusion probability (VIP; frequency of non-zero values over the bootstrap iterations) are reported.
- ```X```: array containing the predictors extracted from the database.
- ```y```: array containing the target extracted from the database.

## 4. Demo <a name="Demo"></a>
The ```elasticnet``` script can be run in order to try the code and perform both nested cross-validation and bootstrap on synthetic data (see section [3. Instructions for Use](#Instructions_for_Use) section for instructions on how to run the script).
The synthetic datasets were created using the ```make_classification``` and ```make_regression``` functions of the *scikit-learn* (version 0.23.2) package in *python*, and they are provided in excel format. The code to produce the synthetic datasets is also available in the demo folder (*create_synthetic_data_classification.py* and *create_synthetic_data_regression.py*): the full path where to save the dataset (in excel) needs to be provided at line 12.

To run the *elasticnet.mat* file, just type ```elasticnet``` in the Command Window (make sure that the script is in the working directory). The expected output should be structured as follows:
```matlab
>> elasticnet

Running cross-validation...
Selected parameters:
External CV: 5 fold
Internal CV: 5 fold

Cross-validation state: done.

Cross-validation results:
MSE = 0.0009
RMSE = 0.0304
R squared = 0.9991

Running Bootstrap...
Selected number of iterations: 500

Bootstrap state: done.

Bootstrap statistics:
     Variables        Mean          Median          SD          LowerCI       UpperCI      VIP 
    ___________    ___________    __________    __________    ___________    __________    ____

    'Intercept'     0.00015337    0.00022764     0.0032277     -0.0064795     0.0061727     100
    'x1'                     0             0             0              0             0       0
    'x2'            1.8433e-05             0    0.00014581    -0.00030421    0.00026734     1.6
    'x3'                     0             0             0              0             0       0
    'x4'                     0             0             0              0             0       0
    'x5'                     0             0             0              0             0       0
    'x6'                     0             0             0              0             0       0
    'x7'                     0             0             0              0             0       0
    'x8'                     0             0             0              0             0       0
    'x9'                0.1807       0.18106     0.0030155        0.17497       0.18679     100
    'x10'                    0             0             0              0             0       0
    'x11'                    0             0             0              0             0       0
    'x12'                    0             0             0              0             0       0
    'x13'              0.96615       0.96618     0.0024425        0.96231       0.97189     100
    'x14'           0.00020816             0    0.00081842     -0.0018122     0.0013959    12.4
    'x15'                    0             0             0              0             0       0
    'x16'                    0             0             0              0             0       0
    'x17'                    0             0             0              0             0       0
    'x18'                    0             0             0              0             0       0
    'x19'          -3.1374e-05             0    0.00034972    -0.00065407    0.00071682     0.8
    'x20'                    0             0             0              0             0       0
```
The seed was set to 1234 for reproducibility. By changing the seed, some statistical variation may be observed, but stability should improve by increasing the number of bootstrap iterations.
The execution took around 70s on a DELL XPS-13 (processor: Intel i7; RAM: 16GB; OS: Windows11).

## 5. Contributors <a name="Contributors"></a>
Federico Calesella\
Silvia Cazzetta\
Federica Colombo\
Beatrice Bravi\
Mariagrazia Palladini\
Benedetta Vai

Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, IRCCS San Raffaele Scientific Institute, Milan, Italy
