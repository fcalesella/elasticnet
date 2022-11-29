# Elastic net
 Cross-validated and bootstrapped elastic-net penalized regression
 
 Created by: Federico Calesella, Silvia Cazzetta, Federica Colombo, & Benedetta Vai\
 Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, IRCCS San Raffaele Scientific Institute, Milan, Italy

## Table of Contents
1. [Project Overview](#Project_Overview)
2. [Setup](#Setup)
   1. [System Requirements](#System_Requirements)
   2. [Installation](#Installation)
5. [Instructions for Use](#Instructions_for_Use)
6. [Demo](#Demo)

## 1. Project Overview <a name="Project_Overview"></a>
The code is intended to estimate elastic net regression in a machine learning framework, for target prediction purposes. Specifically, it comprises the following pipelines: 
- K-Fold nested cross-validation: for predictive accuracy assessment; 
- bootstrap: for assessment of relevant predictors.

Relevant specifications:
- both linear and logistic regression are supported;
- both alpha and lambda hyper-parameters can be optimized (for the lambda optimization please refer to the [```lassoglm``` MATLAB documentation](https://it.mathworks.com/help/stats/lassoglm.html));
- the cross-validation procedure is stratified (i.e., the proportion of classes in the dataset is maintained throughout the folders);
- the optional use of class weights for imbalanced classification tasks is made available;
- the optional use of stratified bootstrap (i.e., the proportion of classes in the dataset is maintained throughout the resamplings) is made available. 

## 2. Setup <a name="Setup"></a>
### i. System Requirements <a name="System_Requirements"></a>
The code was tested on the following operating systems:
- Linux
- Mac OSX
- Windows

The code was tested on MATLAB versions 2016b and 2017a. Previous and subsequent versions might differ in some MATLAB built-in employed functions.\
Please note that the MATLAB Statistics and Machine Learning Toolbox and Parallel Computing Toolbox are required in order to run the code.

### ii. Installation <a name="Installation"></a>
For installation the repository folder must be downloaded or cloned.\
If the ZIP folder is downloaded, then it will be required to un-zip it, whereas to clone the repository a working github version will be required.  
Lastly, the folder and subfolders will have to be added to the MATLAB path. The setup process should require only a few minutes.

## Instructions for Use <a name="Instructions_for_Use"></a>
To estimate elastic net regression both in a stratified nested cross-validation and bootstrap routines, the ```elastic_net``` script (demo folder) can be run on your data. Data are expected to be organized in an excel file with subjects in rows and features in columns. The dataset should include the target variable (it is suggested to be in the first or last column).\
Input parameters of ```elastic_net``` and their explanation can be found in the first lines of the script.\
Relevant outputs are:
- ```params```: is a structure array containing the parameters passed to ```lassoglm``` for elastic-net penalized regression fitting.
- ```cv```: is a structure array containing the results of the nested cross-validation procedure (the following data are saved in the structure array over the cross-validations: coefficients, best lambda value, best alpha value, AUC, ROC coordinates, true observations, and predictions).
- ```measures```: is a structure array containing the accuracy measures of the cross-validated model.
- ```figures```: is a structure array containing the plots derived from the cross-validated model.
- ```bootprint```: is a structure array containing the results of the bootstrap. The fit information, coefficients, alpha value, and lambda value are reported for each bootstrap iteration.
- ```resboot```: is a table containing the statistics derived from the bootstrap procedure. The mean, median, standard deviation, lower bound of the confidence intervals, upper bound of the confidence intervals, and variable inclusion probability (VIP; frequency of non-zero values over the bootstrap iterations) are reported.

Once the analyses are done, the following code (file names and formats have to be adapted) can be used to save the model and the results in the working directory:
```matlab
save('model_name.mat');
writetable(resboot, 'file_name.xlsx');
saveas(figures.figure_name, 'file_name.desired_format');
```

If of interest, then, the ```sig_test``` script might be run in order to test the model including only some relevant predictors as resulted from the bootstrap. This script will load and use some of the variables created by the ```elastic_net``` script (i.e., ```data```, ```params```, and ```resboot```).\
The explanation of the input parameters is reported in the first lines of the script.\
Relevant outputs are:
- ```out```: is a structure array containing the true observations and the predicted ones.
- ```final_measures```: is a structure array containing the accuracy measures of the final predictions obtained from the bootstrapped coefficients.
- ```final_figures```: is a structure array containing the plots of the final model obtained from the bootstrapped coefficients.

To save the model in the working directory, type (file names and formats have to be adapted):
```matlab
save('final_model_name.mat');
saveas(final_figures.figure_name, 'file_name.desired_format');
```

## Demo <a name="Demo"></a>
The ```elastic_net``` script (demo folder) can be run in order to try the code and perform both nested cross-validation and bootstrap on synthetic data (see the [Instructions for Use](#Instructions_for_Use) section for instructions on how to run the script)
The synthetic data was created using the ```make_classification``` function of the *scikit-learn* (version 0.23.2) package in *python*, and it is in excel format. The code to produce the synthetic data is also available in the demo folder (*create_synthetic_data.py*): the full path where to save the dataset (in excel) needs to be provided at line 12.

To run the *elastic_net.mat* file, it is recommended to position the MATLAB working directory in the demo folder, and type ```elastic_net``` in the Command Window. The expected output should be structured as follows:
```matlab
>> elastic_net

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
The seed was set 1234 for reproducibility. By changing the seed, some statistical variation may be observed, but stability should improve by increasing the number of bootstrap iterations.
The execution took around 70s on a DELL XPS-13 (processor: Intel i7; RAM: 16GB; OS: Windows11).

The model might be saved using the following code:
```matlab
save('model.mat');
```
