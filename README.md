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
Warning: Variable names were modified to make them valid MATLAB identifiers. The original names are saved in the VariableDescriptions property. 
Starting parallel pool (parpool) using the 'local' profile ... connected to 12 workers.

Selected parameters:
External CV: 10 fold
Internal CV: 10 fold
Bootstrap: 500 iterations

Running cross-validation...
Cross-validation state: done.

Sensitivity = 0.9600
Specificity = 0.8800
PPV = 0.8889
NPV = 0.9565
Balance Accuracy = 0.9200
Diagnostic Odd Ratio = 176.0000
AUC across cross-validations:
    1.0000    1.0000    1.0000    1.0000    1.0000    0.9200    0.6800    1.0000    1.0000    0.9200

AUC statistics:
Mean = 0.9520
S.D. = 0.1012


Running Bootstrap...
Bootstrap state: done.

Bootstrap statistics:

     Variables       Mean        Median        SD       LowerCI     UpperCI     VIP 
    ___________    _________    _________    _______    ________    ________    ____

    'Intercept'      -1.3915      -1.2248    0.81054     -2.9802     0.19718     100
    'x1'            -0.77865     -0.66295    0.62717     -2.0079     0.45059    95.4
    'x2'            -0.65308     -0.58322    0.61492     -1.8583     0.55216    90.8
    'x3'             -2.1757      -1.9838    0.85562     -3.8527     -0.4987     100
    'x4'           -0.070612            0    0.43332    -0.91992      0.7787      75
    'x5'             0.26694      0.27465    0.66673     -1.0398      1.5737    87.8
    'x6'             -1.1764      -1.0401    0.79724     -2.7389     0.38623    95.8
    'x7'              0.3972      0.30265    0.58046    -0.74051      1.5349    82.4
    'x8'             0.21779      0.06965    0.41958     -0.6046      1.0402    73.2
    'x9'             0.27531      0.15965    0.44729    -0.60137       1.152    77.6
    'x10'            0.43749      0.36465    0.49671    -0.53605       1.411    88.2
    'x11'             2.6697       2.4471    0.97285     0.76287      4.5764     100
    'x12'           -0.29775     -0.19324    0.51413     -1.3054     0.70993    85.6
    'x13'           -0.69496     -0.59408    0.58651     -1.8445     0.45461    92.4
    'x14'          -0.035402            0    0.18405    -0.39614     0.32534    24.4
    'x15'            0.12551            0    0.57889     -1.0091      1.2601    75.4
    'x16'             2.6031       2.3375     1.1744     0.30134      4.9048     100
    'x17'           -0.58761      -0.4839    0.60248     -1.7685     0.59324      85
    'x18'           -0.13754    -0.013791    0.44043     -1.0008      0.7257    75.6
    'x19'             -1.431      -1.2922    0.77691     -2.9538    0.091725    99.6
    'x20'            0.74556      0.54392    0.81383    -0.84954      2.3407    88.6
```
Some statistical variation is expected in the results and the number of workers found might vary depending on the computer.
The execution is expected to take a few minutes on a desktop computer and approximately 20-30 minutes on a laptop.
Two figures will also prompt for the results of the cross-validation: 
- the confusion matrix
- the ROC curve

The model might be saved using the following code:
```matlab
save('model.mat');
```

If of interest then, the *sig_test.mat* file might be run by typing ```sig_test``` in the Command Window, in order to estimate predictions using only the most relevant predictors as resulted from the bootstrap procedure (see the [Instructions for Use](#Instructions_for_Use) section for further explanations). This script can be run only after the ```elastic_net``` script was run and the model saved. The script will output the results of the cross-validation in MATLAB Command Window, as well as the confusion matrix and the ROC curve of the model:
```matlab
>> sig_test
Sensitivity = 0.9800
Specificity = 0.9400
PPV = 0.9423
NPV = 0.9792
Balance Accuracy = 0.9600
Diagnostic Odd Ratio = 767.6667
AUC = 0.9964
```
