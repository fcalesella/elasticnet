function [measures, figures] = metricsnplots(reg_type, cv)
% Compute accuracy measures and create the plots.
% Inputs:
%   reg_type: type of regression. If logistic regression has been performed
%       pass 'binomial', otherwise 'normal' if linear regression was
%       performed.
%   cv: structure array containing the test (field Tests) and predicted 
%       (field Preds) observations. If logistic regression was performed 
%       and the AUC field is present, than it assumed that the structure 
%       array contains the results of the cross-validation. If the AUC 
%       field is not present, than it is assumed that the structure array 
%       contains the results of the final prediction. In this last case, a
%       field with the continous predictions (field PredsContinuous,
%       containing the non-binarized predictions) is needed.
% Outputs:
%   measures: structure array containing all the accuracy measures
%       computed.
%   figures: structure array containing all the plots created.

if strcmp(reg_type, 'binomial')
    
    % For classification.
    
    % Compute confusion matrix and the derived measures (true poitives, 
    % true negatives, false negatives, and false positives).
    [~,cm,~,~] = confusion(cv.Tests, cv.Preds);
    TP = cm(2,2);
    TN = cm(1,1);
    FN = cm(2,1);
    FP = cm(1,2);

    % Model sensitivity.
    Sensitivity = TP/(TP + FN);
    fprintf('Sensitivity = %.4f\n', Sensitivity)
    measures.Sensitivity = Sensitivity;

    % Model specificity. 
    Specificity = TN/(TN+FP);
    fprintf('Specificity = %.4f\n', Specificity)
    measures.Specificity = Specificity;

    % Model PPV.
    PPV = TP/(TP+FP);
    fprintf('PPV = %.4f\n', PPV)
    measures.PPV = PPV;
    % Model NPV.
    NPV = TN/(TN+FN);
    fprintf('NPV = %.4f\n', NPV)
    measures.NPV = NPV;

    % Balance Accuracy.
    BalanceAccuracy = (Sensitivity + Specificity)/2;
    fprintf('Balance Accuracy = %.4f\n', BalanceAccuracy)
    measures.BalanceAccuracy = BalanceAccuracy;

    % Diagnostic Odd Ratio.
    dorn = Specificity*Sensitivity;
    dord = (1 - Specificity) * (1 - Sensitivity);
    DOR = dorn/dord;
    fprintf('Diagnostic Odd Ratio = %.4f\n', DOR)
    measures.DOR = DOR;

    % Plot confusion matrix for accuracy.
    f1 = figure;
    plotconfusion(cv.Tests, cv.Preds)
    figures.f1 = f1;
    
    if isfield(cv, 'AUC')
        
        k = length(cv.AUC);
        % Mean and standard deviation of the  AUC across cross-validation.
        fprintf('AUC across cross-validations:\n')
        disp(cv.AUC)
        auc_mean = mean(cv.AUC); 
        auc_sd = std(cv.AUC);
        fprintf('AUC statistics:\nMean = %.4f\nS.D. = %.4f\n\n', auc_mean, auc_sd)
        measures.AUC = cv.AUC;
        measures.AUC_Mean = auc_mean;
        measures.AUC_SD = auc_sd;
        
        % Mean of folds (coordinate of ROC curve).
        XmeanROC = foldmean (cv.XFold);
        YmeanROC = foldmean (cv.YFold);
        % Plot ROC.
        f2 = figure;
        legend_var = plot_roc(cv.XFold, cv.YFold, k);
        plot(XmeanROC,YmeanROC, 'r ', 'LineWidth', 2)
        xlabel('False positive rate'); 
        ylabel('True positive rate');
        title_var = sprintf('ROC for %i folds Cross Validation', k);
        title(title_var)
        legend(legend_var, 'Mean across Folds')
        hold off
        figures.f2 = f2;
        
    else
        
        % If the AUC field is not present in cv, calculate it and plot the
        % ROC.
        f2= figure;
        [xfold,yfold,~,auc] = perfcurve(cv.Tests,cv.PredsContinuous,'1');
        plot(xfold,yfold,'r ', 'LineWidth', 2);
        xlabel('False positive rate'); ylabel('True positive rate');
        title('ROC Curves of bootstrap coefficients')
        fprintf('AUC = %.4f\n', auc);
        measures.AUC = auc;
        figures.f2 = f2;
        
    end
    
else
    
    % For linear regression.
    
    % Model accuracy with MSE.
    MSE = mean((cv.Preds - cv.Tests).^2);
    fprintf('MSE = %.4f\n', MSE)
    measures.MSE = MSE;

    % Model accuracy with RMSE.
    RMSE = sqrt(MSE);
    fprintf('RMSE = %.4f\n', RMSE)
    measures.RMSE = RMSE;

    % Model accuracy with R square.
    sst = sum((cv.Tests - mean(cv.Tests)).^2);
    sse = sum((cv.Preds - cv.Tests).^2);
    Rsq = 1 - sse / sst;
    fprintf('R squared = %.4f\n', Rsq)
    measures.Rsq = Rsq;
    
    % Plot predictions.
    f1 = figure();
    f1(1) = scatter(cv.Tests, cv.Preds, 'filled', 'MarkerFaceColor', 'k');
    f1(2) = lsline;
    f1(2).Color = 'k';
    xlabel('True value')
    ylabel('Prediciton')
    figures.f1 = f1;
    
end

end