function [bk] = cross_val_best_k(dataSetNr,k_choice)
%CROSS_VAL_BEST_K Summary of this function goes here
%dataSetNr is the dataset number 1,2,3 or 4 
%k_choice a set of numbers which are choices for best k

[X, D, L] = loadDataSet( dataSetNr );

%let us divide data into 3 partitions to perform cross validation on them
numBins = 3;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);
acc = 0;
best_acc = 0;
for i = k_choice
%training on 1,2 and test on 3
XTrain1 = combineBins(XBins, [1,2]);
XTest1 = combineBins(XBins, 3);
LTrain1 = combineBins(LBins, [1,2]);
LTest1 = combineBins(LBins,3);


LPredTest1  = kNN(XTest1 , i, XTrain1, LTrain1);

% The confucionMatrix
cM1 = calcConfusionMatrix(LPredTest1, LTest1);

% The accuracy
acc1 = calcAccuracy(cM1);

%training on 2,3 and test on 1
XTrain2 = combineBins(XBins, [2,3]);
XTest2 = combineBins(XBins, 1);
LTrain2 = combineBins(LBins, [2,3]);
LTest2 = combineBins(LBins,1);

LPredTest2  = kNN(XTest2 , i, XTrain2, LTrain2);

% The confucionMatrix
cM2 = calcConfusionMatrix(LPredTest2, LTest2);

% The accuracy
acc2 = calcAccuracy(cM2);

%training on 1,3 and test on 2
XTrain3 = combineBins(XBins, [1,3]);
XTest3 = combineBins(XBins, 2);
LTrain3 = combineBins(LBins, [1,3]);
LTest3 = combineBins(LBins,2);

LPredTest3  = kNN(XTest3 , i, XTrain3, LTrain3);

% The confucionMatrix
cM3 = calcConfusionMatrix(LPredTest3, LTest3);

% The accuracy
acc3 = calcAccuracy(cM3);

acc(i) = (acc1 + acc2 + acc3)/3;

if ( acc(i) > best_acc)
    best_acc = acc(i);
    bk = i;
end





end

plot(k_choice,acc)
title('Cross Validation')
xlabel('K')
ylabel('Accuracy')

end

