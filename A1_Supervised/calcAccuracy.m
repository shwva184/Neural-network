function [ acc ] = calcAccuracy( cM )
% CALCACCURACY Takes a confusion matrix amd calculates the accuracy

% Add your own code here
acc = 0;
acc = sum(diag(cM))/sum(sum(cM));

end

