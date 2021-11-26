function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);
dist = zeros(size(XTrain,1),1);

for i = 1:size(X,1)
    for j = 1:size(XTrain,1)
        dist(j) = sqrt(sum((X(i,:)-XTrain(j,:)).^2));
    end
    
    %now let us sort the distances obtained 
    [~,dist_order] = sort(dist);
    top_k_label = [LTrain(dist_order), sort(dist)];
    
    %Now let us keep only k neighbouring labels
    
    top_k_label = top_k_label(1:k,:);
    
    %now let us vote the classes 
    count = [classes, zeros(NClasses,1), zeros(NClasses,1)];
    for c = 1:NClasses
        for b = 1:k
            if (top_k_label(b,1) == classes(c))
                count(c,2) = count(c,2) + 1 ;
                count(c,3) = count(c,3) + top_k_label(b,2);
            end
        end
    end
    %now we will sort descending on votes , bringing the top voted classes
    %above , next we sort ascendingly on distance to break the ties that
    %might occur 
    count = sortrows(count,[-2,3]);
    LPred(i) = count(1,1);
end

    
    
    
    

end

