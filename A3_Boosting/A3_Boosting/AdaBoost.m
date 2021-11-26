%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 100;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;
% Number of weak classifiers
nbrWeakClassifiers = 40;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;

faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

D = ones(1,size(xTrain,2))*1/size(xTrain,2);   % Initialisation of weights

for i = 1:nbrWeakClassifiers                   
    h = 0;
    polarities = 0;
    threshold = 0;
    minErr = inf;
    indice = 0;
    
    for j = 1:nbrHaarFeatures                   
         thresh = xTrain(j,:);
         
         for t = thresh                        
            p = 1;                             
            C = WeakClassifier(t, p, xTrain(j,:));
            E = WeakClassifierError(C, D, yTrain);
            
            if (E > 0.5)                        % If the error is greater than 0.5, reverse the polarity
                p = -1;
                E = 1 - E;
            end
            
            if (minErr > E)                     
                minErr = E;
                h = C * p;
                indice = j;
                threshold = t;
                polarities = p;
            end
        end
    end
    
    alpha = .5 * log((1-minErr) / minErr);
    
    D = D .* exp(-alpha * yTrain .* h);        % Update weights
    D = D/sum(D);                              % Normalize
    
    thresholds(i) = threshold;
    polarity(i) = polarities;
    indices(i) = indice;
    alphas(i) = alpha;
end

%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

classifierTest = zeros(nbrHaarFeatures,nbrTestImages);
classifierTrain = zeros(nbrHaarFeatures,nbrTrainImages);

accTest = zeros(1, nbrWeakClassifiers);
accTrain = zeros(1, nbrWeakClassifiers);

for k = 1 : nbrWeakClassifiers
    classifierTest(k,:) = alphas(k) * WeakClassifier(thresholds(k),polarity(k),xTest(indices(k),:));
    accTest(k) = sum(sign(sum(classifierTest)) == yTest) / nbrTestImages;
    
    classifierTrain(k,:) = alphas(k) * WeakClassifier(thresholds(k),polarity(k),xTrain(indices(k),:));
    accTrain(k) = sum(sign(sum(classifierTrain)) == yTrain) / nbrTrainImages;
end

TrainAccuracy = accTrain(nbrWeakClassifiers)
TestAccuracy = accTest(nbrWeakClassifiers)

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

figure(4);
plot(accTest);
hold on
plot(accTrain);
legend({'test','train'}, 'Location', 'southeast')
xlabel('Number of weak classifiers')
ylabel('Accuracy')
hold off


%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

result = sign(sum(classifierTest));
missclass_faces = find(yTest == 1 & yTest ~= result);
missclass_nonfaces = find(yTest == -1 & yTest ~= result);

figure(6);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(testImages(:,:,missclass_faces(k)));
    axis image;
    axis off;
end

figure(7);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(testImages(:,:,missclass_nonfaces(k)));
    axis image;
    axis off;
end

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

figure(8);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,indices(k)),[-1 2]);
    axis image;
    axis off;
end
