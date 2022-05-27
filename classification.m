clc;  % Clear command window.
clear;  % Delete all variables.
close all;  % Close all figure windows except those created by imtool.
imtool close all;  % Close all figure windows created by imtool.
workspace;  % Make sure the workspace panel is showing.

outputFolder =fullfile('segmented');
rootFolder = fullfile(outputFolder,'result');

categories={'Alstonia_Scholaris_(P2)','Arjun_(P1)','Basil_(P8)','Chinar_(P11)','Jamun_(P5)','Jatropha_(P6)','Lemon_(P10)','Mango_(P0)' ,'Pomegranate_(P9)','Pongamia_Pinnata_(P7)'};
%first arg is location of images- names, value
imds= imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');

CatgoryCount = countEachLabel(imds); % # of image in each category
minCount = min(CatgoryCount{:,2}); % the file that has least images

imds=splitEachLabel(imds,minCount,'randomize'); % equalizing # of images in each file

net = resnet50(); %pretrained model
%figure
%plot(net);
%title('Architecture of ResNet-50');
%set(gca,'YLim',[150 170]);
%net.Layers(1); %first layer input
%net.Layers(1); %properties of last layer
[trainset,testset]= splitEachLabel(imds,0.6,'randomize'); %splitting data set into train and test
imageSize = net.Layers(1).InputSize; %getting required image size
augmentedtrain=augmentedImageDatastore(imageSize,trainset, 'ColorPreprocessing','gray2rgb');
augmentedtest=augmentedImageDatastore(imageSize,testset, 'ColorPreprocessing','gray2rgb');
w1 = net.Layers(2).Weights; %getting second layer weights(first weight)
w1= mat2gray(w1);% converting matrix to grey scale image 

featureLayer ='fc1000'; %name of feature layer
trainingFeatures = activations(net,augmentedtrain,featureLayer,'MiniBatchSize',32,'OutputAs','columns'); %extracting feature
trainingLables = trainset.Labels;
classifier=fitcecoc(trainingFeatures,trainingLables,'Learner','Linear',...
    'Coding','onevsall','ObservationsIn','columns'); %to train SVM - returns full trained model multiclass error correction output codec model
testFeatures = activations(net,augmentedtest,featureLayer,'MiniBatchSize',32,'OutputAs','columns');
predictLabels= predict(classifier,testFeatures,'ObservationsIn','columns');% returns predicted class based on trained classifier
testLabels=testset.Labels;
confMat = confusionmat(testLabels,predictLabels);%confussion matrix
confMat= bsxfun(@rdivide, confMat, sum(confMat,2));

mean(diag(confMat));

newImage = imread('D:\term 7\Digital Image\materials\_Ground_Truth\_Ground_Truth_Masked\Chinar_(P11)\0011_0002.JPG');
figure;
imshow(newImage);
testimage=augmentedImageDatastore(imageSize,newImage,'ColorPreprocessing','gray2rgb');
tImageFeatures= activations(net,testimage,featureLayer,'MiniBatchSize',32,'OutputAs','columns');
imagelabel= predict(classifier,tImageFeatures,'ObservationsIn','columns');
sprintf('The uploaded image belongs to %s class', imagelabel)