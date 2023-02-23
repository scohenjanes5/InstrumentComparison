%% Open graphical tool
deepNetworkDesigner

%% once trained and exported, classify input image
%import image
I = imread("MerchDataTest.jpg");
%resize to match network size
I = imresize(I, [224 224]);
% classify
[YPred,probs] = classify(trainedNetwork_1,I);
imshow(I)
label = YPred;
title(string(label) + ", " + num2str(100*max(probs),3) + "%");


