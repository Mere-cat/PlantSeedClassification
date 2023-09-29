% 1. Read all images
dataSet = imageDatastore("plant-seedlings-classification/train/", "IncludeSubfolders", true, "LabelSource", "foldernames", "FileExtensions", ".png");
testSet = imageDatastore("plant-seedlings-classification/test/", "IncludeSubfolders", true, "LabelSource", "foldernames", "FileExtensions", ".png");
dataNum = 4750;
testNum = 794;
disp("[Done] Read all images.");

% 2. Find training accuracy
% 2.1 split dataSet
[trainSet, valSet] = splitEachLabel(dataSet, 0.5, "randomized");
trainNum = numel(trainSet.Files);
valNum = numel(valSet.Files);
disp("[Done] Split data set.");

% 2.2 extract feature using LBP, and store the features
trainFeature = zeros(trainNum, 59);
for i=1:trainNum
     tmp = readimage(trainSet,i);
     tmp = rgb2gray(tmp);
     features = extractLBPFeatures(tmp);
     trainFeature(i,:) = features;
end


valFeature = zeros(valNum, 59);
for i=1:valNum
     tmp = readimage(valSet,i);
     tmp = rgb2gray(tmp);
     features = extractLBPFeatures(tmp);
     valFeature(i,:) = features;
end

disp("[Done] Extract features.");

% 2.3 nn search
resIdx = knnsearch(trainFeature, valFeature);
disp("[Done] nn search.")

% 2.4 caculate the accuracy for NN method
matchNum = min(trainNum, valNum);
correctCnt = 0;
for i=1:matchNum
    if (trainSet.Labels(resIdx(i)) == valSet.Labels(i))
        correctCnt = correctCnt + 1;
    end
end
acc = correctCnt / matchNum;
disp("LBP SSD acc:" + acc);
disp("[Done] Claculate training accuracy.")
disp("=====================================");

% 3. Find testing accuracy
% 3.1 extract feature from dataSet and trestSet, and store the features
allTrainFeature = zeros(dataNum, 59);
for i=1:dataNum
     tmp = readimage(dataSet,i);
     tmp = rgb2gray(tmp);
     features = extractLBPFeatures(tmp);
     allTrainFeature(i,:) = features;
end

testFeature = zeros(testNum, 59);
for i=1:testNum
     tmp = readimage(testSet,i);
     tmp = rgb2gray(tmp);
     features = extractLBPFeatures(tmp);
     testFeature(i,:) = features;
end

disp("[Done] Extract features.")

% 3.2 nn search
resIdx = knnsearch(allTrainFeature, testFeature);
disp("[Done] nn search.")

% 3.3 output classification result as .csv file
output = strings(testNum, 2);
output(1, 1) = "file";
output(1, 2) = "species";
for i=1:testNum
    [img,info] = readimage(testSet, i);
    [filepath,name,ext] = fileparts(info.Filename);
    fileName = string(name) + string(ext);
    output(i+1, 1) = fileName;
    output(i+1, 2) = dataSet.Labels(resIdx(i));
end
writematrix(output, "submission.csv");