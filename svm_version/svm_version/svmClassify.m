function testLabels = svmClassify(trainData,trainLabels,testData)
    % get unique labels and number of classes
    uniqueLabels = unique(trainLabels);
    numClasses = numel(uniqueLabels);

    % train SVM for each class (using one-vs-all approach)
    SVMModel = cell(numClasses,1);
    for i = 1:numClasses
        disp('Training 1 - 17')
        disp(i)
        currentClass = (trainLabels==uniqueLabels(i));
        SVMModel{i} = fitcsvm(trainData,currentClass,...
            'KernelFunction','polynomial',...
            'Standardize',true,...
            'ClassNames',[false,true],...
            'KernelScale','auto');
    end
    
    % classify test data
    score = zeros(size(testData,1),numClasses);
    for i = 1:numClasses
        disp('Classifying 1 - 17');
        disp(i)
        [~,tempScores] = predict(SVMModel{i},testData);
        score(:,i) = tempScores(:,2);
    end
    [~,testLabels] = max(score,[],2);

end