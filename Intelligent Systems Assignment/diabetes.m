%load diabetes table data
diabetesData = readtable("diabetes.csv");
%get the actual data from diabetes
Y = diabetesData.Outcome;
%get the predicted data from tree
treeY = resubPredict(diabetesTree.ClassificationTree);
%get the predicted data from gaussian naive bayes
naiveBayesY = resubPredict(gausNaiveBayes.ClassificationNaiveBayes);
%get the predicted data from SVM
linearSVM_Y = resubPredict(linearSVM.ClassificationSVM);

%create a confusion matrix for tree
cm = confusionmat(Y, treeY);
%create a confusion matrix for gaussian naive bayes
gauscm = confusionmat(Y, naiveBayesY);
%create a confusion matrix for SVM
svmcm = confusionmat(Y, linearSVM_Y);

% ---- TREE CALUCLATIONS ----
%transpose matrix
cmt = cm';
%calculate precision
diagonal = diag(cmt);
row_sum = sum(cmt, 2);
precision = diagonal ./ row_sum;
overall_precision = mean(precision);
%calculate recall
column_sum = sum(cmt, 1);
recall = diagonal ./ column_sum';
overall_recall = mean(recall);
%calculate f1score
f1_score = 2*((overall_precision*overall_recall)/(overall_precision+overall_recall));
%calculate accuracy
all = sum(sum(cm));
accuracy1 = sum(diagonal) / all;

% ---- NAIVE BAYES CALCULATIONS ----
%transpose matrix
gauscmt = gauscm';
%calculate precision
diagonal2 = diag(gauscmt);
row_sum2 = sum(gauscmt, 2);
precision2 = diagonal2 ./ row_sum2;
overall_precision2 = mean(precision2);
%calculate recall
column_sum2 = sum(gauscmt, 1);
recall2 = diagonal2 ./ column_sum2';
overall_recall2 = mean(recall2);
%calculate f1score
f1_score2 = 2*((overall_precision2*overall_recall2)/(overall_precision2+overall_recall2));
%calculate accuracy
all2 = sum(sum(gauscm));
accuracy2 = sum(diagonal2) / all2;

% ---- SVM CALCULATIONS ----
%transpose matrix
svmcmt = svmcm';
%calculate precision
diagonal3 = diag(svmcmt);
row_sum3 = sum(svmcmt, 2);
precision3 = diagonal3 ./ row_sum3;
overall_precision3 = mean(precision3);
%calculate recall
column_sum3 = sum(svmcmt, 1);
recall3 = diagonal3 ./ column_sum3';
overall_recall3 = mean(recall3);
%calculate f1score
f1_score3 = 2*((overall_precision3*overall_recall3)/(overall_precision3+overall_recall3));
%calculate accuracy
all3 = sum(sum(svmcm));
accuracy3 = sum(diagonal3) / all3;

%add data to table
Models = ["Decision Tree";"Gaussian Naive Bayes";"Linear SVM"];
F1_scores = [f1_score; f1_score2; f1_score3];
Precisions = [precision; precision2; precision3];
Overall_Precisions = [overall_precision; overall_precision2; overall_precision3];
Overall_Recalls = [overall_recall; overall_recall2; overall_recall3];
Accuracy = [accuracy1; accuracy2; accuracy3];

dataTable = table(Models, F1_scores, Overall_Precisions, Overall_Recalls, Accuracy);

%display table
fig = uifigure;
uit = uitable(fig,"Data", dataTable, "Position", [20 20 350 300]);

%chart confusion matrix for tree
cmc = confusionchart(Y, treeY);
cmc.Title = "Confusion Matrix for Decision Tree";
%chart confusion matrix for gaussian naive bayes;
figure();
cmc2 = confusionchart(Y, naiveBayesY);
cmc2.Title = "Confusion Matrix for Gaussian Naive Bayes";
%chart confusion matrix for SVM
figure();
cmc3 = confusionchart(Y, linearSVM_Y);
cmc3.Title = "Confusion Matrix for Linear SVM";


