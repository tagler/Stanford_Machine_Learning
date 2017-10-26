function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
values = [ 0.1 0.2 0.3 0.5 1 ] ;
data = zeros(size(values)(2)*size(values)(2),3);
k = 1;

for i = 1:size(values)(2)
    for j = 1:size(values)(2)
    
        model = svmTrain(X, y, values(i), @(x1, x2) gaussianKernel(x1, x2, values(j) )); 

        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));

        data(k,1) = prediction_error;
        data(k,2) = values(i);
        data(k,3) = values(j);
        k = k+1
    end
end

[a,b] = min( data(:,1) )
C = data(b,2);
sigma = data(b,3);

% =========================================================================

end
