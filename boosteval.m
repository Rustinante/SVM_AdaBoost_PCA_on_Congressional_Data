%% adaBOOST model EVALuator
%%  Uses a trained AdaBoost algorithm to classify data.
%% Inputs 
%%  X - Matrix with observations (in columns) to classify.
%%  params - Output of boostlearn.m (weak learner parameters).
%%  weights - Output of boostlearn.m (weak learner mixing coefficients).
%% Outputs
%%  C - A matrix with predicted class labels (-1 or 1) for the input
%%    observations in X.


function [C] = boosteval(X, params, weights)
    [nrow,ncol]=size(X);
    [M,N]=size(weights);
    C=zeros(ncol,1);
    for i=1:ncol
        sum=0;
        for m=1:M
            prediction=weakeval(X(:,i),params(:,m));
            sum=sum+weights(m,1)*prediction;
        end
        C(i,1)=sign(sum);
    end
end

















