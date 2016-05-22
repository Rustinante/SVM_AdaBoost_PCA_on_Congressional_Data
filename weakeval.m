%% WEAK learner EVALuate
%%  Uses a simple classifier with given parameters to classify data.
%% Inputs 
%%  X - Matrix with, in each column, an observation to assign labels.
%%  params - Parameters from weaklearn.m for a weak classifier.
%% Outputs
%%  C - Vector of class labels (1 or -1) for each input observation.

function [C] = weakeval(X, params)
    C = ((X'*params(1:(end-1)) + params(end)) > 0)*2 - 1;
end