%% WEAK LEARNer
%%  Trains a simple classifier which achieves at least 50 percent accuracy.
%% Inputs 
%%  X0 - Matrix with, in each column, an observation from class 1.
%%  X1 - Matrix with, in each column, an observation from class -1.
%%  W0, W1 - (Optional) Column vectors with data weights. Length must
%%    correspond to X0 and X1. Defaults to uniform weights.
%% Outputs
%%  params - Parameters for the weak trained model (column vector).

function [params] = weaklearn(X0, X1, W0, W1)
    if nargin == 2
        W0 = ones(size(X0,2),1);
        W1 = ones(size(X1,2),1);
    end
    
    best_d = 1;
    best_x = 0;
    best_err = inf;
    is_01 = 1;
    
    X = [X0, X1];
    W = [-W0; W1];
    for d = 1:size(X0,1)
        [~,IX] = sort(X(d,:));
        
        err = cumsum(W(IX));
        [min_cum,min_k] = min(err);
        best_01 = sum(W0) + min_cum;
        best_01_x = X(d,IX(min_k));
        
        err = cumsum(-W(IX));
        [min_cum,min_k] = min(err);
        best_10 = sum(W1) + min_cum;
        best_10_x = X(d,IX(min_k));
       
        if best_01 < best_err
            best_d = d;
            best_x = best_01_x;
            best_err = best_01;
            is_01 = 1;
        end
        
        if best_10 < best_err
            best_d = d;
            best_x = best_10_x;
            best_err = best_10;
            is_01 = 0;
        end
    end
    
    beta = zeros(size(X0,1),1);
    beta(best_d) = 1;
    params = [beta;-best_x];
    if is_01
        params = -params;
    end
end