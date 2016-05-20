%% adaBOOST model LEARNer
%%  Uses the AdaBoost algorithm to train a classifier on data.
%% Inputs 
%%  X0 - Observations from the first data class.
%%  X1 - Observations from the second data class.
%%  T - The number of weak learners to include in the ensemble.
%% Outputs
%%  params - A matrix containing the parameters for the T weak learners.
%%  weights - A vector of weights used to combine the results of the
%%    T weak learners.




function [params, weights] = boostlearn(X0, X1, M)
    [X0_num_row,X0_num_col]=size(X0);
    [X1_num_row,X1_num_col]=size(X1);
    N=X0_num_col+X1_num_col;
    W0=ones(X0_num_col,1)/N;
    W1=ones(X1_num_col,1)/N;
    params=zeros(X0_num_row+1,M);
    weights=zeros(M,1);
    for i=1:M
        params(:,i)=weaklearn(X0, X1, W0, W1);
        err=0;
        C0=weakeval(X0,params(:,i));
        for j=1:X0_num_col
            if(C0(j,1)==-1)
                err=err+W0(j,1);
            end
        end
        C1=weakeval(X1,params(:,i));
        for j=1:X1_num_col
            if(C1(j,1)==1)
                err=err+W1(j,1);
            end
        end
        
        alpha=0.5*log((1-err)/err);
        weights(i,1)=alpha;
        
        for j=1:X0_num_col
            W0(j,1)=W0(j,1)*exp(-1*weakeval(X0(:,j),params(:,i))*alpha);
        end
        
        for j=1:X1_num_col
            W1(j,1)=W1(j,1)*exp(weakeval(X1(:,j),params(:,i))*alpha);
        end
        
        normalization_factor=sum(W0);
        normalization_factor=normalization_factor+sum(W1);
        
        W0=W0/normalization_factor;
        W1=W1/normalization_factor;
        
    end
    
end






