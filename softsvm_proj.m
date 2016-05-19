%% SOFT Support Vector Machine
%%  Learns an approximately separating hyperplane for the provided data.
%% Inputs - 
%%  X - Matrix with observations in each row.
%%  t - Vector of length equal to the number of columns of X, with either a 1 or -1 
%%     indicating the class label.
%%  gamma - Slack penalty parameter. Higher implies greater violation penalty.
%% Outputs - 
%%  w - Normal vector for the output hyperplane (plane equation is <w,x> + b = 0)
%%  b - Constant offset for the output hyperplane.

function [w, b] = softsvm_proj(X, t, gamma)

    N = size(X, 1);
    D = size(X, 2);
    
    H = spdiags([zeros(N,1); ones(D,1); 0], 0, N+D+1, N+D+1);
    size(H)
    
    f = [gamma*ones(N,1); zeros(D,1); 0];
    T = spdiags(t,0,N,N);
    
    eyeN = spdiags(ones(N,1), 0, N, N);
 
    
    mat2 = -1 * T * X;
   
    
    A = horzcat((-1*eyeN), mat2, -1*t);
    size (A)
    b_in = -1*(ones(N,1)); 
    lb = [zeros(N,1); -Inf*ones(D,1); -Inf];
    
    
    %% End of making parameters %%
 
    xi_w_b = quadprog(H, f, A, b_in, [], [], lb);
    size(xi_w_b)
    
    E = xi_w_b(1:N);
    w = xi_w_b(N+1:N+D);
    b = xi_w_b(N+D+1);

end