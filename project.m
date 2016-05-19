% -1 = Republican
%  1 = Democrat
%  1 = Yes
% -1 = No
%  0 = ?
votes = csvread('votes.csv'); %Creates 435 by 17 array 


t = votes(:,1); % True labels of D or R

X = votes(:, 2:end); % Data : Voting record
N = size(X,1); %435
D = size(X,2); %16
gamma = 0.005;
%[w,b] = softsvm_proj(X, t, gamma);
w = [0.0762;
    0.0118;
    0.1909;
   -0.4809;
   -0.1359;
    0.0151;
   -0.0353;
    0.0447;
    0.1054;
   -0.0326;
    0.1320;
   -0.1404;
   -0.0518;
   -0.0623;
    0.0557;
   -0.0269 ];
b = 0.2675;

% Most and least distinguishable vote for classification
w_square = w.^2;
[max_val_vote max_ind_vote] = max(w_square); %4=physician_fee_freeze
[min_val_vote min_ind_vote] = min(w_square); %2=water-project-cost-sharing
% These results make snse if you look at the probabilities for them from 
% data description online.

[min_val_person, min_index_person] = min(x_axis); %303 --> Most Republican
[max_val_person, max_index_person] = max(x_axis); %106 --> Most Democratic

x_axis = X * w + b;
figure
plot(t, x_axis, 'b*')
title('Classification')
xlabel('t, the correct labels')
ylabel('X*w +b')


R_indices = find(t==-1);
D_indices = find(t== 1);

incorrectNumR = size(find(x_axis(R_indices) > 0)); %Number of incorrectly classified R = 8
incorrectNumD = size(find(x_axis(D_indices) < 0)); %Number of incorrectly classified D = 16



 
 
 %%PCA on data
 cov_mat = cov(X);
 [U S V] = svd(cov_mat);
 PCs = U;
 PC1 = PCs(:,1);
 PC2 = PCs(:,2);
 
 % Project data onto first two PCs
 x_axis = zeros(N,1);
 y_axis = zeros(N,1);
 for i = 1:N
     data_point = X(i, :);
     x_axis(i) = dot(data_point, PC1);
     y_axis(i) = dot(data_point, PC2);
     
 end %for_loop
 
 figure
 hold on
 scatter(x_axis(R_indices), y_axis(R_indices), 'r*')
 scatter(x_axis(D_indices), y_axis(D_indices), 'b*')
 title('PCA')
 xlabel('PC1')
 ylabel('PC2')
 hold off
 
