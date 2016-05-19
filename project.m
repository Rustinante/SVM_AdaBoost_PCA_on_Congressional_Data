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


%% Do soft SVM on 80% and test on 20%
trainingX = csvread('training_data.csv');
testingX = csvread('testing_data.csv');

t_training = trainingX(:,1); % True labels of D or R

trainingX = trainingX(:, 2:end); % Data : Voting record
N_training = size(trainingX,1); 
D_training = size(trainingX,2); %16

%%[w_training,b_training] = softsvm_proj(trainingX, t_training, gamma);
b_training = 0.2648;
w_training = [
    0.0643;
    0.0258;
    0.2078;
   -0.4054;
   -0.1456;
    0.0166;
   -0.0133;
    0.0687;
    0.1113;
   -0.0379;
    0.1344;
   -0.1264;
   -0.0504;
   -0.0570;
    0.0613;
   -0.0032];

t_testing = testingX(:,1);
testingX = testingX(:, 2:end);
N_testing = size(testingX,1);
D_testing = size(testingX, 2);
testing_predictions = zeros(N_testing, 1);
num_correct = 0; %4  (4.6% incorrect)
num_incorrect = 0; %83 (95.4% correct)
for i = 1:N_testing
    data_point = testingX(i,:);
    temp = dot(w_training, data_point) + b_training;
    if (temp > 0)
        testing_predictions(i) = 1;
    else
        testing_predictions(i) = -1;
    end %if
    
    if(testing_predictions(i) == t_testing(i))
        num_correct = num_correct + 1;
    else
        num_incorrect = num_incorrect + 1;
    end %if
end %for_loop
 
 %% ================================================================%%
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
 
