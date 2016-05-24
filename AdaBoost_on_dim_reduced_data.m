%Adaboost on dimensionally reduced dataset

%first we get the principal components
votes = csvread('votes.csv'); %Creates 435 by 17 array 
X = votes(:, 2:end); % Data : Voting record
N = size(X,1); %435
D = size(X,2); %16
cov_mat = cov(X);
[U S V] = svd(cov_mat);
PCs = U;
PC1 = PCs(:,1);
PC2 = PCs(:,2);
%------------%

training_data=csvread('training_data.csv');
testing_data=csvread('testing_data.csv');

democrat_indices=find(training_data(:,1)==1);
republican_indices=find(training_data(:,1)==-1);
X0_training=training_data(democrat_indices,:);
X1_training=training_data(republican_indices,:);
%removing the labels
X0_training=X0_training(:,2:end);
X1_training=X1_training(:,2:end);

%now we project the data to the 2 components

num_training_democrats=size(X0_training,1);
num_training_republicans=size(X1_training,1);
projected_X0_training=zeros(num_training_democrats,2);
projected_X1_training=zeros(num_training_republicans,2);
for i=1:num_training_democrats
    projected_X0_training(i,1)=X0_training(i,:)*PC1;
    projected_X0_training(i,2)=X0_training(i,:)*PC2;
end

for i=1:num_training_republicans
    projected_X1_training(i,1)=X1_training(i,:)*PC1;
    projected_X1_training(i,2)=X1_training(i,:)*PC2;
end


M=10
[params, weights] = boostlearn(projected_X0_training.', projected_X1_training.', M);

X=[projected_X0_training.',projected_X1_training.'];
C=boosteval(X,params,weights);

predicted_democrat_indices=find(C==1);
predicted_republican_indices=find(C==-1);

num_errors=sum(predicted_democrat_indices>216);
num_errors=num_errors+sum(predicted_republican_indices<217);
%num_errors=21
%training_accuracy=327/348=94.0%

%------------%
%now we test our model

democrat_testing_indices=find(testing_data(:,1)==1);
republican_testing_indices=find(testing_data(:,1)==-1);
X0_testing=testing_data(democrat_testing_indices,:);
X1_testing=testing_data(republican_testing_indices,:);
%removing the labels
X0_testing=X0_testing(:,2:end);
X1_testing=X1_testing(:,2:end);



%now we project the data to the 2 components

num_testing_democrats=size(X0_testing,1);
num_testing_republicans=size(X1_testing,1);
projected_X0_testing=zeros(num_testing_democrats,2);
projected_X1_testing=zeros(num_testing_republicans,2);
for i=1:num_testing_democrats
    projected_X0_testing(i,1)=X0_testing(i,:)*PC1;
    projected_X0_testing(i,2)=X0_testing(i,:)*PC2;
end

for i=1:num_testing_republicans
    projected_X1_testing(i,1)=X1_testing(i,:)*PC1;
    projected_X1_testing(i,2)=X1_testing(i,:)*PC2;
end

X_testing=[projected_X0_testing.',projected_X1_testing.'];
C_testing_labels=boosteval(X_testing,params,weights);

testing_predicted_democrat_indices=find(C_testing_labels==1);
testing_predicted_republican_indices=find(C_testing_labels==-1);

testing_num_errors=sum(testing_predicted_democrat_indices>51);
testing_num_errors=testing_num_errors+sum(testing_predicted_republican_indices<52);

%accuracy=78/87=89.7%






