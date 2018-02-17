
%% 3 columns
clear all; clc; close all;
load train_x.mat

data1 = [train_x(1:1438,1) train_x(2:1439,1) train_x(3:1440,1)];
data2 = [train_x(1:1438,2) train_x(2:1439,2) train_x(3:1440,2)];
data3 = [train_x(1:1438,3) train_x(2:1439,3) train_x(3:1440,3)];

csvwrite('data_1.csv',data1);
csvwrite('data_2.csv',data2);
csvwrite('data_3.csv',data3);

load train_y.mat
data_y = train_y(3:1440);
csvwrite('data_y.csv',data_y);

load test_x.mat
data1 = [train_x(1:958,1) train_x(2:959,1) train_x(3:960,1)];
data2 = [train_x(1:958,2) train_x(2:959,2) train_x(3:960,2)];
data3 = [train_x(1:958,3) train_x(2:959,3) train_x(3:960,3)];

csvwrite('test_1.csv',data1);
csvwrite('test_2.csv',data2);
csvwrite('test_3.csv',data3);

load test_y.mat
data_y = train_y(3:960);
csvwrite('test_y.csv',data_y);

%% 1 column
clear all; clc; close all;
load train_x.mat
csvwrite('data_1.csv',train_x(:,1));
csvwrite('data_2.csv',train_x(:,2));
csvwrite('data_3.csv',train_x(:,3));

load train_y.mat
csvwrite('data_y.csv',train_y);

load test_x.mat
csvwrite('test_1.csv',test_x(:,1));
csvwrite('test_2.csv',test_x(:,2));
csvwrite('test_3.csv',test_x(:,3));

load test_y.mat
csvwrite('test_y.csv',test_y);



