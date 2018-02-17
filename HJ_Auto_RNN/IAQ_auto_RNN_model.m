clear all; close all; clc;
disp('트래이닝 수행(1) or 트래이닝 완료(2)')
flag = input('숫자를 입력하시오: ');

load data_filter_Original.mat
%% Variable
Platform_PM10 = data2(:,2); % PM10 conc. at platform
Outdoor_PM10 = data2(:,3);  % PM10 conc. at outdoor
RPM = data2(:,4);           % Ventilation fan speed
Subway = data2(:,5);        % Number of trains

figure(1);
set(1,'color','white');
subplot(2,2,1); plot(data2(:,1),Platform_PM10); xlabel('Time(day)'); ylabel('PM_{10} conc. at platform (ppm)');
subplot(2,2,2); plot(data2(:,1),Outdoor_PM10); xlabel('Time(day)'); ylabel('PM_{10} conc. at outdoor (ppm)');
subplot(2,2,3); plot(data2(:,1),RPM); xlabel('Time(day)'); ylabel('Ventilation fan speed (RPM)');
subplot(2,2,4); plot(data2(:,1),Subway); xlabel('Time(day)'); ylabel('Subway schedule');

%% Outlier detection
Platform_PM10_filt = filt_Olsson( Platform_PM10, 0.1, 0.3 );
outlier_index = outlier( Platform_PM10-Platform_PM10_filt );
filtered_Platform_PM10 = Platform_PM10;

figure(2)
set(2,'color','white')
subplot(2,1,1); plot(data2(:,1),Platform_PM10 - Platform_PM10_filt); hold on;
for i = 1:length(outlier_index)
    filtered_Platform_PM10(outlier_index(i)) = Platform_PM10_filt(outlier_index(i));
    plot(data2(outlier_index(i),1),Platform_PM10(outlier_index(i)) - Platform_PM10_filt(outlier_index(i)),'rs');
    text(data2(outlier_index(i)+30,1),Platform_PM10(outlier_index(i)) - Platform_PM10_filt(outlier_index(i)),['Sample number: ' num2str(outlier_index(i))]);
end
hold off;
xlabel('Time(day)'); ylabel('Data variation');

subplot(2,1,2); 
plot(data2(:,1),Platform_PM10,'Color',[155/255, 155/255, 155/255]); hold on; 
plot(data2(:,1),filtered_Platform_PM10,'-r'); hold off;
xlabel('Time(day)'); ylabel('PM_{10} conc. at platform(ppm)');
legend('Measured data','Filtered data');



%% Normalization
[ax_input_outPM,mx_input_outPM,stdx_input_outPM] = auto(Outdoor_PM10);
[ax_input_RPM,mx_input_RPM,stdx_input_RPM] = auto(RPM);
[ax_input_Subway,mx_input_Subway,stdx_input_Subway] = auto(Subway);
[ax_output_platPM,mx_output_platPM,stdx_output_platPM] = auto(filtered_Platform_PM10);

figure(3);
set(3,'color','white');
subplot(2,2,1); plot(data2(:,1),ax_output_platPM); xlabel('Time(day)'); ylabel('PM_{10} conc. at platform (ppm)');
subplot(2,2,2); plot(data2(:,1),ax_input_outPM); xlabel('Time(day)'); ylabel('PM_{10} conc. at outdoor (ppm)');
subplot(2,2,3); plot(data2(:,1),ax_input_RPM); xlabel('Time(day)'); ylabel('Ventilation fan speed (RPM)');
subplot(2,2,4); plot(data2(:,1),ax_input_Subway); xlabel('Time(day)'); ylabel('Subway schedule');

%% Cross correlation
[ax_input_RPM_r,ax_input_RPM_lag,ax_input_RPM_limit] = cross_correlation(ax_input_RPM,ax_output_platPM);
[ax_input_Subway_r,ax_input_Subway_lag,ax_input_Subway_limit] = cross_correlation(ax_input_Subway,ax_output_platPM);
[ax_input_outPM_r,ax_input_outPM_lag,ax_input_outPM_limit] = cross_correlation(ax_input_outPM,ax_output_platPM);
%% Auto correlation
[ax_output_platPM_r,ax_output_platPM_lag,ax_output_platPM_limit] = auto_correlation(ax_output_platPM);

max_flag = 480; % 2400 data / 5days = 480 
                % max([ax_input_RPM_lag ax_input_Subway_lag ax_input_outPM_lag ax_output_platPM_lag])/10;

train_y = ax_output_platPM(max_flag + 1 :1440);
train_x1 = [];
train_x2 = [];
train_x3 = [];
train_x4 = [];
for i = 1:max_flag
    train_x1 = [train_x1 ax_input_RPM(i:length(train_y)+(i-1))];
    train_x2 = [train_x2 ax_input_Subway(i:length(train_y)+(i-1))];
    train_x3 = [train_x3 ax_input_outPM(i:length(train_y)+(i-1))];
    train_x4 = [train_x4 ax_output_platPM(i:length(train_y)+(i-1))];
end

test_y = ax_output_platPM(max_flag + 1 + 1440-480 :2400);
test_x1 = [];
test_x2 = [];
test_x3 = [];
test_x4 = [];
for i = 1:max_flag
    test_x1 = [test_x1 ax_input_RPM(i+1440-480:length(test_y)+(i-1)+1440-480)];
    test_x2 = [test_x2 ax_input_Subway(i+1440-480:length(test_y)+(i-1)+1440-480)];
    test_x3 = [test_x3 ax_input_outPM(i+1440-480:length(test_y)+(i-1)+1440-480)];
    test_x4 = [test_x4 ax_output_platPM(i+1440-480:length(test_y)+(i-1)+1440-480)];
end


csvwrite('./data/data_1.csv',train_x1);
csvwrite('./data/data_2.csv',train_x2);
csvwrite('./data/data_3.csv',train_x3);
csvwrite('./data/data_4.csv',train_x4);
csvwrite('./data/data_y.csv',train_y);

csvwrite('./data/test_1.csv',test_x1);
csvwrite('./data/test_2.csv',test_x2);
csvwrite('./data/test_3.csv',test_x3);
csvwrite('./data/test_4.csv',test_x4);
csvwrite('./data/test_y.csv',test_y);

%% Execution simple RNN
if flag == 1
    dos('python IAQ_RNN.py');
else
    dos('python IAQ_RNN_restore.py');
end

%% Prediction result visualization
load train_y_out.mat;
load test_y_out.mat;

re_train_y = auto_return(train_y,mx_output_platPM,stdx_output_platPM);
re_test_y = auto_return(test_y,mx_output_platPM,stdx_output_platPM);
re_train_y_out = auto_return(train_y_out,mx_output_platPM,stdx_output_platPM);
re_test_y_out = auto_return(test_y_out,mx_output_platPM,stdx_output_platPM);

% Draw figure
figure(4);
set(4,'color','white');
plot(data2(max_flag+1:1440),re_train_y,'b.'); hold on; 
plot(data2(max_flag+1+1440-480:2400),re_test_y,'g.');
plot(data2(max_flag+1:1440),re_train_y_out,'r-'); 
plot(data2(max_flag+1+1440-480:2400),re_test_y_out,'m-'); hold off;
xlabel('Time(day)'); ylabel('PM_{10} conc. at platform (ppm)');
legend('Train data','Test data','Prediction using train data','Prediction using test data');

% Root mean squre error
train_rmse = sqrt(mean((re_train_y-re_train_y_out).^2));
test_rmse = sqrt(mean((re_test_y-re_test_y_out).^2));

disp('================================');
disp('|     Train     |     Test     |')
disp('--------------------------------');
disp(['|       ' num2str(train_rmse) ' |      ' num2str(test_rmse) ' |']);
disp('================================');

% R square
y_bar = sum(re_train_y_out)/length(re_train_y_out);
ss_tot = sum((re_train_y_out-y_bar).^2);
ss_res = sum((re_train_y-re_train_y_out).^2);
r2 = 1 - ss_res/ss_tot;
figure(5);
set(5,'color','white');
plot(re_train_y,re_train_y_out,'.');  hold on;
plot(re_test_y,re_test_y_out,'*'); hold off;
line([20,180],[20,180],'Color','Black'); grid on;
xlabel('Measured PM_{10} conc. at platform');
ylabel('Predicted PM_{10} conc. at platform');
legend('Training data','Test data');
