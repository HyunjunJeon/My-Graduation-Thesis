clear all; close all; clc;
disp('트래이닝 수행(1) or 트래이닝 완료(2)')
flag = input('숫자를 입력하시오: ');  

load data_filter_Original.mat;
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
plot(data2(:,1),Platform_PM10,'Color',[153/255, 153/255, 153/255]); hold on; 
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

%% Data partition
train_x = [ax_input_outPM(1:1440) ax_input_RPM(1:1440) ax_input_Subway(1:1440)];
train_y = [ax_output_platPM(1:1440)];
test_x = [ax_input_outPM(1441:2400) ax_input_RPM(1441:2400) ax_input_Subway(1441:2400)];
test_y = [ax_output_platPM(1441:2400)];
save('train_x.mat','train_x');
save('train_y.mat','train_y');
save('test_x.mat','test_x');
save('test_y.mat','test_y');

%% Execution simple DNN
if flag == 1
    dos('python IAQ_DNN.py');
else
    dos('python IAQ_DNN_restore.py');
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
plot(data2(1:1440),re_train_y,'b.'); hold on; 
plot(data2(1441:2400),re_test_y,'g.');
plot(data2(1:1440),re_train_y_out,'r-'); 
plot(data2(1441:2400),re_test_y_out,'m-'); hold off;
xlabel('Time(day)'); ylabel('PM_{10} conc. at platform (ppm)');
legend('Train data','Test data','Prediction using train data','Prediction using test data');

% Root mean squre error (RMSE)
train_rmse = sqrt(mean((re_train_y-re_train_y_out).^2));
test_rmse = sqrt(mean((re_test_y-re_test_y_out).^2));

disp('================================');
disp('|     Train     |     Test     |')
disp('--------------------------------');
disp(['|       ' num2str(train_rmse) ' |      ' num2str(test_rmse) ' |']);
disp('================================');
