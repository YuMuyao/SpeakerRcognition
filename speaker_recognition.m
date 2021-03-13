%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%本代码适用于单人的说话人确认
clear all;
close all;
MFCC_size=12;%mfcc的维数
GMMM_component=16;%GMM component 个数

mu_model=zeros(MFCC_size,GMMM_component);%高斯模型 分量 均值（返回一个12*16的零矩阵）
sigma_model=zeros(MFCC_size,GMMM_component);%高斯模型 分量 方差
weight_model=zeros(GMMM_component);%高斯模型 分量 权重（返回一个16*16的零矩阵）

train_file_path='.\training\';%模型训练文件路径
test_file_path='.\testing\';%测试文件路径


all_train_feature=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%train model
FileList=dir(train_file_path);%读取该路径下的所有文件
model_num=1;%注册模型的个数
error_num=0;%识别错误的个数

T = clock; 
disp([num2str(T(4)),':',num2str(T(5)),':',num2str(T(6))]);
%FXY-01################
correct_num=0;
ubm_y=[];
ubm_fs=[];
ubm_file_directory = '.\ubm_file\'; %address of training audio files
ubm_all_files = dir([ubm_file_directory '*.wav']);
for ubm_i=1:length(ubm_all_files)
    [ubm_y_temp, ubm_fs_temp] = audioread(sprintf('%s%s', ubm_file_directory, ubm_all_files(ubm_i).name));
    ubm_y_temp = ubm_y_temp(1:900000);% Taking minimum of all files to have same input length
    ubm_fs(ubm_i,1)=ubm_fs_temp;
    ubm_y(:,ubm_i)=ubm_y_temp;
end

%%
ubm_nSpeakers = 4; %number of speakers
ubm_nDims = 12; % dimensionality of feature vectors
ubm_nMixtures = 16; % How many mixtures used to generate data
ubm_nChannels = 1; % Number of channels (sessions) per speaker
ubm_nFrames = 1000; % Frames per speaker (10 seconds assuming 100 Hz)
ubm_nWorkers = 2; % Number of parfor workers, if available
ubm_final_niter = 15;
ubm_ds_factor = 1;
rng('default'); % To promot reproducibility.
ubm_mfccs1=[];
for ubm_i=1:ubm_nSpeakers
%     display(i);
    ubm_mfccs1(:,:,ubm_i) = melcepst(ubm_y(:,ubm_i), ubm_fs(ubm_i));
end

ubm_mfccsdata=cell(ubm_nSpeakers,ubm_nChannels);

for ubm_j=1:ubm_nSpeakers
    for ubm_i=1:ubm_nChannels
        ubm_mfccsdata{ubm_j,ubm_i}=(ubm_mfccs1((ubm_i*(ubm_nFrames)-(ubm_nFrames-1)):ubm_i*ubm_nFrames,:,ubm_j))';
        ubm_speakerID(ubm_j,ubm_i) = ubm_j;
    end
end
ubm_trainSpeakerData=ubm_mfccsdata;
ubm = gmm_em(ubm_trainSpeakerData(:), ubm_nMixtures, ubm_final_niter, ubm_ds_factor, ...
ubm_nWorkers);
%ubm = gmm_em('ubm.lst',16, 15, 1,8);尝试list方式调用微软的函数失败,要基于HTK平台提取MFCC参数（故尝试用cell方式传入2）
%FXY-01################
T = clock; 
disp([num2str(T(4)),':',num2str(T(5)),':',num2str(T(6))]);

%该路径下是否是文件夹
for i=1:length(FileList)
    if(FileList(i).isdir==1&&~strcmp(FileList(i).name,'.')&&~strcmp(FileList(i).name,'..'))
        all_model_name{model_num,1}=FileList(i).name;%存储模型名称
        fprintf('Train:%s\n',all_model_name{model_num,1});
        one_train_file_path=[train_file_path  all_model_name{model_num,1} '\'];
        all_train_file=dir(fullfile(one_train_file_path,'/*.wav'));%读取该路径下的所有文件
        all_train_feature = [];
        for j=1:length(all_train_file)
            file_name=all_train_file(j).name;%wav文件名
            train_file=[one_train_file_path file_name];
           % fprintf('  train file:%s\n',train_file);
            [wav_data ,fs]=audioread(train_file);
            train_feature=melcepst(wav_data ,fs);
            all_train_feature=[all_train_feature;train_feature];
        end
        dirName=['.\model\' all_model_name{model_num,1} '\'];%初始的model_num为1
              
        %FXY-04################
        
        %[mu_model,sigma_model,weight_model]=gmm_estimate(all_train_feature',GMMM_component);
        map_tau = 10.0;
        config = 'mwv';
        map_all_train_feature=cell(1,1);
        map_all_train_feature{1,1}=all_train_feature';
        map_gmm=mapAdapt( map_all_train_feature, ubm, map_tau, config);
        mu_model=map_gmm.mu;
        sigma_model=map_gmm.sigma;
        weight_model=map_gmm.w;
       
        %FXY-04################        
        if ~exist( dirName, 'dir')
            mkdir(dirName);
        end
        save([dirName 'mu_model.mat'],'mu_model');
        save([dirName 'sigma_model.mat'],'sigma_model');
        save([dirName 'weight_model.mat'],'weight_model');
        model_num=model_num+1;
    end
end
save('.\model\all_model_name.mat','all_model_name');

all_model_name=importdata('.\model\all_model_name.mat');%从上面训练数据时保存的模型文件信息中读取模型文件信息
model_num=length(all_model_name);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = clock; 
disp([num2str(T(4)),':',num2str(T(5)),':',num2str(T(6))]);

%FXY-06################
ut_gmm=cell(model_num,1);
mydata=zeros(270,1);
%FXY-06################


%test1
FileList=dir(test_file_path);%读取该路径下的所有文件


%该路径下是否是文件夹
for i=1:length(FileList)
    if(FileList(i).isdir==1&&~strcmp(FileList(i).name,'.')&&~strcmp(FileList(i).name,'..'))
        test_name=FileList(i).name;
        one_test_file_path=[test_file_path  test_name '\'];
        all_test_file=dir(fullfile(one_test_file_path,'/*.wav'));%读取该路径下的所有文件
        %fprintf('测试类型：%s\n',test_name);
        for j=1:length(all_test_file)
            file_name=all_test_file(j).name;%wav文件名
            test_file=[one_test_file_path file_name];
            [wav_data ,fs]=audioread(test_file);
            test_feature=melcepst(wav_data ,fs);%上面皆是与训练模型时同样的操作
            %fprintf('Test：%s\n',test_file);
            for k=1:model_num%是上述训练数据时获得的模型个数
                model_path=['.\model\' all_model_name{k,1} '\'];
                mu_model=importdata([model_path 'mu_model.mat']);
                sigma_model=importdata([model_path 'sigma_model.mat']);
                weight_model=importdata([model_path 'weight_model.mat']);
                %FXY-05################
                gmm_struct = struct('w',weight_model,'mu',mu_model,'sigma',sigma_model);              
                ut_gmm{k,1}=gmm_struct;
                %FXY-05################
                [lYM, lY] = lmultigauss(test_feature', mu_model, sigma_model, weight_model');%test_feature'表矩阵转置
                score(j,k) = mean(lY);%返回包含每列均值的行向量(该条语音对第k个模型的打分)
                %训练库里有44个人的模型，一个测试文件下有18条语音(有三个人，每人6条语音)
                % fprintf('Model:%s  score:%f\n',all_model_name{k,1},score(j,k));          
            end
            
            %FXY-07################
            ut_gmmScores=zeros(9,1);
            ut_nSpeakerstest=1;
            ut_nChannels=1;
            ut_test_feature=cell(1,1);
            ut_test_feature{1,1}=test_feature';
            nSpeakers=model_num;
            ut_trials = zeros(ut_nSpeakerstest*ut_nChannels*model_num, 2);
            answers = zeros(ut_nSpeakerstest*ut_nChannels*nSpeakers, 1);
            for ix = 1 : nSpeakers
            b = (ix-1)*ut_nSpeakerstest*ut_nChannels + 1;
            e = b + ut_nSpeakerstest*ut_nChannels - 1;
            ut_trials(b:e, :) = [ix * ones(ut_nSpeakerstest*ut_nChannels, 1),(1:ut_nSpeakerstest*ut_nChannels)'];
            answers((ix-1)*ut_nChannels+b : (ix-1)*ut_nChannels+b+ut_nChannels-1) = 1;
            end
            ut_gmmScores = score_gmm_trials(ut_gmm, reshape(ut_test_feature, ut_nSpeakerstest*ut_nChannels,1), ut_trials, ubm);
            ut_gmmScores=reshape(ut_gmmScores,nSpeakers*ut_nChannels, ut_nSpeakerstest);
            [val, idx] = max(ut_gmmScores);
            a=all_model_name{idx,1};
            fprintf('\n Identified Speaker is %s \n', a);
            mydata((i-2)*18+j,1)=val;
            if(strcmp(a,'3140102441-W1'))
              
               correct_num=correct_num+1;
            end
            
            %FXY-07################
            
            %FXY-03################
%             [max_score,max_id]=max(score(j,:));
%             if(max_score>-21)
%                  fprintf('PEOPLE:%s\n',all_model_name{max_id,1});
%                  fprintf('SCORE:%s\n',max_score);
%             else
%                 fprintf('UNKNOWN:%s\n',test_file);
%             
%             end
%             
%             if(~strcmp(all_model_name{max_id,1},'3140102441-W1'))
%               error_num = error_num + 1;
%                error_file{error_num,1} = all_model_name{max_id,1};
%                error_file_m{error_num,1} = test_file;
%                error_file_score(error_num) = max_score;
%                error_file_score_m(error_num) = score(j,15);
%             
%             else
%                correct_num=correct_num+1;
%             end
            %FXY-03################
            
            
            %fprintf('MAX------%s\n',all_model_name{max_id,1});
            
         end
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %result
%         [max_score,max_id]=max(score(:,k));
%         [min_score,min_id]=min(score(:,k));
%         %fprintf('Max score:%f  file:%s\nMin score:%f  file:%s\n\n',max_score,all_test_file(max_id).name,min_score,all_test_file(min_id).name);
    end
end
T = clock; 
disp([num2str(T(4)),':',num2str(T(5)),':',num2str(T(6))]);

fprintf('Total Correct Number = %d\n',correct_num);
error_num=270-correct_num;
fprintf('Total Error Number = %d\n',error_num);
accuracy = 1 - error_num/270;
fprintf('CorrectRate = %f\n',accuracy);
%FXY-02################


% for t=1:error_num
%     fprintf('%s : %f    %s : %f\n',error_file_m{t,1},error_file_score_m(t), error_file{t,1}, error_file_score(t));
% end