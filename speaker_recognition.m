 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�����������ڵ��˵�˵����ȷ��
clear all;
close all;
MFCC_size=12;%mfcc��ά��
GMMM_component=16;%GMM component ����

mu_model=zeros(MFCC_size,GMMM_component);%��˹ģ�� ���� ��ֵ������һ��12*16�������
sigma_model=zeros(MFCC_size,GMMM_component);%��˹ģ�� ���� ����
weight_model=zeros(GMMM_component);%��˹ģ�� ���� Ȩ�أ�����һ��16*16�������

train_file_path='.\training\';%ģ��ѵ���ļ�·��
test_file_path='.\testing\';%�����ļ�·��

all_train_feature=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%train model
FileList=dir(train_file_path);%��ȡ��·���µ������ļ�
model_num=1;%ע��ģ�͵ĸ���
error_num=0;%ʶ�����ĸ���
%FXY-01################
correct_num=0;
%FXY-01################

%��·�����Ƿ����ļ���
for i=1:length(FileList)
    if(FileList(i).isdir==1&&~strcmp(FileList(i).name,'.')&&~strcmp(FileList(i).name,'..'))
        all_model_name{model_num,1}=FileList(i).name;%�洢ģ������
        fprintf('Train:%s\n',all_model_name{model_num,1});
        one_train_file_path=[train_file_path  all_model_name{model_num,1} '\'];
        all_train_file=dir(fullfile(one_train_file_path,'/*.wav'));%��ȡ��·���µ������ļ�
        all_train_feature = [];
        for j=1:length(all_train_file)
            file_name=all_train_file(j).name;%wav�ļ���
            train_file=[one_train_file_path file_name];
          
            % fprintf('  train file:%s\n',train_file);
            [wav_data ,fs]=audioread(train_file);
            wav_data1=vad2(wav_data,j);
            train_feature=melcepst(wav_data1 ,fs);
            all_train_feature=[all_train_feature;train_feature];
        end
        dirName=['.\model\' all_model_name{model_num,1} '\'];%��ʼ��model_numΪ1
        [mu_model,sigma_model,weight_model]=gmm_estimate(all_train_feature',GMMM_component);
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

all_model_name=importdata('.\model\all_model_name.mat');%������ѵ������ʱ�����ģ���ļ���Ϣ�ж�ȡģ���ļ���Ϣ
model_num=length(all_model_name);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%test
%WLY-01########################################
% FileList=dir(test_file_path);%��ȡ��·���µ������ļ�
%��·�����Ƿ����ļ���

% for i=1:length(FileList)
%     if(FileList(i).isdir==1&&~strcmp(FileList(i).name,'.')&&~strcmp(FileList(i).name,'..'))
%       test_name=FileList(i).name;
%���ʶԻ���
h = questdlg('��Ҫ����������','������ʾ','�ǵ�','�������Կ�����','Yes');
if strcmp(h,'�ǵ�')
    b='����������ļ���(��:wly.wav)';
    prompt={char(b)};
    title='¼��';
    an=cellstr(inputdlg(prompt,title));
    try 
        (strcmp(an(1),'')~=1);
    catch
         h=errordlg('��ȡ���˲���');waitfor(h);return
    end
    fs = 8000; 
    success=1;
    try 
        wb = waitbar(0,'�ڽ�����������ʼ¼��������3���ӵ�¼��ʱ��','name','��׼��...');
        for i=1:100,waitbar(i/100,wb);pause(0.01);end
        set(wb,'name','��˵����');
        y = audiorecorder;
        recordblocking(y, 3);
        disp('End of Recording.');
        play(y);
        myRecording = getaudiodata(y);
        close(wb);
        wav=an(1);
        wav=wav{1,1};
        audiowrite(wav,myRecording,fs);
    catch
        success = 0;
        wav=[];
    end;
    if success == 1
        hBox = msgbox('¼���ļ��ɹ�','�ɹ�');
        %waitfor(hBox);
        pause(1);
        close(hBox);
    else
        hBox = msgbox('¼���ļ�ʧ��','ʧ��','warn');
        %waitfor(hBox);
        pause(1);
        close(hBox);
    end
    h=msgbox(wav);
    waitfor(h);
    test_file = wav;
% else
%       one_test_file_path=[test_file_path  test_name '\'];
%       all_test_file=dir(fullfile(one_test_file_path,'/*.wav'));%��ȡ��·���µ������ļ�
%         fprintf('�������ͣ�%s\n',test_name);
%       for j=1:length(all_test_file)
%             file_name=all_test_file(j).name;%wav�ļ���
%             test_file=[one_test_file_path file_name];
            [wav_data ,fs]=audioread(test_file);
            %������
            wav_data1=vad2(wav_data,j);
            %������
            test_feature=melcepst(wav_data1 ,fs);%���������ѵ��ģ��ʱͬ���Ĳ���
            %fprintf('Test��%s\n',test_file);

            for k=1:model_num%������ѵ������ʱ��õ�ģ�͸���
                model_path=['.\model\' all_model_name{k,1} '\'];
                mu_model=importdata([model_path 'mu_model.mat']);
                sigma_model=importdata([model_path 'sigma_model.mat']);
                weight_model=importdata([model_path 'weight_model.mat']);
                [lYM, lY] = lmultigauss(test_feature', mu_model, sigma_model, weight_model);%test_feature'�����ת��
%               score(j,k) = mean(lY);%���ذ���ÿ�о�ֵ��������(���������Ե�k��ģ�͵Ĵ��)
                score(1,k) = mean(lY);
                %ѵ��������44���˵�ģ�ͣ�һ�������ļ�����18������(�������ˣ�ÿ��6������)
                %fprintf('Model:%s  score:%f\n',all_model_name{k,1},score(1,k));
            end
            %FXY-03################
            %[max_score,max_id]=max(score(j,:));
            [max_score,max_id]=max(score(1,:));
            %WLY-01##########################################
            if(max_score>-21.714218183)
                 fprintf('PEOPLE:%s\n',all_model_name{max_id,1});
                 fprintf('SCORE:%s\n',max_score);
            else
                fprintf('UNKNOWN:%s\n',test_file);
            
            end
            
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
           
        %end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %result
        [max_score,max_id]=max(score(:,k));
        [min_score,min_id]=min(score(:,k));
        %fprintf('Max score:%f  file:%s\nMin score:%f  file:%s\n\n',max_score,all_test_file(max_id).name,min_score,all_test_file(min_id).name);
%     end
% end
%FXY-02################
% fprintf('Total Correct Number = %d\n',correct_num);
% fprintf('Total Error Number = %d\n',error_num);
% accuracy = 1 - error_num/270;
% fprintf('CorrectRate = %f\n',accuracy);
%FXY-02################
else
   FileList=dir(test_file_path);%��ȡ��·���µ������ļ�
%��·�����Ƿ����ļ���
for i=1:length(FileList)
    if(FileList(i).isdir==1&&~strcmp(FileList(i).name,'.')&&~strcmp(FileList(i).name,'..'))
        test_name=FileList(i).name;
        one_test_file_path=[test_file_path  test_name '\'];
        all_test_file=dir(fullfile(one_test_file_path,'/*.wav'));%��ȡ��·���µ������ļ�
        %fprintf('�������ͣ�%s\n',test_name);
        for j=1:length(all_test_file)
            file_name=all_test_file(j).name;%wav�ļ���
            test_file=[one_test_file_path file_name];
            [wav_data ,fs]=audioread(test_file);
            %������
            wav_data1=vad2(wav_data,j);
            %������
            test_feature=melcepst(wav_data1 ,fs);%���������ѵ��ģ��ʱͬ���Ĳ���
            %fprintf('Test��%s\n',test_file);
            for k=1:model_num%������ѵ������ʱ��õ�ģ�͸���
                model_path=['.\model\' all_model_name{k,1} '\'];
                mu_model=importdata([model_path 'mu_model.mat']);
                sigma_model=importdata([model_path 'sigma_model.mat']);
                weight_model=importdata([model_path 'weight_model.mat']);
                [lYM, lY] = lmultigauss(test_feature', mu_model, sigma_model, weight_model);%test_feature'�����ת��
                score(j,k) = mean(lY);%���ذ���ÿ�о�ֵ��������(���������Ե�k��ģ�͵Ĵ��)
                %ѵ��������44���˵�ģ�ͣ�һ�������ļ�����18������(�������ˣ�ÿ��6������)
                % fprintf('Model:%s  score:%f\n',all_model_name{k,1},score(j,k));
            end
            %FXY-03################
             [max_score,max_id]=max(score(j,:));
%             if(max_score>-21)
%                  fprintf('PEOPLE:%s\n',all_model_name{max_id,1});
%                  fprintf('SCORE:%s\n',max_score);
%             else
%                 fprintf('UNKNOWN:%s\n',test_file);
%             
%             end
            
            if(~strcmp(all_model_name{max_id,1},'3140102441-W1'))
              error_num = error_num + 1;
               error_file{error_num,1} = all_model_name{max_id,1};
               error_file_m{error_num,1} = test_file;
               error_file_score(error_num) = max_score;
               error_file_score_m(error_num) = score(j,15);
            
            else
               correct_num=correct_num+1;
            end
            %FXY-03################
            
            %fprintf('MAX------%s\n',all_model_name{max_id,1});
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %result
        [max_score,max_id]=max(score(:,k));
        [min_score,min_id]=min(score(:,k));
        %fprintf('Max score:%f  file:%s\nMin score:%f  file:%s\n\n',max_score,all_test_file(max_id).name,min_score,all_test_file(min_id).name);
    end
end
%FXY-02################
fprintf('Total Correct Number = %d\n',correct_num);
fprintf('Total Error Number = %d\n',error_num);
accuracy = 1 - error_num/270;
fprintf('CorrectRate = %f\n',accuracy);
%FXY-02################
end
for t=1:error_num
    fprintf('%s : %f    %s : %f\n',error_file_m{t,1},error_file_score_m(t), error_file{t,1}, error_file_score(t));
end