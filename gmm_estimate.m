function [mu,sigm,c]=gmm_estimate(X,M,iT,mu,sigm,c,Vm)
% [mu,sigma,c]=gmm_estimate(X,M,<iT,mu,sigm,c,Vm>)
% 
% X   : the column by column data matrix (LxT)
% M   : number of gaussians
% iT  : number of iterations, by defaut 10
% mu  : initial means (LxM)
% sigm: initial diagonals for the diagonal covariance matrices (LxM)
% c   : initial weights (Mx1)
% Vm  : minimal variance factor, by defaut 4 ->minsig=var/(M�Vm�?)

  % *************************************************************
  % GENERAL PARAMETERSһ�����
  [L,T]=size(X);        % data length L��T��(all_train_feature��һ��1995*12��mfccά�ȣ��ľ���)
  varL=var(X')';    % variance for each row data; ����ÿһ�еķ���
  
  min_diff_LLH=0.001;   % convergence criteria �������б�׼��

  % DEFAULTS
  %nargin��matlab��ԭ�е�һ����������ʾ���Ǻ��������������Ŀ����û������һ��ֵ������£�ʹ��Ĭ��ֵ��
  if nargin<3  iT=10; end   % number of iterations, by defaut 10
  if nargin<4  mu=X(:,[fix((T-1).*rand(1,M))+1]); end % mu def: M rand vect. T��12���μ���������14�д��룩
  %���ͣ���������fix����˼�ǳ�0�������룬��������Ĭ��muֵ�Ƕ�(T-1).*rand(1,M)����������1
  %���ͣ�(T-1).*rand(1,M)�������".*"��ʾ��ˣ��������Ԫ����������ӦԪ����˵õ��Ľ�����������M�����Ǵ����16
  if nargin<5  sigm=repmat(varL./(M.^2),[1,M]); end % sigm def: same variance varL(X��ÿһ�еķ���)���M�ĵ�˷�
  %���ͣ�����repmat��ʾ�ظ�����varL./(M.^2)����M�и��þ���ƴ�ӳ�Ϊ1��
  if nargin<6  c=ones(M,1)./M; end  % c def: same weight 16��1�е�Ȩ�ؾ��󣨳�ʼ��ȫΪ1��
  if nargin<7  Vm=4; end   % minimum variance factor
  
  min_sigm=repmat(varL./(Vm.^2*M.^2),[1,M]);   % MINIMUM sigma!��С��sigma

  % VARIABLES
  lgam_m=zeros(T,M);    % prob of each (X:,t) to belong to the kth mixture 12��16�е������
  lB=zeros(T,1);        % log-likelihood 12��1�е������ 
  lBM=zeros(T,M);       % log-likelihhod for separate mixtures 12��16�е������

  old_LLH=-9e99;        % initial log-likelihood -9*(10^99)

  % START ITERATATIONS  
  for iter=1:iT
    % ESTIMATION STEP ****************************************************
    [lBM,lB]=lmultvigauss(X,mu,sigm,c);
    
    LLH=mean(lB);%mean���ذ���ÿ�о�ֵ��������

    lgam_m=lBM-repmat(lB,[1,M]);  % logarithmic version
    gam_m=exp(lgam_m);            % linear version           -Equation(1)
    
    
    % MAXIMIZATION STEP **************************************************
    sgam_m=sum(gam_m);            % sum of gam_m for all X(:,t)
    
     
    % gaussian weights ************************************
    new_c=mean(gam_m)';      %                                -Equation(4)

    % means    ********************************************
    % (convert gam_m and X to (L,M,T) and .* and then sum over T)
    mu_numerator=sum(permute(repmat(gam_m,[1,1,L]),[3,2,1]).*...
               permute(repmat(X,[1,1,M]),[1,3,2]),3);
    % convert  sgam_m(1,M,N) -> (L,M,N) and then ./
    new_mu=mu_numerator./repmat(sgam_m,[L,1]);              % -Equation(2)

    % variances *******************************************
    sig_numerator=sum(permute(repmat(gam_m,[1,1,L]),[3,2,1]).*...
                permute(repmat(X.*X,[1,1,M]),[1,3,2]),3);
    
    new_sigm=sig_numerator./repmat(sgam_m,[L,1])-new_mu.^2; % -Equation(3)

    % the variance is limited to a minimum
    new_sigm=max(new_sigm,min_sigm);
    

    %*******
    % UPDATE

    if old_LLH>=LLH-min_diff_LLH
        disp('converge');
      break;
    else
      old_LLH=LLH;
      mu=new_mu;
      sigm=new_sigm;
      c=new_c;
    end
    
    %******************************************************************
  end
