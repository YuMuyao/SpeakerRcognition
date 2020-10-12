function [mu,sigm,c]=gmm_estimate(X,M,iT,mu,sigm,c,Vm)
% [mu,sigma,c]=gmm_estimate(X,M,<iT,mu,sigm,c,Vm>)
% 
% X   : the column by column data matrix (LxT)
% M   : number of gaussians
% iT  : number of iterations, by defaut 10
% mu  : initial means (LxM)
% sigm: initial diagonals for the diagonal covariance matrices (LxM)
% c   : initial weights (Mx1)
% Vm  : minimal variance factor, by defaut 4 ->minsig=var/(M锟絍m锟?)

  % *************************************************************
  % GENERAL PARAMETERS一般参数
  [L,T]=size(X);        % data length L行T列(all_train_feature是一个1995*12（mfcc维度）的矩阵)
  varL=var(X')';    % variance for each row data; 计算每一行的方差
  
  min_diff_LLH=0.001;   % convergence criteria 收敛性判别准则

  % DEFAULTS
  %nargin是matlab中原有的一个参数，表示的是函数输入参数的数目（在没有输入一下值的情况下，使用默认值）
  if nargin<3  iT=10; end   % number of iterations, by defaut 10
  if nargin<4  mu=X(:,[fix((T-1).*rand(1,M))+1]); end % mu def: M rand vect. T是12（参见本函数第14行代码）
  %解释：上述函数fix的意思是朝0四舍五入，所以上述默认mu值是对(T-1).*rand(1,M)四舍五入后加1
  %解释：(T-1).*rand(1,M)，这里的".*"表示点乘（矩阵各个元素与另矩阵对应元素相乘得到的结果），这里的M是我们传入的16
  if nargin<5  sigm=repmat(varL./(M.^2),[1,M]); end % sigm def: same variance varL(X中每一行的方差)点除M的点乘方
  %解释：这里repmat表示重复矩阵varL./(M.^2)，把M列个该矩阵拼接成为1行
  if nargin<6  c=ones(M,1)./M; end  % c def: same weight 16行1列的权重矩阵（初始化全为1）
  if nargin<7  Vm=4; end   % minimum variance factor
  
  min_sigm=repmat(varL./(Vm.^2*M.^2),[1,M]);   % MINIMUM sigma!最小的sigma

  % VARIABLES
  lgam_m=zeros(T,M);    % prob of each (X:,t) to belong to the kth mixture 12行16列的零矩阵
  lB=zeros(T,1);        % log-likelihood 12行1列的零矩阵 
  lBM=zeros(T,M);       % log-likelihhod for separate mixtures 12行16列的零矩阵

  old_LLH=-9e99;        % initial log-likelihood -9*(10^99)

  % START ITERATATIONS  
  for iter=1:iT
    % ESTIMATION STEP ****************************************************
    [lBM,lB]=lmultvigauss(X,mu,sigm,c);
    
    LLH=mean(lB);%mean返回包含每列均值的行向量

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
