function y=vad2(x,num)
%[x,fs]=audioread('E:/matlab/bin/N4.wav');%x:时域语音信号、fs:采样频率
x = double(x);
x = x / max(abs(x));
FrameLen = 240;%帧长为240点
FrameInc = 80;%帧移为80点
 
amp1 = 10;%初始短时能量高门限
amp2 = 2;%初始短时能量低门限
zcr1 = 10;%初始短时过零率高门限
zcr2 = 5;%初始短时过零率低门限
maxsilence = 8;  % 8*10ms  = 80ms
minlen  = 15;    % 15*10ms = 150ms
%语音段的最短长度，若语音段长度小于此值，则认为其为一段噪音
status  = 0;     %初始状态为静音状态
count   = 0;     %初始语音段长度为0
silence = 0;     %初始静音段长度为0
tmp1  = enframe2(x(1:end-1), FrameLen, FrameInc);
tmp2  = enframe2(x(2:end)  , FrameLen, FrameInc);
signs = (tmp1.*tmp2)<0;
diffs = (tmp1 -tmp2)>0.02;
zcr   = sum(signs.*diffs, 2);
%计算短时能量
amp = sum(abs(enframe2(x, FrameLen, FrameInc)), 2);
%调整能量门限
amp1 = min(amp1, max(amp)/4);
amp2 = min(amp2, max(amp)/8); 
%开始端点检测
x1 = 0;
x2 = 0;
for n=1:length(zcr) %length（zcr）得到的是整个信号的帧数
   goto = 0;
   switch status
   case {0,1}                   % 0 = 静音, 1 = 可能开始
      if amp(n) > amp1          % 确信进入语音段
         x1 = max(n-count-1,1);
         status  = 2;
         silence = 0;
         count   = count + 1;
      elseif amp(n) > amp2 | ... % 可能处于语音段
             zcr(n) > zcr2
         status = 1;
         count  = count + 1;
      else                       % 静音状态
         status  = 0;
         count   = 0;
      end
   case 2,                       % 2 = 语音段
      if amp(n) > amp2 | ...     % 保持在语音段
         zcr(n) > zcr2
         count = count + 1;
      else                       % 语音将结束
         silence = silence+1;
         if silence < maxsilence % 静音还不够长，尚未结束
            count  = count + 1;
         elseif count < minlen   % 语音长度太短，认为是噪声
            status  = 0;
            silence = 0;
            count   = 0;
         else                    % 语音结束
            status  = 3;
         end
      end
   case 3,
      break;
   end
end  
count = count-silence/2;
x2 = x1 + count -1;
y=x(x1*FrameInc:x2*FrameInc);
if num==1
    subplot(211)    %subplot(3,1,1)表示将图排成3行1列，最后的一个1表示下面要画第1幅图
    plot(x)
    axis([1 length(x) -1 1])    %函数中的四个参数分别表示xmin,xmax,ymin,ymax，即轴的范围
    ylabel('Speech');
    line([x1*FrameInc x1*FrameInc], [-1 1], 'Color', 'red');
    line([x2*FrameInc x2*FrameInc], [-1 1], 'Color', 'red');
    subplot(212)
    plot(y)
    axis([1 length(y) -1 1])
    ylabel('delsil');
end