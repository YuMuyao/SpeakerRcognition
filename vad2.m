function y=vad2(x,num)
%[x,fs]=audioread('E:/matlab/bin/N4.wav');%x:ʱ�������źš�fs:����Ƶ��
x = double(x);
x = x / max(abs(x));
FrameLen = 240;%֡��Ϊ240��
FrameInc = 80;%֡��Ϊ80��
 
amp1 = 10;%��ʼ��ʱ����������
amp2 = 2;%��ʼ��ʱ����������
zcr1 = 10;%��ʼ��ʱ�����ʸ�����
zcr2 = 5;%��ʼ��ʱ�����ʵ�����
maxsilence = 8;  % 8*10ms  = 80ms
minlen  = 15;    % 15*10ms = 150ms
%�����ε���̳��ȣ��������γ���С�ڴ�ֵ������Ϊ��Ϊһ������
status  = 0;     %��ʼ״̬Ϊ����״̬
count   = 0;     %��ʼ�����γ���Ϊ0
silence = 0;     %��ʼ�����γ���Ϊ0
tmp1  = enframe2(x(1:end-1), FrameLen, FrameInc);
tmp2  = enframe2(x(2:end)  , FrameLen, FrameInc);
signs = (tmp1.*tmp2)<0;
diffs = (tmp1 -tmp2)>0.02;
zcr   = sum(signs.*diffs, 2);
%�����ʱ����
amp = sum(abs(enframe2(x, FrameLen, FrameInc)), 2);
%������������
amp1 = min(amp1, max(amp)/4);
amp2 = min(amp2, max(amp)/8); 
%��ʼ�˵���
x1 = 0;
x2 = 0;
for n=1:length(zcr) %length��zcr���õ����������źŵ�֡��
   goto = 0;
   switch status
   case {0,1}                   % 0 = ����, 1 = ���ܿ�ʼ
      if amp(n) > amp1          % ȷ�Ž���������
         x1 = max(n-count-1,1);
         status  = 2;
         silence = 0;
         count   = count + 1;
      elseif amp(n) > amp2 | ... % ���ܴ���������
             zcr(n) > zcr2
         status = 1;
         count  = count + 1;
      else                       % ����״̬
         status  = 0;
         count   = 0;
      end
   case 2,                       % 2 = ������
      if amp(n) > amp2 | ...     % ������������
         zcr(n) > zcr2
         count = count + 1;
      else                       % ����������
         silence = silence+1;
         if silence < maxsilence % ����������������δ����
            count  = count + 1;
         elseif count < minlen   % ��������̫�̣���Ϊ������
            status  = 0;
            silence = 0;
            count   = 0;
         else                    % ��������
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
    subplot(211)    %subplot(3,1,1)��ʾ��ͼ�ų�3��1�У�����һ��1��ʾ����Ҫ����1��ͼ
    plot(x)
    axis([1 length(x) -1 1])    %�����е��ĸ������ֱ��ʾxmin,xmax,ymin,ymax������ķ�Χ
    ylabel('Speech');
    line([x1*FrameInc x1*FrameInc], [-1 1], 'Color', 'red');
    line([x2*FrameInc x2*FrameInc], [-1 1], 'Color', 'red');
    subplot(212)
    plot(y)
    axis([1 length(y) -1 1])
    ylabel('delsil');
end