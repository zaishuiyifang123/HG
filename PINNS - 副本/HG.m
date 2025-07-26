
% 产生厄米高斯模式的强度和相位图像
% [Is,phase]=getHGmode_F(w0,z,N,pix,Gmode,varargin)
% w0——高斯光束腰斑半径，mm
% z——沿z轴传播的距离
% N——图像分辨率
% Gmode——HG模式阶数(字符类型)
% varargin——缺省参数,这里用来构造倾斜模式
% copyright @adride  qq:466193203
% $date 2017-3-3$ $version 3.0$
% $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

%% 设定主要参数
h=1080*10^-6;%mm,波长
Dx=0.02;%光场平面的物理分辨率
Dy=Dx;
N=500;
z=0;
w0=1.4;
x1=(-N/2:N/2-1)*Dx;
y1=(-N/2:N/2-1)*Dy;
[x,y]=meshgrid(x1,y1);
f=pi*w0^2/h;%瑞利长度
R=z+f^2/(z+eps)+eps;%曲率半径
w=w0*sqrt(1+(z/f)^2);%传输到z处的光斑大小
k=2*pi/h;%波数
%% 数据预处理
m=4;%HG的m阶数
n=0;%HG的n阶数
X=x;
Y=y;
%% HG函数项
HGphase=exp(-i*(k*(z+(x.^2+y.^2)/2/R)-(m+n+1)*atan(z/f)));%HG标准相位
switch m
    case 0
        Hm=1;%0阶厄米函数
    case 1
        Hm=2*X;%1阶厄米函数
    case 2
        Hm=4*X.^2-2;%2阶厄米函数
    case 3
        Hm=8*X.^3-12*X;%3阶厄米函数
    case 4
        Hm=16*X.^4-48*X.^2+12;%4阶厄米函数
    case 5
        Hm=32*X.^5-160*X.^3+120*X;%5阶厄米函数
    case 6
        Hm=64*X.^6-480*X.^4+720*X.^2-120;%6阶厄米函数
    case 7
        Hm=128*X.^7-1344*X.^5+3360*X.^3-1680*X;%7阶厄米函数
    case 8
        Hm=256*X.^8-3584*X.^6+13440*X.^4-13440*X.^2+1680;%8阶厄米函数
    case 9
        Hm=512*X.^9-9216*X.^7+48384*X.^5-80640*X.^3+30240*X;%9阶厄米函数
end
switch n
    case 0
        Hn=1;%0阶厄米函数
    case 1
        Hn=2*Y;%1阶厄米函数
    case 2
        Hn=4*Y.^2-2;%2阶厄米函数
    case 3
        Hn=8*Y.^3-12*Y;%3阶厄米函数
    case 4
        Hn=16*Y.^4-48*Y.^2+12;%4阶厄米函数
    case 5
        Hn=32*Y.^5-160*Y.^3+120*Y;%5阶厄米函数
    case 6
        Hn=64*Y.^6-480*Y.^4+720*Y.^2-120;%6阶厄米函数
    case 7
        Hn=128*Y.^7-1344*Y.^5+3360*Y.^3-1680*Y;%7阶厄米函数
    case 8
        Hn=256*Y.^8-3584*Y.^6+13440*Y.^4-13440*Y.^2+1680;%8阶厄米函数
    case 9
        Hn=512*Y.^9-9216*Y.^7+48384*Y.^5-80640*Y.^3+30240*Y;%9阶厄米函数
end
Hmn=Hm.*Hn;
%% HGmode function
As=1/w*Hmn.*exp(-(x.^2+y.^2)/w^2).*HGphase;
As=double(As);
%%  数据后处理
Is=As.*conj(As);
% Is=abs(As);
Is=Is/sum(sum(Is));%对模式归一化
phase=angle(As);
%-the end
figure,
imshow(Is,[]);