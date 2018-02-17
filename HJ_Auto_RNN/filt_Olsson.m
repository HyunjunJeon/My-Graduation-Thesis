%
% Filter the specified time series given a filter
% time constant and using an 8th order Butterworth
% algorithm with zero phase shift
%
%     yf = filt( y, delt, tau )
%
% where   y    = input time series
%         delt = sampling time interval
%         tau  = filter time constant
%         yf   = filtered time series
%
% Bob Newell, April 1996
%
%----------------------------------------------------------
function yf = filt( y, delt, tau )
%
% first calculate the filter parameters
%
a = [ 1 8 28 56 70 56 28 8 1 ];
b = [ 1 8 28 56 70 56 28 8 1 ];
c = [ 5.1258 13.1371 21.8462 ...
         25.6884 21.8462 13.1371 5.1258 1 ];
aa = [ 1 6 14 14 0 -14 -14 -6 -1; ...
       1 4 4 -4 -10 -4 4 4 1; ...
	   1 2 -2 -6 0 6 2 -2 -1; ...
	   1 0 -4 0 6 0 -4 0 1; ...
	   1 -2 -2 6 0 -6 2 2 -1; ...
	   1 -4 4 4 -10 4 4 -4 1; ...
	   1 -6 14 -14 0 14 -14 6 -1; ...
	   1 -8 28 -56 70 -56 28 -8 1 ];
%
t = 2 * tau / delt;
tt = 1;
for i = 1:8,
  tt = tt * t;
  a = a + aa(i,:) * c(i) * tt; 
end;
b = b / a(1);
a = a / a(1);
%
% now double filter with reversals to get zero phase shift
% 
ym = mean( y );
x = [ y - ym ];
yf = filter( b, a, x );
x = yf( length(x):-1:1 );
yf = filter( b, a, x );
x = yf( length(x):-1:1 );
yf = x + ym;
%
return;
%----------------------------------------------------------
% That's all
