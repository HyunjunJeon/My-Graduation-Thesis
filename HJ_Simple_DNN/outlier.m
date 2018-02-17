%
% Detection and Removal of Outliers in Data Sets
%    ( Rosner's many-outlier test)
%
%       index = outlier( y [, crit] )
%
% where  index = indices of outliers in the data
%        y     = data set (should be stationary)
%        crit  = detection criterion (default 2)
%
% Bob Newell, February 1996
%
%-----------------------------------------------------
%
function index = outlier( y, crit )
%
y = y(:);
n = length( y );
k = 10;
if nargin > 1,
  lambda = crit;
else,
  lambda = 2;
end;
R = zeros( k+1, 1 );
% sort deviations from the mean
ybar = mean( y );
[ ys, is ] = sort( abs( y - ybar ) );
% calculate statistics for up to k outliers
for i = 0:k,
  yy = ys(1:n-i);
  R(i+1) = abs( yy(n-i) - mean(yy) ) / std(yy);
end; 
% statistical test to find outliers
index = [];
for i = 1:k,
  if R(i)/R(k+1) > lambda,
    index = [ index is(n-i+1) ];
  end;
end;
% report results
disp( [ 'Outliers detected = ' ...
                num2str( length( index ) ) ] )
if length( index ) > 0,
  disp( 'Outlier indices are:' )
  disp( index )
end;
%-----------------------------------------------------
% the end
