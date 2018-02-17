function [r,lag,limit] = cross_correlation(x,y)
limit = 2/sqrt(length(x));


x_mean = mean(x);
y_mean = mean(y);

x_mean_sub = x - x_mean;
y_mean_sub = y - y_mean;

x_std = std(x);
y_std = std(y);

r = zeros(100,1);
for k =  1:100
    sum1 = 0;
    for i = 1:(length(x)-k)
        if i-k<1
            sum1 = sum1 + 0;
        else
            sum1 = sum1 + x_mean_sub(i,1)*y_mean_sub(i-k,1);
        end
    end
    r(k,1) = (sum1/(length(x)-1))/sqrt(x_std*y_std);
end

lag = min(find(r<limit))-1;