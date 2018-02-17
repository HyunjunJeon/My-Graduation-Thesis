function [r,lag,limit] = auto_correlation(x)
limit = 2/sqrt(length(x));


x1_mean = mean(x);
x2_mean = mean(x);

x1_mean_sub = x - x1_mean;
x2_mean_sub = x - x2_mean;

x1_std = std(x);
x2_std = std(x);

r = zeros(100,1);
for k =  1:100
    sum1 = 0;
    for i = 1:(length(x)-k)
        if i-k<1
            sum1 = sum1 + 0;
        else
            sum1 = sum1 + x1_mean_sub(i,1)*x2_mean_sub(i-k,1);
        end
    end
    r(k,1) = (sum1/(length(x)-1))/sqrt(x1_std*x2_std);
end

lag = min(find(r<limit))-1;