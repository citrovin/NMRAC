function y = Sigmoid_SAT(x, SAT)
    y = (2*(1-exp(-SAT.*x)))./(SAT*(1+exp(-SAT.*x)));
end