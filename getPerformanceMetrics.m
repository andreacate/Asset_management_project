function Metrics =...
    getPerformanceMetrics(portfolio, returns, risk_free, EW_rel_RC)

% portfolio vettore colonna

% Computing portfolio returns across time
portfolio_values = cumprod(returns*portfolio);
portfolio_returns = 100.*portfolio_values/portfolio_values(1);

% Metrics computation

% Computing annualized return
AnnRet = ( portfolio_returns(end) / portfolio_returns(1) )...
    ^ (250 / size(returns, 1)) - 1;

% Computing annualized volatility
AnnVol = sqrt(250) *...
    std(portfolio_returns(2:end) ./ portfolio_returns(1:end-1) - 1);

% Computing sharpe ratio
Sharpe = (AnnRet - risk_free) / AnnVol;

% Computing maximum drowdown
dd = zeros(1, length(portfolio_returns));
for i = 1:length(portfolio_returns)
    dd(i) = ( portfolio_returns(i) / max(portfolio_returns(1:i)) ) - 1;
end
MaxDD = min(dd);

% Computing calmar ratio
Calmar = (AnnRet - risk_free) / MaxDD;

% Computing diversification ratio
DR = getDiversificationRatio(portfolio, returns);

% Computing entropy
Entropy = getEntropy(portfolio);

% Computing risk contribution
[relRC, RC, ~] = getRiskContributions(portfolio, returns);

% Computing risk contribution MSE
MSE = mse_risk_contribution(portfolio, returns, EW_rel_RC);

% Storing metrics in a struct
Metrics = struct('AnnRet', AnnRet, ...
                 'AnnVol', AnnVol, ...
                 'Sharpe', Sharpe, ...
                 'MaxDD', MaxDD, ...
                 'Calmar', Calmar, ...
                 'DR', DR, ...
                 'Entropy', Entropy, ...
                 'relRC', relRC, ...
                 'RC', RC, ...
                 'MSE', MSE);

end