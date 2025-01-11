 function [sharpe_ratio_VaR] = fun_VarCovMethod(x, Ret, pval)
    % function to be maximize in order to get the portfolio with the
    % maximum sharp_ratio_VaR

    
    retPf = x'*Ret';
    mu = mean(retPf);
    std_ = std(retPf);


    VaR_Norm = mu + std_*norminv(1-pval);

    % daily sharpe ratio
    sharpe_ratio_VaR= mu / VaR_Norm;
    
    % yearly sharpe ratio
    %sharpe_ratio_VaR= sum(retPf) / (VaR_Norm * sqrt(length(retPf)));

end