function mse = mse_risk_contribution(x, LogRet,tgt)
    [relRC, RC, mVol] = getRiskContributions(x, LogRet);
    mse = mean((relRC-tgt).^2);
end