function [] = plot_performance(weights, returns, dates)

figure()
for i = 1 : size(weights, 2)
    
    % Computing portfolio returns across time
    portfolio_values = cumprod(returns * weights(:, i));
    portfolio_returns = 100.*portfolio_values/portfolio_values(1);

    plot(dates(2 : end), portfolio_returns)
    hold on
end

grid on
title('Evolution of portfolios performance across time')
xlabel('Time')
ylabel('Portfolio value')
legend('Portfolio EW','Portfolio A', 'Portfolio B', 'Portfolio C',...
    'Portfolio D','Portfolio E','Portfolio F','Portfolio G',...
    'Portfolio H','Portfolio I','Portfolio L','Portfolio M',...
    'Portfolio N','Portfolio P')
hold off
end