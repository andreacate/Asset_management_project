clear 
close all
clc


%% Loading Data
load('capitalizations.mat');
load('table_prices.mat')


%% Transform prices from table to timetable

dt = table_prices(:,1).Variables; % Date

values = table_prices(:,2:end).Variables; % Prices

nm = table_prices.Properties.VariableNames(2:end); % Firms' names

myPrice_dt = array2timetable(values, 'RowTimes', dt,'Variablenames', nm); 


%% Selection of a subset of Dates

start_dt = datetime('01/01/2023', 'InputFormat', 'dd/MM/yyyy');
end_dt = datetime('31/12/2023', 'InputFormat', 'dd/MM/yyyy');
rng = timerange(start_dt, end_dt, 'closed'); % Closed include extreme date

% We can do this exploiting the time-based indexing of the timetable type
subsample = myPrice_dt(rng,:); 
prices_val = subsample.Variables;
dates_ = subsample.Time;


%% Processing data

% Calculate log-returns
ret = prices_val(2 : end, :) ./ prices_val(1 : end - 1, :);
LogRet = log(ret);

% Calculate moments
ExpRet = mean(LogRet);
V = cov(LogRet);

% Parameters
NumAssets = size(ExpRet,2);
number_ptf_frontier = 100;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Exercise 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Frontier Computation with Object portfolio

% Building the portfolio object
P1 = Portfolio('AssetList', nm, 'NumAssets', NumAssets);
P1 = P1.setAssetMoments(ExpRet, V);
P1 = P1.setDefaultConstraints();

% Computing the frontier
Frontier_1 = estimateFrontier(P1, number_ptf_frontier);
[Vol_F1, Ret_F1] = estimatePortMoments(P1, Frontier_1);


%% Minimum Variance and Maximum Sharpe Portofolio

% Minimum Variance
[VolPtfA, min_var_index_1] = min(Vol_F1);
Portfolio_A = Frontier_1(:, min_var_index_1);

% Maximum Sharpe
Portfolio_B = estimateMaxSharpeRatio(P1);
[VolPtfB, RetPtfB] = estimatePortMoments(P1, Portfolio_B);


%% Plot
figure()
% Plot the efficient frontier 
plot(Vol_F1, Ret_F1, 'k', 'LineWidth', 1.5) % Black
hold on

% Minimum variance portfolio with red marker
plot(Vol_F1(min_var_index_1), Ret_F1(min_var_index_1), ...
    'o', 'Color', [0.8 0 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.4 0.4]) % Red

% Maximum sharpe portfolio with green marker
plot(VolPtfB, RetPtfB, ...
    'o', 'Color', [0 0.5 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.3 0.8 0.3]) % Green

% Grid and labels
grid on
legend('Efficient Frontier', 'Portfolio A (Min Variance)', ...
    'Portfolio B (Max Sharpe)', 'Location', 'best')
title('Efficient Frontier Standard Constraints')
xlabel('Volatility')
ylabel('Expected Return')
hold off


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Exercise 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Object portfolio

% Building the portfolio object
P2 = Portfolio('AssetList', nm, 'NumAssets', NumAssets);
P2 = P2.setAssetMoments(ExpRet, V);
P2 = P2.setDefaultConstraints();


%% Additional constraints

% Building the constraints
inequalities_number = 3;
equalities_number = 2;
[Aeq, beq, A, b] = ...
    matrix_constraints(equalities_number, inequalities_number, NumAssets);

% Adding the constraints to the portfolio
P2 = setInequality(P2, A, b);
P2 = setEquality(P2, Aeq, beq);


%% Computing the frontier
Frontier_2 = estimateFrontier(P2, number_ptf_frontier);
[Vol_F2, Ret_F2] = estimatePortMoments(P2, Frontier_2);


%% Minimum Variance and Maximum Sharpe Portofolio

% Minimum Variance
[min_var_2, min_var_index_2] = min(Vol_F2);
Portfolio_C = Frontier_2(:, min_var_index_2);

% Maximum Sharpe
Portfolio_D = estimateMaxSharpeRatio(P2);
[VolPtfD, RetPtfD] = estimatePortMoments(P2, Portfolio_D);


%% Plot
figure()
% Efficient frontier
plot(Vol_F2, Ret_F2, 'Color', [0.4 0.7 1], 'LineWidth', 1.5) 
hold on

% Minimum Variance Portfolio
plot(Vol_F2(min_var_index_2), Ret_F2(min_var_index_2), ...
    'o', 'Color', [0.8 0 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.4 0.4]) % Red

% Maximum Sharpe Portfolio
plot(VolPtfD, RetPtfD, ...
    'o', 'Color', [0 0.5 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.3 0.8 0.3]) % Green

grid on
legend('Efficient Frontier', 'Portfolio C (Min Variance)', ...
    'Portfolio D (Max Sharpe)', 'Location', 'best')
title('Efficient Frontier With Additional Constraints')
xlabel('Volatility')
ylabel('Expected Return')
hold off


figure()
% Efficient frontier no constraints
plot(Vol_F1, Ret_F1, 'k', 'LineWidth', 1.5) 
hold on

% Efficient frontier with constraints
plot(Vol_F2, Ret_F2, 'Color', [0.4 0.7 1], 'LineWidth', 1.5) 

% Portfolio B
plot(VolPtfB, RetPtfB, ...
    'o', 'Color', [0 0.5 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.3 0.8 0.3]) 

% Portfolio A
plot(Vol_F1(min_var_index_1), Ret_F1(min_var_index_1), ...
    'o', 'Color', [0.8 0 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.4 0.4]) 

% Portfolio D
plot(VolPtfD, RetPtfD, ...
    'o', 'Color', [0 0.5 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.3 0.8 0.3]) 

% Portfolio C
plot(Vol_F2(min_var_index_2), Ret_F2(min_var_index_2), ...
    'o', 'Color', [0.8 0 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.4 0.4]) 

grid on
legend('No Constraints', 'With Constraints', ...
    'Location', 'best')
title('Comparison of Efficient Frontiers')
xlabel('Volatility')
ylabel('Expected Return')
hold off


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Exercise 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parameters
M_samples = 150;


%% Exercise 1 constraints
Vol_Rob_F1 = zeros(number_ptf_frontier, M_samples);
Ret_Rob_F1 = zeros(number_ptf_frontier, M_samples);

%n. sim, n.assets, n portf in a single frontier
Weights_Rob_F1 = zeros(NumAssets, number_ptf_frontier);

for n = 1 : M_samples
    
    % Sampling the expected return vector and covariance matrix
    R = mvnrnd(ExpRet, V, length(LogRet));
    NewExpRet = mean(R);
    NewCov = cov(R);
    
    % Initializing a portfolio with default constraints
    psim = Portfolio('AssetList', nm, 'NumAssets', NumAssets);
    psim = psim.setAssetMoments(NewExpRet, NewCov);
    psim = psim.setDefaultConstraints();

    % Estimating the frontier
    w_sim = estimateFrontier(psim, number_ptf_frontier);

    % Updating the result vectors
    [pf_riskSim, pf_RetnSim] = estimatePortMoments(psim, w_sim);
    Ret_Rob_F1(:,n) = pf_RetnSim;
    Vol_Rob_F1(:, n) = pf_riskSim;
    Weights_Rob_F1 = Weights_Rob_F1 + w_sim;

end

% Normalizing the portfolios on the frontier
Weights_Rob_F1 = Weights_Rob_F1 / M_samples;

%% Robust frontier

% Returns
Ret_RobustFront_1 = mean(Ret_Rob_F1, 2);

% Volatilities
Vol_RobustFront_1 = mean(Vol_Rob_F1, 2);


%% Minimum Variance and Maximum Sharpe Portfolio

% Minimum Variance Portfolio
[var_min_rob_1, min_var_index_rob_1] = min(Vol_RobustFront_1);
Portfolio_E = Weights_Rob_F1(:, min_var_index_rob_1);
 
% Maximum Sharpe Portfolio
Frontier_Sharpe_Rob_1 = Ret_RobustFront_1 ./ Vol_RobustFront_1;
[max_sharpe_rob_1, max_sharpe_index_rob_1] = max(Frontier_Sharpe_Rob_1);
Portfolio_G = Weights_Rob_F1(:, max_sharpe_index_rob_1);


%% Plot

figure()
% Robust frontier
plot(Vol_RobustFront_1, Ret_RobustFront_1, 'b', 'LineWidth', 1.5) % Blue
hold on
% Non robust frontier
plot(Vol_F1, Ret_F1, 'k', 'LineWidth', 1.5)

% Minimum Variance Portfolio robust
plot(Vol_RobustFront_1(min_var_index_rob_1), ...
    Ret_RobustFront_1(min_var_index_rob_1), 'o', 'Color', [0.8 0 0], ...
    'MarkerSize', 6, 'MarkerFaceColor', [1 0.4 0.4]) % Red

% Maximum Sharpe Portfolio non robust
plot(Vol_RobustFront_1(max_sharpe_index_rob_1), ...
    Ret_RobustFront_1(max_sharpe_index_rob_1), 'o', 'Color', [0 0.5 0], ...
    'MarkerSize', 6, 'MarkerFaceColor', [0.3 0.8 0.3]) % Green

legend('Robust Frontier (Standard Constraints)', ...
    'Original Frontier (Standard Constraints)', ...
    'Robust Minimum Variance', 'Robust Maximum Sharpe', 'Location', 'best')
title('Robust Frontier Standard Constraints')
xlabel('Volatility')
ylabel('Expected Return')
grid on
hold off


%%

%% Exercise 2 constraints
Vol_Rob_F2 = zeros(number_ptf_frontier, M_samples);
Ret_Rob_F2 = zeros(number_ptf_frontier, M_samples);

%n. sim, n.assets, n portf in a single frontier
Weights_Rob_F2 = zeros(NumAssets, number_ptf_frontier);

for n = 1 : M_samples
    
    % Sampling the expected return vector and covariance matrix
    R = mvnrnd(ExpRet, V, length(LogRet));
    NewExpRet = mean(R);
    NewCov = cov(R);

    % Initializing a portfolio with default constraints
    psim = Portfolio('AssetList', nm, 'NumAssets', NumAssets);
    psim = psim.setAssetMoments(NewExpRet, NewCov);
    psim = psim.setDefaultConstraints();

    % Setting additional constraints to portfolio
    psim = setInequality(psim, A, b);
    psim = setEquality(psim, Aeq, beq);


    % Estimating the frontier
    w_sim = estimateFrontier(psim, number_ptf_frontier);

    % Updating the result vectors
    [pf_riskSim, pf_RetnSim] = estimatePortMoments(psim, w_sim);
    Ret_Rob_F2(:,n) = pf_RetnSim;
    Vol_Rob_F2(:, n) = pf_riskSim;
    Weights_Rob_F2 = Weights_Rob_F2 + w_sim;

end

% Normalizing the portfolios on the frontier
Weights_Rob_F2 = Weights_Rob_F2 / M_samples;

%% Robust frontier

% Returns
Ret_RobustFront_2 = mean(Ret_Rob_F2, 2);

% Volatilities
Vol_RobustFront_2 = mean(Vol_Rob_F2, 2);


%% Minimum Variance and Maximum Sharpe Portfolio

% Minimum Variance Portfolio
[var_min_rob_2, min_var_index_rob_2] = min(Vol_RobustFront_2);
Portfolio_F = Weights_Rob_F2(:, min_var_index_rob_2);
 
% Maximum Sharpe Portfolio
Frontier_Sharpe_Rob_2 = Ret_RobustFront_2 ./ Vol_RobustFront_2;
[max_sharpe_rob_2, max_sharpe_index_rob_2] = max(Frontier_Sharpe_Rob_2);
Portfolio_H = Weights_Rob_F2(:, max_sharpe_index_rob_2);


%% Plot

figure()
% Robust Frontier
plot(Vol_RobustFront_2, Ret_RobustFront_2, 'Color', [0.5 0.5 0.5], ...
    'LineWidth', 1.5) % Dark grey
hold on
% Non robust frontier
plot(Vol_F2, Ret_F2, 'Color', [0.4 0.7 1], 'LineWidth', 1.5)

% Minimum Variance Portfolio robust
plot(Vol_RobustFront_2(min_var_index_rob_2), ...
    Ret_RobustFront_2(min_var_index_rob_2), ...
    'o', 'Color', [0.8 0 0], 'MarkerSize', 6, ...
    'MarkerFaceColor', [1 0.4 0.4]) % Red

% Maximum Sharpe Portfolio robust
plot(Vol_RobustFront_2(max_sharpe_index_rob_2), ...
    Ret_RobustFront_2(max_sharpe_index_rob_2), ...
    'o', 'Color', [0 0.5 0], 'MarkerSize', 6, ...
    'MarkerFaceColor', [0.3 0.8 0.3]) % Green

legend('Robust Frontier (With Constraints)', ...
    'Original Frontier (With Constraints)', 'Min Variance (Robust)', ...
    'Max Sharpe (Robust)', 'Location', 'best')
title('Robust Frontier With Additional Constraints')
xlabel('Volatility')
ylabel('Expected Return')
grid on
hold off


%% Figura combinata

figure()
% Non rubust standard constraints 
plot(Vol_F1, Ret_F1, 'k', 'LineWidth', 1.5)
hold on

% Robust standard constraints
plot(Vol_RobustFront_1, Ret_RobustFront_1, 'b', 'LineWidth', 1.5) 
hold on

% Non robust with constaints
plot(Vol_F2, Ret_F2, 'Color', [0.4 0.7 1], 'LineWidth', 1.5)

% Robust with constraints
plot(Vol_RobustFront_2, Ret_RobustFront_2, 'Color', [0.5 0.5 0.5], ...
    'LineWidth', 1.5)

% Portfolio A
plot(Vol_F1(min_var_index_1), Ret_F1(min_var_index_1), ...
    'o', 'Color', [0.8 0 0], 'MarkerSize', 6, ...
    'MarkerFaceColor', [1 0.4 0.4])
% Portfolio B
plot(VolPtfB, RetPtfB, ...
    'o', 'Color', [0 0.5 0], 'MarkerSize', 6, ...
    'MarkerFaceColor', [0.3 0.8 0.3])   
% Portfolio E
plot(Vol_RobustFront_1(min_var_index_rob_1), ...
    Ret_RobustFront_1(min_var_index_rob_1), 'o', 'Color', [0.8 0 0], ...
    'MarkerSize', 6, 'MarkerFaceColor', [1 0.4 0.4])
% Portfolio G
plot(Vol_RobustFront_1(max_sharpe_index_rob_1), ...
    Ret_RobustFront_1(max_sharpe_index_rob_1), 'o', 'Color', [0 0.5 0], ...
    'MarkerSize', 6, 'MarkerFaceColor', [0.3 0.8 0.3])
% Portfolio C
plot(Vol_F2(min_var_index_2), Ret_F2(min_var_index_2), ...
    'o', 'Color', [0.8 0 0], 'MarkerSize', 6, ...
    'MarkerFaceColor', [1 0.4 0.4]) 
% Portfolio D
plot(VolPtfD, RetPtfD, ...
    'o', 'Color', [0 0.5 0], 'MarkerSize', 6, ...
    'MarkerFaceColor', [0.3 0.8 0.3])
% Portfolio F
plot(Vol_RobustFront_2(min_var_index_rob_2), ...
    Ret_RobustFront_2(min_var_index_rob_2), 'o', 'Color', [0.8 0 0], ...
    'MarkerSize', 6, 'MarkerFaceColor', [1 0.4 0.4])
% Portfolio H
plot(Vol_RobustFront_2(max_sharpe_index_rob_2), ...
    Ret_RobustFront_2(max_sharpe_index_rob_2), 'o', 'Color', [0 0.5 0], ...
    'MarkerSize', 6, 'MarkerFaceColor', [0.3 0.8 0.3])

grid on
legend('Frontier Std Constraints', 'Robust Frontier Std Constraints', ...
    'Frontier With Constraints', 'Robust Frontier With Constraints', ...
    'Location', 'best')
title('Combined Frontiers: Standard and With Constraints')
xlabel('Volatility')
ylabel('Expected Return')
hold off


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Exercise 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Building the views

% Parameters
v = 2; 
tau = 1 / length(LogRet);
P = zeros(v, NumAssets);
q = zeros(v, 1);
Omega = zeros(v);

% View 1: Information Technology outperforms Financial by 2 %
P(1,1) = 1;
P(1,2) = -1;
q(1) = 0.02;

% View 2: Momentum outperforms Low Volatility by 1 %
P(2, 12) = 1;
P(2, 16) = -1;
q(2) = 0.01;

% Views' volatility
Omega(1, 1) = tau .* P(1, :) * V * P(1, :)';
Omega(2, 2) = tau .* P(2, :) * V * P(2, :)';

% From annual view to daily view
bizyear2bizday = 1 / 252;
q = q * bizyear2bizday;
Omega = Omega * bizyear2bizday;

% Plot views distribution
X_views = mvnrnd(q, Omega, 200);
figure()
histogram(X_views)
title('Views Distribution in Black Litterman')
hold off

%% Market implied returns 

% Building the market equilibrium implied by the market 
% capitalization-weighted index
cap = capitalizations(:, 2 : end).Variables;
cap = cap';
wMKT = cap(1 : 16) / sum( cap(1 : 16) );


%% Building the prior distribution

% Parameters for the prior distribution
lambda = 1.2;
mu_mkt = lambda .* V * wMKT;
C = tau .* V;

% Plot prior distribution
X = mvnrnd(mu_mkt, C, 200);
figure()
histogram(X)
title('Prior Distribution in Black Litterman')
hold off


%% Black Litterman

% Parameters
muBL = (inv(C) + P' * (Omega\P)) \ (P' * (Omega\q) + C \ mu_mkt); 
covBL = inv(P' * (Omega\P) + inv(C));


%% Black-Litterman PTF

% Building the portfolio object with default constraints
portBL = Portfolio('NumAssets', NumAssets, 'Name', 'MV with BL');
portBL = setDefaultConstraints(portBL);
portBL = setAssetMoments(portBL, muBL, V+covBL); 

%% Computing the frontier

pBL = estimateFrontier(portBL, 100);
[VolFrontierBL, RetFrontierBL] = estimatePortMoments(portBL, pBL);
[VolPtfI, idxMinVolBL]=min(VolFrontierBL);
Portfolio_I = pBL(:, idxMinVolBL);

% Getting Max sharpe ratio and Portfolio moments
Portfolio_L = estimateMaxSharpeRatio(portBL);
[VolPtfL, RetPtfL] = estimatePortMoments(portBL, Portfolio_L);


%% Plot

figure()
% Plot the robust frontier
plot(VolFrontierBL, RetFrontierBL, 'LineWidth', 1.5)
hold on

% Plot the efficient frontier 
plot(Vol_F1, Ret_F1, 'k', 'LineWidth', 1.5) % Black

% Portfolio I
plot(VolPtfI, RetFrontierBL(idxMinVolBL), ...
    'o', 'Color', [0.8 0 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.4 0.4]) % Red
% Portfolio L
plot(VolPtfL, RetPtfL, ...
    'o', 'Color', [0 0.5 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.3 0.8 0.3]) % Green

% Portfolio A
plot(Vol_F1(min_var_index_1), Ret_F1(min_var_index_1), ...
    'o', 'Color', [0.8 0 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [1 0.4 0.4]) % Red
% Portfolio B
plot(VolPtfB, RetPtfB, ...
    'o', 'Color', [0 0.5 0], 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.3 0.8 0.3]) % Green

% Grid and labels
grid on
legend('Frontier Black Litterman', 'Efficient Frontier',...
    'Portfolio I', 'Portfolio L', 'Location', 'best')
title('Black Litterman Frontier')
xlabel('Volatility')
ylabel('Expected Return')
hold off


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Exercise 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Equally Weighted Ptf
Portfolio_EW = 1 / NumAssets * ones(NumAssets, 1);


%% Maximum diversification Portfolio

% Constraints
Aeq = ones(1,16);
beq = 1;
lb = zeros(1,16);
ub = ones(1,16);
x0 = Portfolio_EW;

% total Cyclical (2,4,6,10,11) greater than 20%
b = -0.2;
A = zeros(1,16);
A(1,[2,4,6,10,11]) = -1;

% Optimization with the previously defined constrints
[Portfolio_M, fval] = fmincon(@(x) -getDiversificationRatio(x, LogRet), ...
    x0, A, b, Aeq, beq, lb, ub, @nonlincon);


%% Maximum Entropy Portfolio

% Optimization
Aeq = ones(1,16);
beq = 1;
lb = zeros(1,16);
ub = ones(1,16);
x0 = zeros(16,1);
x0(1,1) = 1;
b= -0.2;
A = zeros(1,16);
A(1,[2,4,6,10,11])= -1;

% Max Entropy in Volatilities
Portfolio_N = fmincon(@(x) -getEntropy(getVolContributions(x, LogRet)), ...
    x0, A, b, Aeq, beq, lb, ub, @nonlincon); % From theory


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Exercise 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parameters

% Computing standardized returns
RetStd = (LogRet-ExpRet)./ std(LogRet);

% Defining target volatility
target_sigma = 0.7;

% Setting threshold for explicability
explicability_threshold = 90;


%% Performing PCA

% PCA on standardized returns
[factorLoading, factorRet, latent, ~, explained] = pca(RetStd);

% Factors that explain the cumulative variance up to the threshold
cumExplainedVar = cumsum(explained);
numFactors = find(cumExplainedVar > explicability_threshold, 1);

% Reducing dimensions to the optimal number of factors
% Eigenvectors
factorLoading = factorLoading(:, 1 : numFactors); 
% Returns computed with the first numFactors factors
factorRet = factorRet(:, 1 : numFactors);
% Eigenvalues
latent_factors = latent(1 : numFactors); 

% Computing reconstructed returns and residuals
reconReturn = (factorRet * factorLoading').*std(LogRet) + ExpRet;
unexplainedRetn = LogRet - reconReturn;

% Computing residual covariance with the unexplained part of the returns
unexplainedCovar = diag(cov(unexplainedRetn));
D = diag(unexplainedCovar);

% Computing the asset covariance matrix
covarFactor = cov(factorRet);
covarAsset = factorLoading * covarFactor * factorLoading' + D;

% Plots about explicability
figure();
bar(1 : numFactors,latent_factors/sum(latent));
title('Explained Variance by Principal Components');
xlabel('Principal Component');
ylabel('Proportion of Total Variance');
hold off

figure();
plot(cumExplainedVar(1:numFactors), 'Color', [0.4 0.7 1], 'LineWidth', 1.5);
hold on;
scatter(1:numFactors, cumExplainedVar(1:numFactors), ...
    50, 'filled', 'MarkerFaceColor', [0.4 0.7 1], 'MarkerEdgeColor', [0.4 0.7 1]);
grid on;
title('Cumulative Explained Variance');
xlabel('Number of Principal Components');
ylabel('Percentage Explained');
hold off


%% Computing Portfolio P

% Objective function
objectiveFunc = @(x) - ( (ExpRet * x) );

% Nonlinear constraint ---> portfolio volatility <= target_sigma
nonlincon = @(x) deal(...
    sqrt( (factorLoading' * x)' * covarFactor * (factorLoading' * x)...
    + x' * D * x) - target_sigma, []);

% Inputs for numerical optimization algorithm
% Initial point ensuring weights sum to 1
x0 = rand(NumAssets, 1);
x0 = x0 / sum(x0); 

% Setting default constraints
lb = zeros(NumAssets, 1); 
ub = ones(NumAssets, 1);
Aeq = ones(1, NumAssets); 
beq = 1;

% Running optimization
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');
[Portfolio_P, ~] = fmincon(objectiveFunc, x0, [], [], Aeq, beq, lb, ub,...
    nonlincon, options);


%% Compute Portfolio Metrics
% Expected return of the portfolio
ExpLogRet_ptf_P = ExpRet * Portfolio_P;

% Portfolio equity curve
ret = prices_val(2:end, :) ./ prices_val(1:end-1, :);
equity_p = cumprod(ret * Portfolio_P);
equity_p = 100 * equity_p / equity_p(1);

% Portfolio volatility
portfolio_volatility = sqrt(Portfolio_P' * covarAsset * Portfolio_P);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Exercise 7 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Verifying Normality assumption
qqplot(LogRet);

for ii=1:NumAssets
    % Perform the Anderson-Darling test
    disp('Anderson-Darling test')
    % Define the non-standard normal distribution
    customDist = makedist('Normal', 'mu', mean(LogRet(:,ii)), ...
        'sigma', std(LogRet(:,ii)));
    h = adtest(LogRet(:,ii), 'Distribution', customDist);
    if h == 0
        disp('Data follows a normal distribution.');
    else
        disp('Data does not follow a normal distribution.');
    end
end


% Grid dimensions
numRows = 4;
numCols = 4;

% Create a new figure
figure('Name', 'Histograms with Normal Distribution Curves');

for ii = 1:NumAssets
    % Select the current subplot
    subplot(numRows, numCols, ii);
    
    % Plot the normalized histogram
    histogram(LogRet(:,ii), 'Normalization', 'pdf');
    hold on;
    
    % Compute the theoretical normal distribution
    x = linspace(min(LogRet(:,ii)), max(LogRet(:,ii)), 100);
    y = normpdf(x, mean(LogRet(:,ii)), std(LogRet(:,ii)));
    plot(x, y, 'r', 'LineWidth', 2);
    
    % Add a title for each subplot
    title(['Asset ', num2str(ii)]);
    
    hold off;
end

% Add a global title to the figure
sgtitle('Histograms of Log Returns with Theoretical Normal Curves');


%% Optmization con target value VaR
x0 = rand(size(LogRet,2),1);
x0 = x0./sum(x0);
lb = zeros(1, size(LogRet,2)); 
ub = ones(1, size(LogRet,2));
Aeq = ones(1, size(LogRet,2));
beq = 1;
pval = 0.05; 


%% Maximizing Sharpe_Ratio_daily 
Portfolio_Q = fmincon(@(x) -fun_VarCovMethod(x,LogRet, pval), x0, [],[], Aeq, beq, lb, ub);

VaR_95_test_daily = quantile(Portfolio_Q'*LogRet',0.05);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Exercise 8 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Portfolio Table
table(nm', ...
    Portfolio_A, Portfolio_B, Portfolio_C, Portfolio_D, Portfolio_E, ...
    Portfolio_F, Portfolio_G, Portfolio_H, Portfolio_I, Portfolio_L, ...
    Portfolio_M, Portfolio_N, Portfolio_P, Portfolio_Q, ...
    'VariableNames', ...
    ["Asset Name", "Portfolio_A", "Portfolio_B", "Portfolio_C", "Portfolio_D", ...
    "Portfolio_E", "Portfolio_F", "Portfolio_G", "Portfolio_H", ...
    "Portfolio_I", "Portfolio_L", "Portfolio_M", "Portfolio_N", ...
    "Portfolio_P", "Portfolio_Q"])

%% Portfolios metrics

Metrics_EW = getPerformanceMetrics(Portfolio_EW, ret, 0, 0);

Metrics_A = getPerformanceMetrics(Portfolio_A, ret, 0, Metrics_EW.relRC);
Metrics_B = getPerformanceMetrics(Portfolio_B, ret, 0, Metrics_EW.relRC);
Metrics_C = getPerformanceMetrics(Portfolio_C, ret, 0, Metrics_EW.relRC);
Metrics_D = getPerformanceMetrics(Portfolio_D, ret, 0, Metrics_EW.relRC);
Metrics_E = getPerformanceMetrics(Portfolio_E, ret, 0, Metrics_EW.relRC);
Metrics_F = getPerformanceMetrics(Portfolio_F, ret, 0, Metrics_EW.relRC);
Metrics_G = getPerformanceMetrics(Portfolio_G, ret, 0, Metrics_EW.relRC);
Metrics_H = getPerformanceMetrics(Portfolio_H, ret, 0, Metrics_EW.relRC);
Metrics_I = getPerformanceMetrics(Portfolio_I, ret, 0, Metrics_EW.relRC);
Metrics_L = getPerformanceMetrics(Portfolio_L, ret, 0, Metrics_EW.relRC);
Metrics_M = getPerformanceMetrics(Portfolio_M, ret, 0, Metrics_EW.relRC);
Metrics_N = getPerformanceMetrics(Portfolio_N, ret, 0, Metrics_EW.relRC);
Metrics_P = getPerformanceMetrics(Portfolio_P, ret, 0, Metrics_EW.relRC);
Metrics_Q = getPerformanceMetrics(Portfolio_Q, ret, 0, Metrics_EW.relRC);

weights = [Portfolio_EW Portfolio_A Portfolio_B Portfolio_C ...
    Portfolio_D Portfolio_E Portfolio_F Portfolio_G Portfolio_H ...
    Portfolio_I Portfolio_L Portfolio_M Portfolio_N Portfolio_P ...
    Portfolio_Q];


%% Plot performances on In-Sample data
plot_performance(weights, ret, dates_)


%% Metrics Table

% Initialize a cell array to store metrics
portfolioNames = {'EW', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', ...
    'L', 'M', 'N', 'P', 'Q'};
metrics = ["AnnRet", "AnnVol", "Sharpe", "MaxDD", "Calmar", "DR"]; 

% Store the computed metrics in a matrix
results = zeros(length(portfolioNames), length(metrics));

% Populate the metrics for each portfolio
Metrics = {Metrics_EW, Metrics_A, Metrics_B, ...
    Metrics_C, Metrics_D, Metrics_E, ...
    Metrics_F, Metrics_G, Metrics_H, ...
    Metrics_I, Metrics_L, Metrics_M, ...
    Metrics_N, Metrics_P, Metrics_Q};

for i = 1:length(portfolioNames)
    currentMetrics = Metrics{i};
    results(i, :) = [currentMetrics.AnnRet, currentMetrics.AnnVol, ...
                     currentMetrics.Sharpe, currentMetrics.MaxDD, ...
                     currentMetrics.Calmar, currentMetrics.DR];
end

% Create a table
metricsTable = array2table(results, 'VariableNames', metrics, ...
    'RowNames', portfolioNames);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PART B %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Out of samples data

outSample_start_dt = datetime('01/01/2024', 'InputFormat', 'dd/MM/yyyy');
outSample_end_dt = datetime('25/10/2024', 'InputFormat', 'dd/MM/yyyy');
outSample_rng = timerange(outSample_start_dt, outSample_end_dt, 'closed'); 

outSample_subsample = myPrice_dt(outSample_rng,:); 
outSample_prices_val = outSample_subsample.Variables;
outSample_dates_ = outSample_subsample.Time;

%% Processing data

% Calculate returns
outSample_ret = outSample_prices_val(2 : end, :) ./ ...
    outSample_prices_val(1 : end - 1, :);


%% Portfolios metrics

outSample_Metrics_EW = getPerformanceMetrics(Portfolio_EW, ...
    outSample_ret, 0, 0);

outSample_Metrics_A = getPerformanceMetrics(Portfolio_A, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_B = getPerformanceMetrics(Portfolio_B, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_C = getPerformanceMetrics(Portfolio_C, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_D = getPerformanceMetrics(Portfolio_D, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_E = getPerformanceMetrics(Portfolio_E, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_F = getPerformanceMetrics(Portfolio_F, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_G = getPerformanceMetrics(Portfolio_G, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_H = getPerformanceMetrics(Portfolio_H, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_I = getPerformanceMetrics(Portfolio_I, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_L = getPerformanceMetrics(Portfolio_L, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_M = getPerformanceMetrics(Portfolio_M, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_N = getPerformanceMetrics(Portfolio_N, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_P = getPerformanceMetrics(Portfolio_P, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);
outSample_Metrics_Q = getPerformanceMetrics(Portfolio_Q, ...
    outSample_ret, 0, outSample_Metrics_EW.relRC);


%% Plot performances over Out-Of-Sample data
plot_performance(weights, outSample_ret, outSample_dates_)


%% Metrics Table
% Store the computed metrics in a matrix
outSample_results = zeros(length(portfolioNames), length(metrics));

% Populate the metrics for each portfolio
outSample_Metrics = {outSample_Metrics_EW, outSample_Metrics_A, ...
    outSample_Metrics_B, ...
    outSample_Metrics_C, outSample_Metrics_D, outSample_Metrics_E, ...
    outSample_Metrics_F, outSample_Metrics_G, outSample_Metrics_H, ...
    outSample_Metrics_I, outSample_Metrics_L, outSample_Metrics_M, ...
    outSample_Metrics_N, outSample_Metrics_P, outSample_Metrics_Q};

for i = 1:length(portfolioNames)
    currentMetrics = outSample_Metrics{i};
    outSample_results(i, :) = [currentMetrics.AnnRet, ...
        currentMetrics.AnnVol, currentMetrics.Sharpe, ...
        currentMetrics.MaxDD, currentMetrics.Calmar, currentMetrics.DR];
end

% Create a table
outSample_metricsTable = array2table(outSample_results, ...
    'VariableNames', metrics, 'RowNames', portfolioNames);


%% Portfolio Pies

figure();

% Define the grid size for subplots
rows = 3; % Fixed to 3 rows
cols = 5; % Fixed to 5 columns
subplot(3,5,1)
 pieHandle = pie(weights(:,1));
title(portfolioNames{1}, 'FontSize', 10);

% Loop through each portfolio and create a pie chart
for i = 2 : size(weights, 2)
    % Extract weights for the current portfolio
    portfolioWeights = weights(:, i);
    
    % Filter weights greater than 0.0001 (for visual clarity)
    validIndices = portfolioWeights > 0.0001;
    filteredWeights = portfolioWeights(validIndices);
    filteredNames = nm(validIndices);
    
    % Create a subplot for the portfolio
    subplot(rows, cols, i);
    
    % Create a pie chart for the filtered weights
    pieHandle = pie(filteredWeights);
    
    % Loop through the pie chart handles to adjust labels
    for j = 1:length(pieHandle)
        if isa(pieHandle(j), 'matlab.graphics.primitive.Text')
            % Get the index for the current label
            idx = ceil(j / 2); % Each label is followed by a number (counting every two handles)
            
            % Check if the weight is greater than 1% (0.01), and set the label accordingly
            if filteredWeights(idx) > 0.01
                pieHandle(j).String = sprintf('%.1f%%', filteredWeights(idx) * 100); % Show percentage for >1%
            else
                pieHandle(j).String = ''; % Hide label for weights <1%
            end
        end
    end
    
    % Add the portfolio name as the title
    title(portfolioNames{i}, 'FontSize', 10);
end

% Add a shared legend for the assets (one legend for all pie charts)
legendLabels = nm; % Asset names
subplot(rows, cols, 1); % Use the last subplot for the legend
hold on;

hold off;

h=legend(legendLabels, 'Position',[0.02 0.4 0.1 0.2]);

axis off;
sgtitle('Portfolio Weights', 'FontSize', 14);
set(gcf, 'Position', [100, 00, 1400, 800]);