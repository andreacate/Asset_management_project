function [c,ceq] = nonlincon(x)

load('capitalizations.mat')
%table_cap = readtable(strcat(path_map, filename));
%cap = table_cap(:,2:end).Variables;
cap = capitalizations(:,2:end).Variables;
cap = cap';
wMKT = cap(1:16)/sum(cap(1:16));
c = 0.2-sum(abs(x-wMKT));
ceq=[];