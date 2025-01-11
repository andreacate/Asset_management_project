function [Aeq, beq, A, b] =...
    matrix_constraints(equalities_number, inequalities_number, NumAssets)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function that computes the matrix used for setting the constrints on our
% portfolio object
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Equalities constraints %%%%%%%
Aeq = zeros(equalities_number, NumAssets);
Aeq(1, 7) = 1;
Aeq(2, 16) = 1;
beq = zeros(equalities_number, 1);


%%%%%% Inequalities constraints %%%%%%
A = zeros(inequalities_number, NumAssets);
b = zeros(inequalities_number, 1);

% The total exposure to sensible sectors has to be greater than 10%
A(1, [1 5 8]) = -1;
b(1) = - 0.1;

% The total exposure on cyclical sectors has to be lower than 30%
A(2, [2 4 6 10 11]) = 1;
b(2) = 0.3;

% The total exposure must be smaller than 80%
sectors = 1 : 11;
A(3, sectors) = 1;
b(3) = 0.8;

end