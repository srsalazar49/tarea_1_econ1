% A continuacion se crea una funcion de MCO generica de manera matricial,
% el cual permite calcular los coeficientes de interes

% Definimos por otro lado el error estandar para cálculos de
% varianza/covarianza

function [beta_gorro, e_gorro] = MCO(Y,X)

% Por formula, beta = (X'X)^-1 (X'Y)
beta_gorro = (X' * X)\(X' * Y);

% Por formula, e = Y - X * beta_gorro
e_gorro = Y - (X * beta_gorro);

end

