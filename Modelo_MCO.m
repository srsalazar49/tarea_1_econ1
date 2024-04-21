% A continuacion se crea una funcion de MCO generica de manera matricial,
% el cual permite calcular los coeficientes de interes
function [beta_gorro] = MCO(Y,X)

% Por formula, beta = (X'X)^-1 (X'Y)
beta_gorro = (X' * X)\(X' * Y);

end