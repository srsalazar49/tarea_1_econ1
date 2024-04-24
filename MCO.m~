% Definimos una funcion MCO generica que permite calcular los coeficientes 
% de manera matricial junto a los residuos del modelo

function [beta_gorro, e_gorro] = MCO(Y,X)

% Por formula, beta = (X'X)^(-1) (X'Y)
beta_gorro = (X' * X)\(X' * Y);

% Por formula, e = Y - X * beta_gorro
e_gorro = Y - (X * beta_gorro);

end

