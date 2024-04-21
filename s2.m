% Creamos la funcion que permite calcular s^2
function [K,s_2] = s2(N, beta_gorro, e_gorro)

% El k que se obtiene es el valor del largo del vector de betas estimado:
K = length(beta_gorro);

% Ahora bien, el s^2 se estima utilizando como input, el e_gorro, N y k
s_2   = (e_gorro' * e_gorro)/(N - K);
end