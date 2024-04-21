% Definimos una funcion de los errores estandar robustos ante
% heterocedasticidad

function [var_robust, ee_robust] = errores_robustos(N, K ,X, e_gorro)

% Primero, calculamos la matriz diagonal de los errores al cuadrado
e_gorro2 = e_gorro.^2;

% Obtenemos la diagonal de ello
D = diag(e_gorro2);

% Ahora, definimos la matriz de varianza y covarianza robusta y escalada
% por N y K
var_robust = (N/(N - K)) *((X' * X)^(-1)) * X' * D * X * ((X' * X)^(-1));

% Finalmente, los errores robustos son la raiz de la diagonal de lo anterior
ee_robust = sqrt(diag(var_robust));
end