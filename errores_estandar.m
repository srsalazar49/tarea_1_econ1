% Definimos una funcion de los errores estandar que consideran el supuesto
% de homocedasticidad y ausencia de correlacion

function [var_bgorro, e_estandar] = errores_estandar(s_2,X)

% Primero se define cual es la matriz de varianza y covarianza
% Por definicion este es var(b|x) = s^2 (X'* X)^-1
var_bgorro = s_2 * ((X' * X)^(-1));

% Ahora bien, los errores estandar vendrian siendo la raiz de lo anterior
e_estandar = sqrt(var_bgorro);
end