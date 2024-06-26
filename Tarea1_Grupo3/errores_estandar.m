% Definimos una funcion de los errores estandar que consideran el supuesto
% de homocedasticidad y ausencia de correlacion
function [var_bgorro, ee_estandar] = errores_estandar(s_2,X)

% Primero se define cual es la matriz de varianza y covarianza
% Por definicion este es var(b|x) = s^2 (X'* X)^-1
var_bgorro = s_2 * ((X' * X)^(-1));

% Ahora, los errores estandar vendrian siendo la raiz de la diagonal de var
ee_estandar = sqrt(diag(var_bgorro));
end