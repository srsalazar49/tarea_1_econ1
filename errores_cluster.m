% Definimos una funcion de los errores cluster, escalado por como lo
% sugiere el Hansen

function [var_cluster, ee_cluster] = errores_cluster(N, K, G, X, e_cluster)

% Calculamos omega a partir de los errores clusterizados estimados antes
omega = e_cluster' * e_cluster;

% Calculamos ahora el parametro que escala la varianza
A = ((N - 1)/(N - K)) * ((G)/(G - 1));

% Ahora, definimos la matriz de varianza y covarianza clusterizada y escalada
% por N, K y G
var_cluster = A *((X' * X)^(-1)) * omega * ((X' * X)^(-1));

% Finalmente, los errores estandar son la raiz de la diagonal de lo anterior
ee_cluster = sqrt(diag(var_cluster));
end
