% Hacemos el calculo de la regresion particionada
function [beta_1,beta_2, e_FWL] = FWL(Y,X1,X2)

% Hacemos la proyeccion de cada X correspondiente
P1 = (X1 * ((X1' * X1)^(-1)) * X1'); % proyeccion del X1
P2 = (X2 * ((X2' * X2)^(-1)) * X2'); % proyeccion del X2

% Calculamos ahora las aniquilaciones de cada X
M1 = (eye(length(X1)) - P1); % aniquilacion X1
M2 = (eye(length(X2)) - P2); % aniquilacion X2

% Aislamos cada beta correspondiente a cada variable
beta_1 = (X1' * M2 * X1)\(X1' * M2 * Y); % beta del X1
beta_2 = (X2' * M1 * X2)\(X2' * M1 * Y); % beta del X2

% Definimos ahora el residuo analogo a MCO
e_FWL = Y - X1 * beta_1 - X2 * beta_2;

end