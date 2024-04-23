clear all
clc

%% Generando los datos del enunciado.

rng(14)               % Semilla. 

beta = [1;2;4];
N = 1000;
G = 40;                 % Tamaño muestral por grupo.
K = 3;                   % Número de variables (con constante).

N_g = N/G;              % Número de grupos.

% Error por grupos: Primero se genera un vector fila con números aleatorios. Posteriormente se 
% replica el resultado de la primera fila, para las 39 filas restantes.
% Finalmente, se aplica el comando reshape para dejarlo como un vector
% columna.

v_g = randn(1,N_g);
v_g = repmat(v_g, G, 1);

% X_1ig

X_1ig = zeros(G,N_g);   % Se aprovecha de generar la matriz X_1ig.

% Para cada observación (g,i) toma el valor solicitado.
for g_id = 1:G
    for i = 1:N_g
        if v_g(g_id,i) < 0 
            X_1ig(g_id, i) = 1*randn(1,1) + 3; % Transpose dimensions
        else
            X_1ig(g_id, i) = 1*randn(1,1) + 5; % Transpose dimensions
        end
    end
end

% column_means = mean(X_ig); % Para probar que cada columna (grupo) tiene la media
% solicitada.

v_g = reshape(v_g, [], 1); % Vector columna de errores por grupo
X_1ig = reshape(X_1ig, [], 1); % Vector columna de errores por grupo

% X_2ig

X_2ig = 1*randn(N,1) + 5;

% Identificador y error para cada grupo: Se genera un identificador por
% grupo, 

g_id = floor((0:size(X_1ig, 1)-1) / 25) + 1;     % Indicador de grupo
g_id = reshape(g_id, [],1);

ep_ig = 1*randn(N,1);


cons = repmat(1,N,1) ;
X = [cons X_1ig X_2ig];

Y_ig = X*beta + ep_ig + v_g;

tabla = [X_1ig, X_2ig, ep_ig, v_g];

%% Pregunta 1: Supuestos del Modelo de Regresión Lineal.

% Y_i independientes e identicamente distribuidas: No, debido a que dada
% la presencia de los grupos las observaciones no son independientes. 


%%  Pregunta 2: Coeficientes de Mínimos Cuadrados Ordinarios.

beta_gorro = inv(X'*X)*X'*Y_ig
    % b_0 = -0.2552
    % b_1 = 2.3359
    % b_2 = 3.9861
% Nota: Al aumtar N, beta 0 no se acerca al valor poblacional.

eig_gorro = Y_ig - X*beta_gorro;


%% Pregunta 3. Cálculo de errores estándares.

% Para esta pregunta, el desarrollo se basa principalmente en el capítulo 4
% del Hansen, en donde se analiza la construcción de las matrices de
% varianza y covarianza para el término de error con y sin presencia de
% heterocedasticidad, como también para datos agrupados.

% (b.) Homocedasticidad y ausencia de autocorrelación.

    s = (eig_gorro'*eig_gorro)/(N-K);    % Varianza estimada del error.
    se = sqrt(s*diag(inv(X'*X)));    % HC0 fórmula.

% (c.) Errores estándares robustos.

    D = diag(eig_gorro.^2);     % Matriz diagonal del término de error.
    se_robusto = sqrt(diag(inv(X'*X)*(X'*D*X)*inv(X'*X))); % HC1 fórmula.

% (d.) Errores estándares agrupados.

    cluster_sums = zeros(G,K);
    for j = 1:K
        cluster_sums(:,j) = accumarray(g_id,X(:,j).*eig_gorro);
    end

    omega = cluster_sums'*cluster_sums;
    a_n = G/(G-1)*(N-1)/(N-K);

    V_clusterizado = a_n*inv(X'*X)*omega*inv(X'*X);     % (4.47) fórmula Hansen.
    se_clusterizado = sqrt(diag(V_clusterizado));  

% Resultados:

display(se);
display(se_robusto);
display(se_clusterizado);


%% Pregunta 4. Testeo de hipótesis.

t1 = (beta_gorro - 1)./se;
t2 = (beta_gorro - 1)./se_robusto;
t3 = (beta_gorro - 1)./se_clusterizado;

errores_estandar = [se, se_robusto, se_clusterizado];

% Resultados:

testeo = [t1, t2, t3];
display(testeo)

% No se observa la presencia de heterocedasticidad, puesto al comparar los 
% estadísticos con y sin corrección por heterocedasticidad (columnas 1 y 2). 
% A su vez, no se observa grandes diferencias en términos del valor del estadístico 
% y su significancia estadística. 

% Lo que sí se observa, es la influencia de datos agrupados, la
% cual impacta principalmente a X_1ig. Esto puesto que dicha variable fue
% construida imponiendo una distribución centrada en una media distinta para 
% cada tramo dependiendo del término de error por grupo. De esta forma,
% variables no observadas a nivel de grupo influyen bastante en la
% inferencia causal de este modelo.


%% Pregunta 5. Efecto fijo por desviaciones en torno a la media (Capt. 15 Hansen).

% Para la siguiente pregunta, se aplicará la "Within Transformation". Dicha transformación
% busca construir un estimador beta que es invariante en el término de error por grupos v_g.
% Para esto, genera una variable 'punto' que corresponde a las desviaciones
% en torno a la media ade dicha variable (dependiente o independiente).

% En estricto rigor, dado:
%   Y_{ig} = \beta_0 + \beta1X_{1ig} + \beta_2X_{2ig} + \epsilon_{ig} + v_{g}
%   \bar{Y_{ig}} = \beta_0 + \beta1\bar{X}_{1ig} + \beta_2\bar{X}_{2ig} + \bar{\epsilon}_{ig} + v_{g}
% Tendremos que la diferencia en torno a la media eliminará tanto la
% constante, como el error por grupos. De esta forma, si 
%       \dot{Y_{ig}} = Y_{ig} - \bar{Y_{ig}}
%       \dot{X_{ig}} = X_{ig} - \bar{X_{ig}}
%       \dot{\epsilon_{ig}} = \epsilon_{ig} - \bar{\epsilon_{ig}},
% el modelo a estimar será:
%   \dot{Y_{ig}} = \beta1\dot{X_{ig}} + \beta_2\dot{X_{2ig}} + \dot{\epsilon_{ig}}

% Calculando las medias:
Yig_mean = accumarray(g_id, Y_ig, [], @mean); % Cluster specific mean.
X1ig_mean = accumarray(g_id, X_1ig, [], @mean);
X2ig_mean = accumarray(g_id, X_2ig, [], @mean);

% Calculando las desviaciones en torno a la media (within transformation) para cada variable.
Y_punto = Y_ig - Yig_mean(g_id);    
X1ig_punto = X_1ig - X1ig_mean(g_id);
X2ig_punto = X_2ig - X1ig_mean(g_id);
X_punto = [X1ig_punto, X2ig_punto]; % Sin constante puesto que Y-Y_mean la elimina. 

K = 2       % Dado que la constante desaparece, el número de incógnitas pasa a ser 2.

beta_ef = inv(X_punto'*X_punto)*(X_punto'*Y_punto)   % Variación entre cada grupo.

eig_fe = Y_punto - X_punto*beta_ef;

% Testeo de Hipótesis:

% (i.) Homocedasticidad y ausencia de autocorrelación.

    s_p5 = (eig_fe'*eig_fe)/(N-K);    
    se_p5 = sqrt(s*diag(inv(X_punto'*X_punto)));   

% (ii.) Errores estándares robustos.

    D_p5 = diag(eig_fe.^2);    
    se_robusto_p5 = sqrt(diag(inv(X_punto'*X_punto)*(X_punto'*D_p5*X_punto)*inv(X_punto'*X_punto)));


% (iii.) Errores estándares agrupados.

    cluster_sums_p5 = zeros(G,K);
    for j = 1:K
        cluster_sums_p5(:,j) = accumarray(g_id,X_punto(:,j).*eig_fe);
    end

    omega_p5 = cluster_sums_p5'*cluster_sums_p5;
    a_n = G/(G-1)*(N-1)/(N-K);

    V_clusterizado_p5 = a_n*inv(X_punto'*X_punto)*omega_p5*inv(X_punto'*X_punto);
    se_clusterizado_p5 = sqrt(diag(V_clusterizado_p5));

% (iv.) Testeo de hipótesis. 
    t1_p5 = (beta_ef - 1)./se_p5;
    t2_p5 = (beta_ef - 1)./se_robusto_p5;
    t3_p5 = (beta_ef - 1)./se_clusterizado_p5;

    errores_estandar_p5 = [se_p5, se_robusto_p5, se_clusterizado_p5]

% Resultados:

testeo_p5 = [t1_p5, t2_p5, t3_p5]

% Hay pérdida de significancia estadística para cada una de las variables,
% pero ambos son estadísticamente significativos. 

%% Teorema de FWGL y equivalencia con desviaciones en torno a la media.

% Como se pudo ver en la parte anterior, el modelo de desviaciones a la
% media nos genera una regresión de la forma: 

% \dot{Y_{ig}} = \beta1\dot{X_{ig}} + \beta_2\dot{X_{2ig}} + \dot{\epsilon_{ig}},

% lo cual se parece bastante a una regresión particionada. 

%  Aplicando el teorema de Frisch-Waugh-Lovell (FWL), tendremos que para X1ig,
%  se elimina toda la información proveniente de X2ig utilizando la matriz
%  aniquiladora de X2ig (M2). 

P_2 = X2ig_punto*inv(X2ig_punto'*X2ig_punto)*X2ig_punto';       % Matriz proyección.
M_2 = eye(length(P_2)) - X2ig_punto*inv(X2ig_punto'*X2ig_punto)*X2ig_punto'; % Matriz aniquilación.

beta1_p6 = inv(X1ig_punto'*M_2*X1ig_punto)*X1ig_punto'*M_2*Y_punto    %\hat{\beta_{1}} = =(X_1'M_2X_1)^{-1}X_1'M_2Y

e1_tilde = Y_ig - beta1_p6*X_1ig;      

% Para obtener X_2ig: Se elimina toda la información de X_1ig. 

P_1 = X1ig_punto*inv(X1ig_punto'*X1ig_punto)*X1ig_punto';       % Matriz proyección.
M_1 = eye(length(P_2)) - X1ig_punto*inv(X1ig_punto'*X1ig_punto)*X1ig_punto'; % Matriz aniquilación.

beta2_p6 = inv(X2ig_punto'*M_1*X2ig_punto)*X2ig_punto'*M_1*Y_punto 

beta_fwg = [beta1_p6, beta2_p6]';

% Comparación:

display(beta_ef);
display(beta_fwg);

% Dado que \beta_1 = 2.07 y \beta_1 = 1.4 es el mismo con ambos procedimientos, el resultado con ambas
% metodologías equivalente. Esto podría deberse a que: ....



% Nota:
% ***** Habría que pensar por qué el teorema de FWG elimina el efecto por
% grupos (v_g) y la constante (b_0), para que podamos pensar que la
% regresión particionada por FWG y la regresión por desviaciones a la media
% es la misma. 

% Nota 2:
% Tambien tendríamos que pensar por qué b_2 difiere bastante de 4 al
% incluir efectos fijos. Quizás podría ser que eliminar el 1 de la
% constante, y eliminar el error por grupo sesga el resultado. Igualmente,
% en el enunciado solo te piden testear b_1, que supongo es la variable de
% interés relevante. 

