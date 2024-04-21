% ----------------------------------------------------------------------- 
%                       TAREA 1 - ECONOMETRÍA I - ME
%        Grupo 3: F. Anguita, N. Bastias, C. Cid, T Munoz O. & R. Salazar
% ----------------------------------------------------------------------- 

%% Directorios por usuario
clc;clear;

%% 0. SIMULACIÓN DE BASE DE DATOS

% Modelo a estimar:
% Y_{ig} = \beta_0 + \beta_1 * X_{1ig} + \beta_2 * X_{2ig} + \epsilon_{ig}
% + \nu_{g}

% Donde:
% \epsilon_{ig} = N(0,1)
% \nu_{g} = N(0,1)
% X_{1ig} = N(3,1) if \nu_{g} < 0
% X_{1ig} = N(5,1) if \nu_{g} >= 0
% X_{2ig} = N(5,1)

% Establecemos parámetros
b = [1, 2, 4];                          % Vector beta
n = 1000;                               % Número de personas
g = 40;                                 % Número de grupos
n_g = n/g;                              % Cantidad de personas por grupo
rng(23);                                 % Semilla

% Generamos residuo ig y x2
e_ig = normrnd(0,1,[n, 1]);             %                       (1000x1)           
x_2ig = normrnd(5,1,[n, 1]);            %                       (1000x1)

% Inicializamos matrices
x_1 = zeros(n_g, g);                    % X1                      (25x40)  
y_ig = zeros(n, 1);                     % Y                       (1000x1)
v = zeros(n_g, 1);                      % v                       (40x1)

% Iteramos para cada grupo el generar su residuo y valores de x1
% residuos por grupo quedan en vector v(40x1) y x1 por grupo quedan en matriz x_1(25x40)(ixg)
for z = 1:g          
    v(z,1) = normrnd(0,1);         
    if  v(z,1) < 0                         
        x_1(:, z) = normrnd(3,1,[n_g, 1]);  
    else
        x_1(:, z) = normrnd(5,1,[n_g, 1]);
    end
end

% Vectorizamos matriz x_1 en x_1ig(1000x1)
x_1ig = reshape(x_1,[],1);

% Generamos v_g(1000x1) repitiendo v de cada grupo en cada obs del grupo
indx = 1;
for i = 1:length(v)
    for j = 1:25
        vg(indx) = v(i);
        indx = indx + 1;
    end
end
v_g = reshape(vg,[],1);

% Generamos residuo del modelo
c_ig = e_ig + v_g;

% Generamos variable dependiente
y_ig = b(1) + b(2)*x_1ig + b(3)*x_2ig + c_ig;

%% 1. SUPUESTOS MRL QUE CUMPLEN LOS DATOS SIMULADOS

% a. Observaciones vienen de una distribucion comun y son independientes (iid)
% b. X e Y satisfacen Y = X'*beta + e y E(e|X)=0 [E(Y|X) = X'*beta) -> Mejor predictor lineal igual a la esperanza
% c. Segundos momentos finitos (varianza acotada) E(Y^2)<inf, E||X||^2<inf
% d. Ausencia de multicolinealidad peftecta -> Q_xx = E(X'X)>0
% e. Homocedasticidad de los errores -> E(e^2|X) = sigma^2 (varianza cte)

%% 2. COEFICIENTES MCO 

% Especificación MCO: y_ig = b_0 + b_1*x_1ig + b_2*x_2ig + c_ig

% Obtenemos los estimadores/coeficientes por fórmula
b_1 = inv(x_1ig'*x_1ig)*(x_1ig'*y_ig);
b_2 = inv(x_2ig'*x_2ig)*(x_2ig'*y_ig);
x_0 = ones(n, 1);
b_0 = inv(x_0'*x_0)*(x_0'*y_ig);                                   

% Alternativa matricial 
x_ig = [ones(n, 1), x_1ig, x_2ig];
b_mco=inv(x_ig'*x_ig)*(x_ig'*y_ig);

% Pregunta seria: pq dan distinto las dos formas?

%% 3. ERRORES ESTANDAR (p.136 - 140 / 150 Hansen)

% Vale la pena definir tipo a = x'x; i_a = inv(a) pa simplificar el código?

% 3.1. Homocedasticidad y ausencia de correlacion

r = y_ig - x_ig*b_mco;                  % Residuos MCO  
s2 = (1./(n-3))*(r'*r);                 % Estimador varianza del error s2 = (r'*r)/(n-3)
mvc = s2*inv(x_ig'*x_ig);               % Matriz varianza covarianza
ee = sqrt(diag(mvc));                   % Errores estándar 

% 3.2. Robustos (heterocedasticidad) 

r = y_ig - x_ig*b_mco;                 % Residuos MCO 
d = diag(r.^2);                        % d = matriz diagonal de r^2 
mvc_w = inv(x_ig'*x_ig)*(x_ig'*d*x_ig)*inv(x_ig'*x_ig);      % Matriz de White (var-cov robusta)
ee_r = sqrt(diag(mvc_w));              % Errores estándar robustos

% 3.3. Agrupados (clausterizados: indepencia entre grupos no al interior de ellos)

%PENDIENTEEEE
grupo = floor((0:size(x_ig, 1)-1) / 25) + 1;     % Indicador de grupo
grupo = reshape(grupo, [],1);                       
x = [x_ig, grupo];                               % X con columna de grupo

r = y_ig - x_ig*b_mco;
dg = dummyvar(grupo);                  % Dummy por grupo
d = diag(r.^2);                        %
mvc_a = inv(x_ig'*x_ig)*dg'*d*dg*inv(x_ig'*x_ig);  % Matriz var-cov agrupada
ee_a = sqrt(diag(mvc_a));

%% 4. TEST DE HIPOTESIS NULA

% H0: b_1 = 1 -> H1: b_1 <> 1

% Test con ee bajo homocedasticidad
tt_1 = b_mco(2)/ee(2);
pv_1 = 2 * (1 - tcdf(abs(t_test), length(y_ig) - size(x_ig, 2)));

% Test con ee robustos
tt_2 = b_mco(2)/ee_r(2);
pv_2 = 2 * (1 - tcdf(abs(t_test), length(y_ig) - size(x_ig, 2)));

% Test con ee agrupados
tt_3 = b_mco(2)/ee_g(2);
pv_3 = 2 * (1 - tcdf(abs(t_test), length(y_ig) - size(x_ig, 2)));

% ALTERNATIVA 2
R = [0; 1; 0];
t1 = (R'*b_mco)/sqrt((ee.^2)*R'*inv(x_ig'*x_ig)*R);
t2 = (R'*b_mco)/sqrt((ee_r.^2)*R'*inv(x_ig'*x_ig)*R);
t3 = (R'*b_mco)/sqrt((ee_a.^2)*R'*inv(x_ig'*x_ig)*R);

%% 5. MODELO CON EFECTOS FIJOS (p.635 hansen)

% Especificación con efectos fijos: y_ig = b_0 + b_1*x_1ig + b_2*x_2ig + v_g + e_ig
% Estimador efectos fijos: b = (x'mx)^(-1)(x'my)

y_ig = b(1) + b(2)*x_1ig + b(3)*x_2ig + v_g + e_ig;

% Obtenemos matriz de aniquilación
P_1 = x_1ig*inv(x_1ig'*x_1ig)*x_1ig';
P_2 = x_2ig*inv(x_2ig'*x_2ig)*x_2ig';

I = eye(n,n);
M_1 = I - P_1;
M_2 = I - P_2;

b1_fe = inv(x_1ig'*M_2*x_1ig)*x_1ig'*M_2*y_ig; 
b2_fe= inv(x_2ig'*M_1*x_2ig)*x_2ig'*M_1*y_ig;


%% 6. FWL Y MODELO DE EFECTOS FIJOS

% Definición FWL 
% Formula de regresion particionada
I = eye(n_g, n_g);
P_1 = zeros(n_g, n_g);
P_2 = zeros(n_g, n_g);
M_1 = zeros(n_g, n_g);
M_2 = zeros(n_g, n_g);

for z = 1:g

%hay que ajustar para que se vaya llenando la matriz
P_1(:, z) = x_1ig*inv(x1_1ig'*x_1ig)*x_1ig';
P_2(:, z) = x_2ig*inv(x2_1ig'*x_2ig)*x_2ig';

M_1(:, z) = I - P_1;
M_2(:, z) = I - P_2;

b_1 = inv(x_1ig'*M_2*x_1ig)*x_1ig'*M_2*y_ig; 
b_2 = inv(x_2ig'*M_1*x_2ig)*x_2ig'*M_1*y_ig;

end

%% 7. REPETICION CON DISTINTA DISTRIBUCION DE X1

clc;clear;

% Parámetros, e_ig, x_2ig, v_g se mantienen igual 
b = [1, 2, 4];                          
n = 1000;                              
g = 40;                                 
n_g = n/g;                             
rng(23)                                 

e_ig = normrnd(0,1,[n, 1]);                        
x_2ig = normrnd(5,1,[n, 1]);           

v = normrnd(0,1,[g,1]);  
indx = 1;
for i = 1:length(v)
    for j = 1:25
        vg(indx) = v(i);
        indx = indx + 1;
    end
end
v_g = reshape(vg,[],1);

% Volvemos a simular x_1ig (que ahora depende de w_i) y recalculamos y_ig
x_1 = zeros(n, 1);                       
y_ig = zeros(n, 1);                     
w_i = zeros(n, 1);                      

for z = 1:n          
    w_i(z,1) = unifrnd(0,1);         
    if  w_i(z,1) < 0.5                         
        x_1(:, z) = normrnd(3,1,[n_g, 1]);  
    else
        x_1(:, z) = normrnd(5,1,[n_g, 1]);
    end
end

c_ig = e_ig + v_g;

y_ig = b(1) + b(2)*x_1ig + b(3)*x_2ig + c_ig;

% Repetimos desarrollo de preguntas 1. a 5. 

%1. SUPUESTOS MRL QUE CUMPLEN LOS DATOS SIMULADOS

    % a. Observaciones vienen de una distribucion comun y son independientes (iid)
    % b. X e Y satisfacen Y = X'*beta + e y E(e|X)=0 [E(Y|X) = X'*beta) -> Mejor predictor lineal igual a la esperanza
    % c. Segundos momentos finitos (varianza acotada) E(Y^2)<inf, E||X||^2<inf
    % d. Ausencia de multicolinealidad peftecta -> Q_xx = E(X'X)>0
    % e. Homocedasticidad de los errores -> E(e^2|X) = sigma^2 (varianza cte)

% 2. COEFICIENTES MCO 

    % Especificación MCO: y_ig = b_0 + b_1*x_1ig + b_2*x_2ig + c_ig

    % Obtenemos los estimadores/coeficientes por fórmula
    b_1 = inv(x_1ig'*x_1ig)*x_1ig'*y_ig;
    b_2 = inv(x_2ig'*x_2ig)*x_2ig'*y_ig;
    b_0                                     %??

    % Alternativa matricial 
    x_ig = [ones(n, 1), x_1ig, x_2ig];
    b_mco=inv(x_ig'*x_ig)*(x_ig'*y_ig);

% 3. Ecnwldcvw

% 4. 

% 5.  


%% 8. RESUMA LO APRENDIDO



