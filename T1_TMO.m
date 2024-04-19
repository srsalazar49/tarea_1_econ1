% ----------------------------------------------------------------------- %
%                       TAREA 1 - ECONOMETRIA I - ME
%        Grupo 3: F. Anguita, N. Bastias, C. Cid, T Munoz O. & R. Salazar
% ----------------------------------------------------------------------- %

%% Agregando los directorios por usuario
clc;clear;

% Path Tamara
if strcmp(char(java.lang.System.getProperty('user.name')),'tamaramunoz')==1
    data='/Users/tamaramunoz/Desktop/1st semester ME/Econometría I/Tarea/Main/';
    
% Path Fernanda 
elseif strcmp(char(java.lang.System.getProperty('user.name')),'ferna')==1
    data='C:\Users\ferna\Desktop\ME_Otoño 2024\Econometría I\Tarea 1'; 
end

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

% Parametros
b = [1, 2, 4];                          % Vector beta
n = 1000;                               % Numero de personas
g = 40;                                 % Numero de grupos
n_g = n/g;                              % Cantidad de personas por grupo

% Creamos un vector que contiene valores del 1-40 para poder identificar a
% cada grupo correspondiente
%group = randi([1 25],1,1000);

% Genero residuo ig y x2
e_ig = normrnd(0,1,[n, 1]);             %                       (1000x1)           
x_2ig = normrnd(5,1,[n, 1]);            %                       (1000x1)

% Inicializo matrices
x_1ig = zeros(n_g, g);                  % X1                      (25x40)  
y_ig = zeros(n, 1);                     % Y                       (1000x1)
v = zeros(g,1);                         % v                       (40x1)

for z = 1:g          
     
    %v(z,1) = normrnd(0,1,[n_g, 1]); 
    v_g = normrnd(0,1,[n_g, 1]);   
                
    if  v_g < 0                         
        x_1ig(:, z) = normrnd(3,1,[n_g, 1]);  
    else
        x_1ig(:, z) = normrnd(5,1,[n_g, 1]);
    end

end

X1 = reshape(x_1ig,[],1);
%falta el vercor del v con 1000x1

c_ig = e_ig + v_g;

y_ig = b(1) + b(2)*x_1ig + b(3)*x_2ig + c_ig;

%% 1. SUPUESTOS MRL QUE CUMPLEN LOS DATOS SIMULADOS

% a. Observaciones vienen de una distribucion comun y son independientes (iid)
% b. X e Y satisfacen Y = X'*beta + e y E(e|X)=0 [E(Y|X) = X'*beta) -> Mejor
% predictor lineal igual a la esperanza
% c. Segundos momentos finitos (varianza acotada) E(Y^2)<inf, E||X||^2<inf
% d. Ausencia de multicolinealidad peftecta -> Q_xx = E(X'X)>0
% e. Homocedasticidad de los errores -> E(e^2|X) = sigma^2 (varianza cte)

%% 2. COEFICIENTES MCO 

% Por formula betas
b_1 = inv(x_1ig'*x_1ig)*x_1ig'*y_ig;
b_2 = inv(x_2ig'*x_2ig)*x_2ig'*y_ig;

% Como dice el internet (ya me perdi, cuantos beta hay que sacar?)
b_mco = zeros(3, g);

for z = 1:g
    
    y = y_ig(:, z) - mean(y_ig(:, z));
    x_1ig_v2 = x_1ig(:, z) - mean(x_1ig(:, z));
    x_2ig_v2 = x_2ig(:, z) - mean(x_2ig(:, z));
    
    x = [ones(n_g, 1), x_1ig_v2, x_2ig_v2];
   
    b_mco(:, z) = inv(x'*x)*x'*y;       % formula matricial beta
end

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

b_1 = inv(x_1ig'*M_2*x_1ig)*x_1ig'*M_2*y_ig; %escalares
b_2 = inv(x_2ig'*M_1*x_2ig)*x_2ig'*M_1*y_ig;
end

%% 3. ERRORES ESTANDAR

% 3.1. Homocedasticidad y ausencia de correlacion

ee = (1./(n-g))*((error'*error)\(X'*X));        % errores estandar
ee = sqrt(diag(ee));
t_stat = beta./ee;

% 3.2. Robustos


% 3.3. Agrupados

%% 4. TEST DE HIPOTESIS NULA

%% 5. MODELO CON EFECTOS FIJOS


%% 6. FWL Y MODELO DE EFECTOS FIJOS

%% 7. REPETICION CON DISTINTA DISTRIBUCION DE X1

%% 8. RESUMA LO APRENDIDO

% Está fucking larga la tarea

