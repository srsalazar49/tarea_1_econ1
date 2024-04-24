% ----------------------------------------------------------------------- %
%                       TAREA 1 - ECONOMETRIA I - ME
%        Grupo 3: F. Anguita, N. Bastias, C. Cid, T Munoz O. & R. Salazar
% ----------------------------------------------------------------------- %

%% Agregando los directorios por usuario

clc;clear;

% Path Tamara
if strcmp(char(java.lang.System.getProperty('user.name')),'tamaramunoz') ==1
    data ='/Users/tamaramunoz/Desktop/1st semester ME/Econometría I/Tarea/Main/';
    
% Path Fernanda 
elseif strcmp(char(java.lang.System.getProperty('user.name')),'ferna') ==1
    data ='C:\Users\ferna\Desktop\ME_Otoño 2024\Econometría I\Tarea 1'; 
    
% Path Romulo
elseif strcmp(char(java.lang.System.getProperty('user.name')),'HP') ==1
    data ='C:\Users\HP\Documents\MAESTRÍA\1. Primer Semestre\Econometría I\Tareas\Tarea 1';
end

%% 0. SIMULACIÓN DE BASE DE DATOS

% Modelo a estimar:
% Y_ig = beta_0 + beta_1 * X_1ig + beta_2 * X_2ig + epsilon_ig + nu_g

% Donde:
% epsilon_ig = N(0,1)
% nu_g = N(0,1)
% X_1ig = N(3,1) if nu_g < 0
% X_1ig = N(5,1) if nu_g >= 0
% X_2ig = N(5,1)

% Definimos los parametros iniciales
beta = [1, 2, 4];                       % Vector beta
N = 1000;                               % Numero de personas
grupo = 40;                             % Numero de grupos
n_g = N/grupo;                          % Cantidad de personas por grupo

% Definimos una matriz auxiliar que guardara los resultados de cada individuo 
% indexados por grupo, luego definiremos por separado los vectores
% correspondientes a cada variable.
matriz = zeros(N,5); 


% Generaramos un loop a nivel de grupo para ir creando todas las variables.
% Se itera por cada grupo y dentro de ello, se itera a nivel de individuo.

rng(14) % fijando la semilla
j = 1; % variable auxiliar

for g = 1:grupo
    
    % Indexamos a cada individuo por grupo
    matriz((j:j+24),1) = g; % columna 1
    
    % Calculamos el error a nivel de grupo
    v_g = normrnd(0,1); 
    matriz((j:j+24),2) = v_g'; % columna 2
    
    % Calculamos el error individual por grupo
    epsilon_ig = normrnd(0,1,[1,n_g]);
    matriz((j:j+24),3) = epsilon_ig'; % columna 3
        
    % Calculamos el X_2ig por grupo
    x_2ig = normrnd(5,1,[1,n_g]);  
    matriz((j:j+24),4) = x_2ig';  % columna 4
        
    % Calculamos el X_1ig condicional al valor que toma 'v_g'
    if  v_g < 0      % valores < 0                   
       x_1ig = normrnd(3,1,[1,n_g]);  
    else             % valores >= a 0
       x_1ig = normrnd(5,1,[1,n_g]);
    end
       
    matriz((j:j+24),5) = x_1ig';  % columna 5
        
    % Para el grupo siguiente, ahora se deben considerar las siguientes 25
    % personas, por lo que a 'j' se le suma 25 para volver a iterar
    j = j + 25;

end

% Ahora bien, el termino de error del modelo Y_ig consider la suma de los
% errores individuales y grupales, por lo que se puede reescribir como
% e_ig = epislon_ig + v_g
e_ig = matriz(:,3) + matriz(:,2);

% Dejamos en 2 matrices separados los X_1ig y X_2ig calculados
% anteriormente
X_1ig = matriz(:,5);
X_2ig = matriz(:,4);

% Finalmente estimamos el modelo inicial con los betas verdaderos
y_ig = beta(1) + beta(2) * X_1ig + beta(3) * X_2ig + e_ig;

% A partir de esto, todas las variables tienen dimensiones de (1000x1),
% salvo los beta que tienen dimension (3x1)

%% 2. COEFICIENTES MCO 
% Estime los coeficientes de MCO. Interprete sus resultados.

% Ahora debemos calcular los diferentes betas de MCO donde debemos utilizar
% la regresion particionada para ello considerando que estamos en presencia
% de un intercept.

% No olvidar que nuestro modelo consta de lo siguiente:
% Y_{ig} = \beta_0 + \beta_1 * X_{1ig} + \beta_2 * X_{2ig} + e_{ig}

% donde e_{ig} = \epsilon_{ig} + \nu_g

% Podemos aprovechar de la forma matricial de MCO para estimar en este caso
% los betas

% Como tenemos una constante de beta_0, esto es equivalente a que este
% multiplicara un X = 1, por lo que podemos crear una matriz de Xs el cual
% contenga en su primera columna una constante

% El Y ya lo tenemos de antes y tiene dimension 1000 x 1, solo le cambiamos
% el nombre
Y = y_ig;
% Ahora generamos la matrix 'X' para estimar los beta de MCO

% Definimos x_0
x_0 = ones(n, 1);

% Cambiamos el nombre de los x_1i y x_2i
X1 = matriz(:,5);
X2 = matriz(:,4);

% Ahora definimos el X que tendra dimension 1000x3
X = [x_0 X1 X2];

% Importamos la base de datos para tener una comparacion con Stata
matrix = [Y X e_ig matriz(:,1)];
writematrix(matrix,'test.csv') 

% Calculamos ahora los beta con la funcion de MCO que definimos
% previamente:
[beta_gorro] = MCO(Y,X); %calculamos los betas con una 
% funcion que definimos (me dio lo mismo que Stata)

% Redondeamos los betas para que tengan hasta 3 decimales
beta_gorro_2 = round(beta_gorro, 2);

% Exportamos los resultados en una tabla
tabla_P2 = table(['\beta_0';'\beta_1';'\beta_2'],... 
    [beta_gorro_2(1);beta_gorro_2(2); beta_gorro_2(3)]);
writetable(tabla_P2,'tabla_P2.txt','Delimiter',' ')  
type 'tabla_P2.txt'

%% 3. ERRORES ESTANDAR

% Calcule los siguientes errores estandar:

% 3.1. Asumiendo homocedasticidad y ausencia de correlacion

% Calculamos entonces los errores estandar asumiendo homocedasticidad como:
% SD = s
% Ya que no observamos a sigma directamente

% Sabemos que s^2 = (1/(n-k)) * e'e

% El e gorro se obtiene de la estimacion de MCO (se incluye entonces el 
% e_gorro en la función definida de MCO)

% Corremos nuevamente el codigo de MCO para tener nuevamente los betas sin
% redondear
[beta_gorro, e_gorro] = MCO(Y,X);

% Ahora corremos la funcion de s^2
N = n;
[K,s_2] = s2(N, beta_gorro, e_gorro);

% Luego, corremos la funcion de los errores estandar que estan en funcion
% de s^2
[var_bgorro, e_estandar] = errores_estandar(s_2,X);
% (tambien me dio lo mismo que stata)

% 3.2. Errores estandar robustos

% Estos errores ahora no asumen homocedasticidad, sino heterocedasticidad,
% por lo que el calculo es ligeramente diferente. Utilizando la definicion
% del estimador de la varianza HC1 del Hansen (el que el libro recomienda), 
% definimos previamente la funcion de ello y lo corremos
[var_robust, ee_robust] = errores_robustos(N, K ,X, e_gorro);
% (tambien me dio lo mismo que stata)

% 3.3. Errores estandar agrupados

% Ahora, para calcular los errores estandar agrupados, necesitamos realizar
% la estimacion inicial de los beta a nivel de grupo ahora y no a nivel de
% cada individuo. Por ello, agrupamos por grupo a cada variable del modelo
% para estimar a Y nuevamente.

% Lista de los diferentes grupos
grupos = matriz(:,1);

% Definimos ahora el numero de clusters
G = g;

% Clusterizamos el error ahora considerando los X originales y los residuos
e_cluster = zeros(G,K);

% Generamos un loop para que clusterice los errores por grupo
for j = 1:K
    e_cluster(:,j) = accumarray(grupos, (X(:,j))'.*e_gorro');
end

% Hacemos ahora la estimacion de los errores estandar robustos
[var_cluster, ee_cluster] = errores_cluster(N, K, G, X, e_cluster);
% (entrega el mismo valor que stata)

%% 4. TEST DE HIPOTESIS NULA

% Quiero testar la hipotesis beta_1 = 1, entonces mi matriz R debe ser:
R = [0 1 0];

% Entonces, probamos los diferentes test t con los diferentes errores
% Errores estandar 
ttest_1 = abs(((R * beta_gorro) - 1)/(e_estandar(2)));
p_value1 = 2 * (1 - tcdf(ttest_1, N-K)); % p-value para 2 colas

% Errores robustos
ttest_2 = abs(((R * beta_gorro) - 1)/(ee_robust(2)));
p_value2 = 2 * (1 - tcdf(ttest_2, N-K)); % p-value para 2 colas

% Errores clusters
ttest_3 = abs(((R * beta_gorro) - 1)/(ee_cluster(2)));
p_value3 = 2 * (1 - tcdf(ttest_3, N-K)); % p-value para 2 colas


%% 5. MODELO CON EFECTOS FIJOS

% Requiere que agreguemos la matriz de aniquilacion ahora para cada X
% M = I - X(X'X)^-1 X'
% M = I - P

% Calculamos la proyeccion para el X generico
P = X * ((X'*X)^(-1))* X';
I = eye(N);
M = I - P;

% Dummies por grupo
Dummy = dummyvar(grupos);
D = diag(Dummy);
M_D = M * Dummy;

X_aux = [X1 X2 Dummy];
X = X_aux;

% Calculo del efecto fijo por grupos
[beta_gorro, e_gorro] = MCO(Y,X);
% (entrega lo mismo que Stata)

% Falta correr lo de los diferentes errores y ttest que es lo mismo
%% 6. FWL Y MODELO DE EFECTOS FIJOS

% Parte 1 de FWL
% Definimos como X1 la matriz de dummies de cada grupo
X1 = Dummy;
% Definimos como X2 la matriz de las 2 variables x_1ig y x_2ig
X2 = [matriz(:,5) matriz(:,4)]; 

% Llamamos la funcion de regresion particionada
[beta_1,beta_2] = FWL(Y,X1,X2);

% Hacemos lo mismo ahora pero considerando la diferencia de medias
% Calculando las medias:

yig_mean = accumarray(grupos, y_ig, [], @mean); % cluster mean y_ig
x1ig_mean = accumarray(grupos, (matriz(:,5)), [], @mean);  % cluster mean x_1ig
x2ig_mean = accumarray(grupos, (matriz(:,4)), [], @mean); % cluster mean x_2ig

% Calculamos las desviaciones en torno a la media (within transformation) 
% para cada variable.
y_punto = y_ig - yig_mean(grupos);    
x1ig_punto = (matriz(:,5)) - x1ig_mean(grupos);
x2ig_punto = (matriz(:,4)) - x2ig_mean(grupos);

% Definimos ahora las variables para estimar por MCO
X = [x1ig_punto, x2ig_punto]; % sin constante ya que su desviacion a la media es 0
Y = y_punto;

% Corremos el MCO
[beta_gorro, e_gorro] = MCO(Y,X);

% Obtenemos desviaciones sobre la media de grupo de las variables 
% Comprobamos que lo de la Fer da lo mismo
my = accumarray(grupos, y_ig, [], @mean);
mx1 = accumarray(grupos,(matriz(:,5)), [], @mean);
mx2 = accumarray(grupos,(matriz(:,4)), [], @mean);

Y = y_ig - my(grupos);
x1 = (matriz(:,5)) - mx1(grupos);
x2 = (matriz(:,4)) - mx2(grupos);

X = [x1 x2];

% Corremos el MCO
[beta_gorro, e_gorro] = MCO(Y,X);

%% 7. REPETICION CON DISTINTA DISTRIBUCION DE X1

% Repetimos nuevamente todo el procedimiento de la 1 cambiando ahora como
% se define el x_1ig
clc;clear;

% Parametros
beta = [1, 2, 4];                       % Vector beta
n = 1000;                               % Numero de personas
grupo = 40;                             % Numero de grupos
n_g = n/grupo;                          % Cantidad de personas por grupo

% Matriz que guardara los resultados de cada individuo indexados por grupo
matriz = zeros(n,4); % Matriz que guardara los resultados de cada individuo

% Generar un loop a nivel de grupo para ir generando todas las variables
% Se itera por cada grupo y dentro de ello, se itera a nivel de individuo,
% cada resultado se guarda en una matriz diferente que luego se van a
% concatenar

rng(14) % fijando la semilla
j = 1; % variable auxiliar
for g = 1:grupo
    
    % Indexamos a cada individuo por grupo
    matriz((j:j+24),1) = g;
    
    % Calculamos el error a nivel de grupo
    v_g = normrnd(0,1);
    matriz((j:j+24),2) = v_g'; % guarda el resultado en la columna 2 de la 
    % matriz
        
    % Calculamos el error individual por grupo
    epsilon_ig = normrnd(0,1,[1,n_g]);
    matriz((j:j+24),3) = epsilon_ig'; % guarda el resultado en la columna 3
        
    % Calculamos el X_{2ig} por grupo
    x_2ig = normrnd(5,1,[1,n_g]);  
    matriz((j:j+24),4) = x_2ig';  % guarda el resultado en la columna 4
end

% Calculamos ahora el w_i
w_i = unifrnd(0,1,[n,1]); 

j = 1;
x = zeros(n,1);
% Hacemos un loop para ahora calcular a x_1ig
for i = 1:n
    
    % Calculamos el X_{1ig} condicional al valor que toma 'w_i'
    if  w_i < 0.5      % valores < 0.5                   
       x_1ig = normrnd(3,1);  
    else             % valores >= a 0.5
       x_1ig = normrnd(5,1);
    end
       
    x(i,1) = x_1ig;  % guarda el resultado en la columna 6
    j = j+1;
end


% Ahora bien, el termino de error del modelo Y_{ig} consider la suma de los
% errores individuales y grupales, por lo que se puede reescribir como
% e_{ig} = epislon_{ig} + v_g
e_ig = matriz(:,3) + matriz(:,2);

% Finalmente entonces, el modelo a estimar considera los diferentes betas
% con los diferentes coeficientes, columnas de la matriz y los errores
y_ig = beta(1) + beta(2) * x + beta(3) * matriz(:,4) + e_ig;

% A partir de esto, todas las variables tienen dimensiones de (1000x1)



