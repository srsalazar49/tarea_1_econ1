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

% Aqui estimamos ahora los coeficientes de MCO del modelo, utilizando las
% variables definidas anteriormente. Para ello, utilizaremos la notacion
% matricial de las variables.

% Renombramos el Y por simplicidad, el cual tiene dimension (1000x1)
Y = y_ig;

% Ahora generamos la matrix 'X' el cual contiene a todas nuestras variables
% mas un vector de 1s para estimar el beta_0:
x_0 = ones(N, 1); % definimos x_0

% Juntamos ahora los 3 en un 'X' que tendra dimension (1000x3)
X = [x_0 X_1ig X_2ig];

% Utilizando una funcion definida por nosotros previamente llamada 'MCO',
% calculamos los coeficientes:
[beta_gorro, e_gorro] = MCO(Y,X);
display(beta_gorro)

% Exportamos ahora los resultados en una tabla
tabla_P2 = table(['\beta_0';'\beta_1';'\beta_2'],... 
    [round(beta_gorro(1),4);round(beta_gorro(2),4);round(beta_gorro(3),4)]);
writetable(tabla_P2,'tabla_P2.txt','Delimiter',' ')  
type 'tabla_P2.txt'

% Como extra, importamos la base de datos para poder tener una comparacion
% con Stata de nuestros resultados
matrix = [Y X e_ig matriz(:,1)];
writematrix(matrix,'test.csv') 

%% 3. ERRORES ESTANDAR

% Calculamos ahora diferentes tipos de errores estandar

% 3.1. Asumiendo homocedasticidad y ausencia de correlacion
% Estos errores se calculan como la raiz de 's', donde utilizamos una
% funcion definida previamente por nosotros que una estimacion de la
% varianza:
[K,s_2] = s2(N, beta_gorro, e_gorro);

% Luego, corremos la funcion de los errores estandar que estan en funcion
% de s^2
[ee_estandar] = errores_estandar(s_2,X);
display(ee_estandar)


% 3.2. Errores estandar robustos

% Estos errores ahora asumen heterocedasticidad. Utilizando la definicion
% del estimador de la varianza HC1 del Hansen, utilizamos la funcion 
% previamente definida de ello:
[ee_robust] = errores_robustos(N, K ,X, e_gorro);
display(ee_robust)


% 3.3. Errores estandar agrupados

% Ahora, para calcular los errores estandar agrupados, clusterizamos el
% error del modelo bajo la definicion del Hansen, el cual considera el
% numero de clusters en los datos:

% Creamos una lista de los indices de los grupos por individuo
grupos = matriz(:,1);

% Definimos ahora el numero de clusters
G = g;

% Clusterizamos el error ahora considerando los X originales y los residuos
e_cluster = zeros(G,K);

% Generamos un loop para que clusterice los errores por grupo
for j = 1:K
    e_cluster(:,j) = accumarray(grupos, (X(:,j))'.*e_gorro');
end

% Hacemos ahora la estimacion de los errores estandar clusterizados
[ee_cluster] = errores_cluster(N, K, G, X, e_cluster);
display(ee_cluster)

%% 4. TEST DE HIPOTESIS NULA

% Queremos testar ahora la hipotesis de beta_1 = 2, por lo que la matriz R 
% debe ser:
R = [0 1 0];
c = 2; % este es el valor de la hipotesis a testear

% Entonces, hacemos los diferentes test-t para los diferentes errores
% Errores estandar 
ttest_1 = abs(((R * beta_gorro) - c)/(ee_estandar(2)));
p_value1 = 2 * (1 - tcdf(ttest_1, N - K)); % p-value para 2 colas

% Errores robustos
ttest_2 = abs(((R * beta_gorro) - c)/(ee_robust(2)));
p_value2 = 2 * (1 - tcdf(ttest_2, N - K)); % p-value para 2 colas

% Errores clusters
ttest_3 = abs(((R * beta_gorro) - c)/(ee_cluster(2)));
p_value3 = 2 * (1 - tcdf(ttest_3, N - K)); % p-value para 2 colas

% Matriz con los estadisticos t
ttest = [ttest_1 ttest_2 ttest_3];

% Matriz con los p-value
p_value = [p_value1 p_value2 p_value3];

% Mostrando los resultados
display(ttest)
display(p_value)

%% 5. MODELO CON EFECTOS FIJOS

% Requiere que agreguemos ahora una matriz que contenga un vector de 1s por
% posicion de cada grupo; esto vendria siendo un desglose de la constante y
% por ello, estimamos ahora el modelo para tener un coeficiente por grupo
% sin considerar la constante del inicio

% Creamos una matriz de dummies por grupo
Dummy = dummyvar(grupos);

% Armamos nuevamente ahora la variable 'X'
X = [X_1ig X_2ig Dummy];

% Calculamos ahora el MCO, el cual considera el efecto fijo por grupo
[beta_gorro, e_gorro] = MCO(Y,X);

% Calculamos ahora nuevamente los diferentes tipos de error:
% Error estandar
[K,s_2] = s2(N, beta_gorro, e_gorro);
[var_bgorro, ee_estandar] = errores_estandar(s_2,X);
display(ee_estandar)

% Error robusto
[var_robust, ee_robust] = errores_robustos(N, K ,X, e_gorro);
display(ee_robust)

% Error clusterizado
% Clusterizamos el error
e_cluster = zeros(G,K);

% Generamos un loop para que clusterice los errores por grupo
for j = 1:K
    e_cluster(:,j) = accumarray(grupos, (X(:,j))'.*e_gorro');
end

[var_cluster, ee_cluster] = errores_cluster(N, K, G, X, e_cluster);
display(ee_cluster)

% Testamos ahora nuevamente la hipotesis nula
% Definimos nuevamente a R
R = [1 zeros(1,41)];

% Errores estandar 
ttest_1 = abs(((R * beta_gorro) - 1)/(ee_estandar(2)));
p_value1 = 2 * (1 - tcdf(ttest_1, N - K)); % p-value para 2 colas

% Errores robustos
ttest_2 = abs(((R * beta_gorro) - 1)/(ee_robust(2)));
p_value2 = 2 * (1 - tcdf(ttest_2, N - K)); % p-value para 2 colas

% Errores clusters
ttest_3 = abs(((R * beta_gorro) - 1)/(ee_cluster(2)));
p_value3 = 2 * (1 - tcdf(ttest_3, N - K)); % p-value para 2 colas

% Matriz con los estadisticos t
ttest = [ttest_1 ttest_2 ttest_3];

% Matriz con los p-value
p_value = [p_value1 p_value2 p_value3];

% Mostrando los resultados
display(ttest)
display(p_value)

%% 6. FWL Y MODELO DE EFECTOS FIJOS

% Ahora particionamos al X en 2 variables, donde el primero sera el grupo
% de dummies de los efectos fijos y el segundo del X_1ig y X_2ig originales

% Definimos como X1 la matriz de dummies de cada grupo
X1 = Dummy;

% Definimos como X2 la matriz de las 2 variables x_1ig y x_2ig
X2 = [X_1ig X_2ig]; 

% Llamamos la funcion de regresion particionada previamente definida por
% nosotros
[beta_1,beta_2] = FWL(Y,X1,X2);
display(beta_1)
display(beta_2)


% Ahora hacemos lo mismo ahora pero considerando la desviacion de medias
% Calculando las medias:
mean_y = accumarray(grupos, y_ig, [], @mean); % cluster mean y_ig
mean_x1 = accumarray(grupos, X_1ig, [], @mean); % cluster mean x_1ig
mean_x2 = accumarray(grupos, X_2ig, [], @mean); % cluster mean x_2ig

% Calculamos las desviaciones en torno a la media (within transformation): 
Y = y_ig - mean_y(grupos);
x1 = (X_1ig - mean_x1(grupos));
x2 = (X_2ig - mean_x2(grupos));

% Definimos la nueva matriz de los x
X = [x1 x2];

% Corremos el MCO
[beta_gorro, e_gorro] = MCO(Y,X);
display(beta_gorro)


%% 7. REPETICION CON DISTINTA DISTRIBUCION DE X1

% Repetimos nuevamente todo el procedimiento de la 1 cambiando ahora como
% se define el X_1ig

clc;clear;

% Fijamos nuevamente los parametros
beta = [1, 2, 4];                       % Vector beta
N = 1000;                               % Numero de personas
grupo = 40;                             % Numero de grupos
n_g = n/grupo;                          % Cantidad de personas por grupo

% Utilizamos nuevamente una matriz auxiliar para las variables
matriz = zeros(n,4);

% Generamos un loop el cual va a calcular el 'v_g', 'epislon_ig' y 'x_2ig',
% de manera separada calculamos el 'w_i' y 'x_1ig'

rng(14) % fijando la semilla
j = 1; % variable auxiliar
for g = 1:grupo
    
    % Indexamos a cada individuo por grupo
    matriz((j:j+24),1) = g;
    
    % Calculamos el error a nivel de grupo
    v_g = normrnd(0,1);
    matriz((j:j+24),2) = v_g'; % columna 2
        
    % Calculamos el error individual por grupo
    epsilon_ig = normrnd(0,1,[1,n_g]);
    matriz((j:j+24),3) = epsilon_ig'; % columna 3
        
    % Calculamos el X_2ig
    x_2ig = normrnd(5,1,[1,n_g]);  
    matriz((j:j+24),4) = x_2ig';  % columna 4
end

% Todos los generados son de dimension (1000x1)


% Calculamos ahora un vector de (1000x1) de w_i
w_i = unifrnd(0,1,[n,1]); 

x = zeros(n,1); % creamos matriz de los X_1ig con un loop en torno a los w_i

% Hacemos un loop para ahora calcular a x_1ig
for i = 1:n
    
    % Calculamos el X_1ig condicional al valor que toma 'w_i'
    if  w_i < 0.5      % valores < 0.5                   
       x_1ig = normrnd(3,1);  
    else             % valores >= a 0.5
       x_1ig = normrnd(5,1);
    end
       
    x(i,1) = x_1ig;  % % guarda el resultado
end

% Renombramos ahora en un vector aparte los X respectivos
X_1ig = x_1ig;
X_2ig = matriz(:,4);

% Volvemos a agrupar los errores tal como se hizo anteriormente
e_ig = matriz(:,3) + matriz(:,2);

% Estimamos nuevamente el modelo original con los nuevos X_1ig
y_ig = beta(1) + beta(2) * X_1ig + beta(3) * X_2ig + e_ig;




