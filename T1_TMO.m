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
beta = [1, 2, 4];                       % Vector beta
n = 1000;                               % Numero de personas
grupo = 40;                             % Numero de grupos
n_g = n/grupo;                          % Cantidad de personas por grupo

% Matriz que guardara los resultados de cada individuo indexados por grupo
matriz = zeros(n,5); % Matriz que guardara los resultados de cada individuo

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
        
    % Calculamos el X_{1ig} condicional al valor que toma 'v_g'
    if  v_g < 0      % valores < 0                   
       x_1ig = normrnd(3,1,[1,n_g]);  
    else             % valores >= a 0
       x_1ig = normrnd(5,1,[1,n_g]);
    end
       
    matriz((j:j+24),5) = x_1ig';  % guarda el resultado en la columna 5
        
    % Para el grupo siguiente, ahora se deben considerar las siguientes 25
    % personas, por lo que a 'j' se le suma 25 para volver a iterar
    j = j + 25;

end

% Ahora bien, el termino de error del modelo Y_{ig} consider la suma de los
% errores individuales y grupales, por lo que se puede reescribir como
% e_{ig} = epislon_{ig} + v_g
e_ig = matriz(:,3) + matriz(:,2);

% Finalmente entonces, el modelo a estimar considera los diferentes betas
% con los diferentes coeficientes, columnas de la matriz y los errores
y_ig = beta(1) + beta(2) * matriz(:,5) + beta(3) * matriz(:,4) + e_ig;

% A partir de esto, todas las variables tienen dimensiones de (1000x1)

%% 1. SUPUESTOS MRL QUE CUMPLEN LOS DATOS SIMULADOS

% a. Observaciones vienen de una distribucion comun y son independientes 
% (iid) entre grupos, pero no entre todas las observaciones
% b. X e Y satisfacen Y = X'*beta + e y E(e_g|X) = 0 ya que los grupos son
% independientes, no obstante, no podemos afirmar que E(e_ig|X) = 0 dado a
% que hay un termino de error a nivel grupal junto con el hecho de que no
% hay independencia en la variable independiente X1. Por ello, 
% [E(Y|X) = X'*beta) -> Mejor predictor lineal igual a la esperanza a nivel
% grupal
% c. Segundos momentos finitos (varianza acotada) E(Y^2)<inf, E||X||^2<inf
% d. Ausencia de multicolinealidad perfecta al menos a nivel grupal, podria
% haber a nivel individual dependiendo; de todas maneras, si se cumple que
% Q_xx = E(X'X) > 0, no deberia haber multicolinealidad
% e. Homocedasticidad de los errores, si a nivel grupal, no necesariamente
% a nivel individual, ya que hay un termino del error que depende del grupo
% en el pertenece el individuo: E(e^2|X) = sigma^2(X) 
% (varianza depende del X)

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


%% 6. FWL Y MODELO DE EFECTOS FIJOS

%% 7. REPETICION CON DISTINTA DISTRIBUCION DE X1

%% 8. RESUMA LO APRENDIDO

% Está fucking larga la tarea

