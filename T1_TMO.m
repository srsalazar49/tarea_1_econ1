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

rng(10) % fijando la semilla
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

% Calculamos ahora los beta con la funcion de MCO que definimos
% previamente:
[beta_gorro] = MCO(Y,X); %calculamos los betas con una funcion que definimos

% Redondeamos los betas para que tengan hasta 3 decimales
beta_gorro = round(beta_gorro, 2);

% Exportamos los resultados en una tabla
tabla_P2 = table(['\beta_0';'\beta_1';'\beta_2'],... 
    [beta_gorro(1);beta_gorro(2); beta_gorro(3)]);
writetable(tabla_P2,'tabla_P2.txt','Delimiter',' ')  
type 'tabla_P2.txt'

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

