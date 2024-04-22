% ----------------------------------------------------------------------- 
%                       TAREA 1 - ECONOMETRÍA I - ME
%        Grupo 3: F. Anguita, N. Bastias, C. Cid, T Munoz O. & R. Salazar
% ----------------------------------------------------------------------- 

%% Directorios por usuario
clc;clear;

%% 0. SIMULACIÓN DE BASE DE DATOS

% Especificación: Y_ig = b_0 + b_1*X_1ig + b_2*X_2ig + c_ig

% Establecemos parámetros
b = [1, 2, 4];                          % Vector beta
n = 1000;                               % Número de personas
g = 40;                                 % Número de grupos
n_g = n/g;                              % Cantidad de personas por grupo
rng(14);                                % Semilla

% Generamos residuo ig y x2
e_ig = normrnd(0,1,[n, 1]);             %                       (1000x1)           
x_2ig = normrnd(5,1,[n, 1]);            %                       (1000x1)

% Inicializamos matrices
x_1 = zeros(n_g, g);                    % X1                      (25x40)  
y_ig = zeros(n, 1);                     % Y                       (1000x1)
v = zeros(n_g, 1);                      % v                       (40x1)

% Iteramos para cada grupo el generar su residuo y valores de x1
    % Residuos por grupo quedan en vector v(40x1) y x1 por grupo quedan en matriz x_1(25x40)
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

% Los supuestos de MRL son:
    % a. Observaciones vienen de una distribucion comun y son independientes (iid)
    % b. X e Y satisfacen Y = X'*beta + e y E(e|X)=0 [E(Y|X) = X'*beta) -> Mejor predictor lineal igual a la esperanza
    % c. Segundos momentos finitos (varianza acotada) E(Y^2)<inf, E||X||^2<inf
    % d. Ausencia de multicolinealidad peftecta -> Q_xx = E(X'X)>0
    % e. Homocedasticidad de los errores -> E(e^2|X) = sigma^2 (varianza cte)

%% 2. COEFICIENTES MCO 

% Especificación: y_ig = b_0 + b_1*x_1ig + b_2*x_2ig + c_ig

% Estimamos coeficientes
x_ig = [ones(n, 1), x_1ig, x_2ig];
b_mco = inv(x_ig'*x_ig)*(x_ig'*y_ig);


%% 3. ERRORES ESTANDAR (p.136 - 140 / 150 Hansen)

% Homocedasticidad y ausencia de correlacion
r = y_ig - x_ig*b_mco;                  % Residuos MCO  
s2 = (1./(n-3))*(r'*r);                 % Estimador varianza del error
mvc = s2*inv(x_ig'*x_ig);               % Matriz varianza covarianza
ee = sqrt(diag(mvc));                   % Errores estándar 

% Robustos (heterocedasticidad) 
r = y_ig - x_ig*b_mco;                 % Residuos MCO 
d = diag(r.^2);                        % d = matriz diagonal de r^2 
mvc_w = inv(x_ig'*x_ig)*(x_ig'*d*x_ig)*inv(x_ig'*x_ig);      % Matriz de White (var-cov robusta)
ee_r = sqrt(diag(mvc_w));              % Errores estándar robustos

% Agrupados (clausterizados: indepencia entre grupos no al interior de ellos)
grupo = floor((0:size(x_ig, 1)-1) / 25) + 1;     % Indicador de grupo
grupo = reshape(grupo,[],1);

% Usar el de Nicolas


%% 4. TEST DE HIPOTESIS NULA

% H0: b_1 = 1 -> H1: b_1 <> 1
% Test t para los distintos tipos de errores calculados antes 
    % 1000 obs y 3 regresores: 997 grados de libertad

% Errores homocedásticos
tt = (b_mco(2)-1)/ee(2);
pv = 2 * (1 - tcdf(abs(tt), 997));

% Errores robustos
tt_r = (b_mco(2)-1)/ee_r(2);
pv_r = 2 * (1 - tcdf(abs(tt_r), 997)); 

% Errores clausterizados
tt_c = (b_mco(2)-1)/ee_c(2);
pv_c = 2 * (1 - tcdf(abs(tt_c), 997));


%% 5. MODELO CON EFECTOS FIJOS (p.635 hansen)

% Especificación con efectos fijos: y_ig = b_0 + b_1*x_1ig + b_2*x_2ig + v_g + e_ig
% Estimador efectos fijos: b1 = (X'MX)^(-1)(X'MY) [regresión particionada]

% Calculamos matriz de aniquilación (estas hay que recalcularlas por
% grupo)
P_1 = x_1ig*inv(x_1ig'*x_1ig)*x_1ig';
P_2 = x_2ig*inv(x_2ig'*x_2ig)*x_2ig';

I = eye(n,n);
M_1 = I - P_1;
M_2 = I - P_2;

% Estimamos coeficientes
b0_fe = 
b1_fe = inv(x_1ig'*M_2*x_1ig)*x_1ig'*M_2*y_ig; 
b2_fe = inv(x_2ig'*M_1*x_2ig)*x_2ig'*M_1*y_ig;

b_fe = [b0_fe b1_fe b2_fe];

% Calculamos errores estándar y test t

% Homocedásticos 
r = y_ig - x_ig*b_fe;                   
s2 = (1./(n-3))*(r'*r);                 
mvc = s2*inv(x_ig'*x_ig);               
ee = sqrt(diag(mvc));                   

tt = (b_fe(2)-1)/ee(2);
pv = 2 * (1 - tcdf(abs(tt_1), 997));

% Robustos
r = y_ig - x_ig*b_fe;                 
d = diag(r.^2);                         
mvc_w = inv(x_ig'*x_ig)*(x_ig'*d*x_ig)*inv(x_ig'*x_ig);     
ee_r = sqrt(diag(mvc_w));

tt_r = (b_fe(2)-1)/ee_r(2);
pv_r = 2 * (1 - tcdf(abs(tt_r), 997)); 

% Agrupados

% código errores agrupados

tt_c = (b_fe(2)-1)/ee_c(2);
pv_c = 2 * (1 - tcdf(abs(tt_c), 997));


%% 6. FWL Y MODELO DE EFECTOS FIJOS

% Teorema FWL: b_1 es el estimador de la regresion de (y_1 - y) con (x_1 - x) 

% Obtenemos desviaciones sobre la media de grupo de las variables 
my = accumarray(grupo,y_ig, [], @mean);
mx1 = accumarray(grupo,x_1ig, [], @mean);
mx2 = accumarray(grupo,x_2ig, [], @mean);

indx = 1;
for i = 1:g
    for j = 1:25
        ym(indx) = my(i);
        indx = indx + 1;
    end
end
ym = reshape(ym,[],1);

indx = 1;
for i = 1:g
    for j = 1:25
        x1m(indx) = mx1(i);
        indx = indx + 1;
    end
end
mx1 = reshape(x1m,[],1);

indx = 1;
for i = 1:g
    for j = 1:25
        x2m(indx) = mx2(i);
        indx = indx + 1;
    end
end
mx2 = reshape(x2m,[],1);

y = y_ig - ym;
x1 = x_1ig - mx1;
x2 = x_2ig - mx2;

% Calculamos estimadores
x_fwl = [ones(n,1) x1 x2];
b_fwl = inv(x_fwl'*x_fwl)*(x_fwl'*y);

b0_fwl = 
b1_fwl = inv(x1'*x1)*(x1'*y);
b2_fwl = inv(x2'*x2)*(x2'*y);

% Sgn yo se aplican efectos fijos indirectamente. 
% En el modelo con efectos fijos teníamos b2_fe= inv(x2'*M1*x2)*x2'*M1*y
% Por FWL se cumple M1*X2 = X2 - media(X2) y M1*Y = Y - media(Y)
% Entonces los coeficientes serán 
            % B2 = inv[(X2-media(X2))'*(X2-media(X2))]*[(X2-media(X2))'*(Y-media(Y))]



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
        x_1(z, 1) = normrnd(3,1);  
    else
        x_1(z, 1) = normrnd(5,1);
    end
end

c_ig = e_ig + v_g;

y_ig = b(1) + b(2)*x_1ig + b(3)*x_2ig + c_ig;

% Repetimos desarrollo de preguntas 1. a 5. 

% 1. SUPUESTOS MRL QUE CUMPLEN LOS DATOS SIMULADOS


% 2. COEFICIENTES MCO 

    x_ig = [ones(n, 1), x_1ig, x_2ig];
    b_mco = inv(x_ig'*x_ig)*(x_ig'*y_ig);


% 3. ERRORES ESTANDAR 

    % Homocedásticos
    r = y_ig - x_ig*b_mco;                    
    s2 = (1./(n-3))*(r'*r);                 
    mvc = s2*inv(x_ig'*x_ig);               
    ee = sqrt(diag(mvc));                   

    % Robustos  
    r = y_ig - x_ig*b_mco;                  
    d = diag(r.^2);                        
    mvc_w = inv(x_ig'*x_ig)*(x_ig'*d*x_ig)*inv(x_ig'*x_ig);      
    ee_r = sqrt(diag(mvc_w));              

    % Agrupados 

    grupo = floor((0:size(x_ig, 1)-1) / 25) + 1;    
    grupo = reshape(grupo,[],1);

    % Usar el de Nicolas


% 4. TEST DE HIPOTESIS NULA

    % Errores homocedásticos
    tt = (b_mco(2)-1)/ee(2);
    pv = 2 * (1 - tcdf(abs(tt), 997));

    % Errores robustos
    tt_r = (b_mco(2)-1)/ee_r(2);
    pv_r = 2 * (1 - tcdf(abs(tt_r), 997)); 

    % Errores clausterizados
    tt_c = (b_mco(2)-1)/ee_c(2);
    pv_c = 2 * (1 - tcdf(abs(tt_c), 997));


% 5. MODELO CON EFECTOS FIJOS 

    % Matrices de aniquilación
    P_1 = x_1ig*inv(x_1ig'*x_1ig)*x_1ig';
    P_2 = x_2ig*inv(x_2ig'*x_2ig)*x_2ig';

    I = eye(n,n);
    M_1 = I - P_1;
    M_2 = I - P_2;

    % Coeficientes
    b0_fe = 
    b1_fe = inv(x_1ig'*M_2*x_1ig)*x_1ig'*M_2*y_ig; 
    b2_fe = inv(x_2ig'*M_1*x_2ig)*x_2ig'*M_1*y_ig;

   
    % Errores estándar y test t

    % Homocedásticos 
    r = y_ig - x_ig*b_fe;                   
    s2 = (1./(n-3))*(r'*r);                 
    mvc = s2*inv(x_ig'*x_ig);               
    ee = sqrt(diag(mvc));                   

    tt = (b_fe(2)-1)/ee(2);
    pv = 2 * (1 - tcdf(abs(tt), 997));

    % Robustos
    r = y_ig - x_ig*b_fe;                 
    d = diag(r.^2);                         
    mvc_w = inv(x_ig'*x_ig)*(x_ig'*d*x_ig)*inv(x_ig'*x_ig);     
    ee_r = sqrt(diag(mvc_w));

    tt_r = (b_fe(2)-1)/ee_r(2);
    pv_r = 2 * (1 - tcdf(abs(tt_r), 997)); 

    % Agrupados

    % código errores agrupados

    tt_c = (b_fe(2)-1)/ee_c(2);
    pv_c = 2 * (1 - tcdf(abs(tt_c), 997));


%% 8. RESUMA LO APRENDIDO



