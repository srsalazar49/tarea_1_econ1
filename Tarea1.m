%--------- TAREA 1 --------%
%------- ECONOMETRÍA 1 -----%
%----- Ultima actualización: 21/04/2024 16:19------%

clear, clc;
cd('C:\Users\HP\Documents\MAESTRÍA\1. Primer Semestre\Econometría I\Tareas\Tarea 1')

%%Caracterización para simulación
beta=[1; 2; 4] %3x1
N=1000 %Total de personas
G=40 %Total de grupos
Ng=N/G %Tamaño de grupos

%%Simulación de la base
rng(7)

%Inicializamos matrices donde almacenaremos los datos
y=zeros(N,1); %Dimensión 1000x1
x1=zeros(N,1); %Dimensión 1000x1
x2=normrnd(5,1,N,1); %Dimensión 1000x1
i1=ones(N,1);

%Generamos los términos de error del modelo
epsilon_ig=normrnd(0,1,N,1)  %Dimensión 1000x1
v=normrnd(0,1,G,1)%Errores por grupo Dimensión 40x1
v_g=repelem(v,N/G) %Dimensión 1000x1, errores repetidos por grupo en vector total

%Llenado de matrices por grupos
for g= 1:G
    %Determinamos si v>0 o v<0
    if v_g<0
        x1((g-1)*Ng+1:g*Ng)=normrnd(3,1,Ng, 1);
    else
        x1((g-1)*Ng+1:g*Ng)=normrnd(5,1,Ng,1);
    end
    %Calcular el modelo establecido apilando los resultados por grupo en el
    %vector 1000x1 según corresponda
    y((g-1)*Ng+1:g*Ng)=beta(1)+beta(2).*x1((g-1).*Ng+1:g*Ng)+beta(3).*x2((g-1)*Ng+1:g*Ng)+epsilon_ig((g-1)*Ng+1:g*Ng)+v_g(g)
end

%%Pregunta 2: Mínimos Cuadrados Ordinarios
%Agrupamos los regresores (intercepto incluido) en esta matriz
X=[i1 x1 x2]; %Dimensión 1000x3
[b,ee,t_stat]=OLS(X, y);

%%Errores estándar: 
%Asumiendo homocedasticidad y ausencia de correlación
e_hat=y-X*b %Defino errores
K=size(X,K) %Número de regresores
s_sqr=1/(N-K)*(e_hat'*e_hat)
var_cov=s_sqr*inv(X'*X) %Matriz varianza-covarianza
se=sqrt(diag(var_cov)) %Errores estandar
