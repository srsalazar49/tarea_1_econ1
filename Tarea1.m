%--------- TAREA 1 --------%
%------- ECONOMETRÍA 1 -----%

%%Caracterización para simulación
beta=[1; 2; 4] %3x1
N=1000 %Total de personas
G=40 %Total de grupos
Ng=N/G %Tamaño de grupos

%%Simulación de la base
%Inicializamos matrices donde almacenaremos los datos
y=zeros(N,1); %Dimensión 1000x1
x1=zeros(N,1); %Dimensión 1000x1
x2=normrnd(5,1,N,1); %Dimensión 1000x1

%Generamos los términos de error del modelo
epsilon_ig=normrnd(0,1,N,1)  %Dimensión 1000x1
v_g=normrnd(0,1,N,1) %Dimensión 1000x1

%Llenado de matrices por grupos
for g= 1:G
    %Determinamos si v>0 o v<0
    if v_g<0
        x1((g-1)*Ng+1:g*Ng)=normrnd(3,1,Ng, 1);
    else
        x1((g-1)*Ng+1:g*Ng)=normrnd(5,1,Ng,1);
    end
    %Calcular el modelo establecido
    y((g-1)*Ng+1:g*Ng)=beta(1)+beta(2).*x1((g-1).*Ng+1:g*Ng)+beta(3).*x2((g-1)*Ng+1:g*Ng)+epsilon_ig((g-1)*Ng+1:g*Ng)+v_g(g)
end