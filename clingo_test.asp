


%% argument(number,claim,assumption(X)) for all assumptions X in support of arugument. 
%% argument(1,a,assumption(a)) %% assumption argument
%% argument(2,b,assumption(b)) %% assumption argument
%% argument(3,f,assumption(a)) rule f :- a,b
%% argument(3,f,assumption(b))



%%%% each assumption induces an argument
argument(I+1,X,assumption(X)) :- assumption(X), I= #count{ Y : assumption(Y), Y < X}.

argument(2,a,assumption(b)).
argument(2,a,assumption(c)).

argument(10,a,assumption(c)).

%%% each rule induces an argumet iff all its body literals are claims of arguments. 
argument(I,X,assumption(Y)) :- head(R,X),   
    argument(_,Z,_) : body(R,Z);    
    assumption(Y),               
    %% choose_argument(M,I,R,U), argument(M,U,assumption(Y)),          
    I = (J+1)*K*L,                             
    J = 0, %% #count{ V : assumption(V), V < X},
    K = #sum{ N : choose_argument(_,N,R,Z), body(R,Z) },
    L = 1. 

%% (J+1)*(K+1), K = #count{ Y : assumption(Y), Y < X}.

#show argument/3.


% for toprule R, body element X, assign argument I  a number J

choose_argument(J,argument(I,X,assumption(Y)),R) :- argument(I,X,assumption(Y)), body(R,X), 
    J = #count{ M : argument(M,X,_) , M < I }.


#show choose_argument/3.

choose_argument_up_to(J,argument(I,X,assumption(Y)),R,U) :- body(R,X), argument(I,X,assumption(Y)), X <= U,
    J = #count{ M : argument(M,X,_) , M < I }.
#show choose_argument_up_to/4.


assumption(a).
% assumption(b).
% assumption(c).
% head(1,e).
% body(1,a).
% body(1,b).

% head(2,u).
% body(2,b).
% body(2,a).

head(3,f).
body(3,a).

% head(4,f).
% body(4,b).

% head(5,p).
% body(5,e).