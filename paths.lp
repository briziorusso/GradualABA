% a---b---c---d
%     \-z-/
edge(edge(0),a,b).
edge(edge(1),a,b).
edge(edge(2),b,c).
edge(edge(3),b,z).
edge(edge(4),z,c).
edge(edge(5),z,c).
edge(edge(6),c,d).
start(b).
end(c).

node(X;Y) :- edge(E,X,Y).

explore(X,X,(X,nil),X)   :- start(X), end(X).
explore(X,Y,(Y,nil),Z)   :- start(X), end(Y), node(Z), Y != Z.
explore(X,Y,(X',P),Z)    :- explore(X,Y,(X',P)), explore(X,Y,P,Z), X != X', X' != Z.
explore(X,Y,(X',(Y',P))) :- explore(X,Y,(Y',P),X'), edge(E,X',Y').

segment(X,Y,(X,P))       :- explore(X,Y,(X,P)).
segment(X,Y,(X',(Y',P))) :- segment(X,Y,(Z,(X',(Y',P)))).

extract(X',Y')   :- segment(X,Y,(X',(Y',P))).
extract(E,X,Y)   :- edge(E,X,Y), extract(X,Y).
extract(E,X,Y,N) :- extract(E,X,Y), N = #count{E' : extract(E',X,Y), E' < E}.

counted(X,Y,N) :- extract(E,X,Y,N-1), not extract(_,X,Y,N).

factor(X,Y,(X',(Y',nil)),N) :- segment(X,Y,(X',(Y',nil))), counted(X',Y',N).
factor(X,Y,(X',(Y',P)),M*N) :- segment(X,Y,(X',(Y',P))), factor(X,Y,(Y',P),M), counted(X',Y',N).

sorted(X,Y,(X,P),I) :- segment(X,Y,(X,P)), I = #count{P' : segment(X,Y,(X,P')), P' < P}.

offset(X,Y,step(0),(X,P),0,0)           :- sorted(X,Y,(X,P),0).
offset(X,Y,step(0),(X,P),I,M+N)         :- sorted(X,Y,(X,P),I), offset(X,Y,step(0),(X,P'),I-1,M), factor(X,Y,(X,P'),N).
offset(X,Y,step(S),(X',(Y',P)),I,O+M*N) :- offset(X,Y,step(S-1),(Z,(X',(Y',P))),I,O), extract(E,Z,X',M), factor(X,Y,(X',(Y',P)),N).

path(O+M,X,Y,S,E)                :- offset(X,Y,S,(X',(Y',nil)),I,O), extract(E,X',Y',M).
path(O+M*N..O+(M+1)*N-1,X,Y,S,E) :- offset(X,Y,S,(X',(Y',P)),I,O), extract(E,X',Y',M), factor(X,Y,(Y',P),N).

#show path/5.
