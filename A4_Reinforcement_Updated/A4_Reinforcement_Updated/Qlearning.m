%% Initialization
%  Initialize the world, Q-table, and hyperparameters

world = 1;
s = gwinit(world);
Q = rand(s.ysize,s.xsize, 4);
episodes = 2000;
epislon = 1;
learning_rate  = 0.2;
discount = 0.95; 

%limit up
Q(1,:,2) = -inf;

%limit down
Q(size(Q, 1),:,1) = -inf;

%limit left
Q(:,1,4) = -inf;

%limit right
Q(:,size(Q, 2),3) = -inf;

%% Training loop
%  Train the agent using the Q-learning algorithm.

for i=1:episodes
  
    while ~s.isterminal
        
        y = s.pos(1);
        x = s.pos(2);
        a = chooseaction(Q, y, x, [1,2,3,4], [1,1,1,1], epislon);
        s = gwaction(a);
        reward = s.feedback;
        y1 = s.pos(1);
        x1 = s.pos(2);
        V = getvalue(Q);
        Q(y,x,a) = (1-learning_rate)*Q(y,x,a)+learning_rate*(reward+discount*V(y1, x1));
        
    end
    epislon = epislon - 1/(episodes);
    s = gwinit(world);
end

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.

figure(1)
V = getvalue(Q);
imagesc(V)

figure(2)
P = getpolicy(Q);
gwinit(world)
gwdraw()
gwdrawpolicy(P)

while ~s.isterminal

        y = s.pos(1);
        x = s.pos(2);
        a = P(y, x);
        s = gwaction(a);
        gwdraw("Policy",P)
        
end 
