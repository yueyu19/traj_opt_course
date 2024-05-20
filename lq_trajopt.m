clear; clc;

% define parameters of the continuous double-integrator dynamics
% dx(t)/dt = Ac*x(t)+Bc*u(t) 
% x(t) = [x_pos(t); y_pos(t); z_pos(t); x_vel(t); y_vel(t); z_vel(t)]
% u(t) = [x_acc(t); y_acc(t); z_acc(t)];

nd = 3; % dimension of the double-integrator dynamics
dt = 0.5; % sampling time

% parameters of the continuous-time dynamics
Ac = [zeros(nd, nd), eye(nd); zeros(nd, 2*nd)];
Bc = [zeros(nd, nd); eye(nd)];

% parameters of the discrete-time dynamics
A = expm(dt.*Ac);
B = integral(@(t) expm(Ac*t), 0, dt, 'ArrayValued', true)*Bc;
% discrete time dynamics: x_{t+1}=A * x_t + B * u_t

% parameters of the cost matrices
[nx, nu] = size(B); 
Q = diag([ones(nd, 1); zeros(nd, 1)]);
Qtau = Q;
R = 1*eye(nu);
tau = 20; % length of the trajectory

p0 = linspace(0, 10, tau);
xhat = [p0; 4*sin(p0); zeros(4, tau)];

%% solve lqr via MOSEK
yalmip('clear')

% variables for trajectory optimization
x = sdpvar(nx, tau, 'full'); % variables for the state trajectory
u = sdpvar(nu, tau-1, 'full'); % variables for the input trajectory

% dynamics constraints for trajectory optimization

constr = [ x(:, 1) == xhat(:, 1) ]; % initial state constraint/ boundary condition 

for t = 1:tau-1
    constr = [constr, x(:, t+1) == A*x(:, t) + B*u(:, t)]; % dynamics constraints
end

% objective function

obj = 0.5*(x(:, tau)-xhat(:, tau))'*Qtau*(x(:, tau)-xhat(:, tau));

for t = 1:tau-1
    obj = obj + 0.5*(x(:, t)-xhat(:, t))'*Q*(x(:, t)-xhat(:, t))...
          +0.5*u(:, t)'*R*u(:, t);
end

options = sdpsettings('verbose',1,'solver','mosek');

solution = optimize(constr, obj, options);

xopt = value(x); % optimal state trajectory
uopt = value(u); % optimal input trajectory

figure('Position',[0,0,800,800])
plot3(xopt(1, :), xopt(2, :), xopt(3, :), 'b', 'LineWidth', 2)
hold on
plot3(xhat(1, :), xhat(2, :), xhat(3, :), 'r--', 'LineWidth', 2)
hold off

%% solve lqr via adjoint systems

yalmip('clear')

% variables for trajectory optimization
x = sdpvar(nx, tau, 'full'); % state trajectory
u = sdpvar(nu, tau-1, 'full'); % input trajectory
w = sdpvar(nx, tau, 'full'); % co-state trajectory

constr = [x(:, 1) == xhat(:, 1), ...
         w(:, tau) == Qtau*(x(:, tau) - xhat(:, tau))];

for t = 1:tau-1
    constr = [constr, x(:, t+1) == A*x(:, t) + B*u(:, t), ...
              w(:, t) == A'*w(:, t+1)+Q*(x(:, t)-xhat(:, t)), ...
              R*u(:, t) == -B'*w(:, t+1)];
end

options = sdpsettings('verbose',1,'solver','mosek');

solution = optimize(constr, 0, options);

xadj = value(x); % optimal state trajectory
uadj = value(u); % optimal input trajectory


% compare the results of adjoint systems and the optimal trajectory
figure('Position',[0,0,800,800])
plot3(xopt(1, :), xopt(2, :), xopt(3, :), 'b', 'LineWidth', 2)
hold on
plot3(xadj(1, :), xadj(2, :), xadj(3, :), 'r--', 'LineWidth', 2)
hold off

%% solve lqr via dynamic programming
yalmip('clear')

% allocating memory space for dynamic programming
P = zeros(nx, nx, tau);
q = zeros(nx, tau);
K = zeros(nu, nx, tau);
d = zeros(nu, tau);
xdp = zeros(nx, tau);
udp = zeros(nu, tau-1);

% initilize the parameters of the value function
P(:, :, tau) = Qtau;
q(:, tau) = -Qtau*xhat(:, tau);

% backward induction
for t = tau-1:-1:1
    K(:, :, t) = -pinv(R+B'*P(:, :, t+1)*B)*B'*P(:, :, t+1)*A;
    d(:, t) = -pinv(R+B'*P(:, :, t+1)*B)*B'*q(:, t+1);
    P(:, :, t) = Q + A'*P(:, :, t+1)*(A+B*K(:, :, t));
    q(:, t) = (A+B*K(:, :, t))'*q(:, t+1) - Q*xhat(:, t);
end
% initialize state trajectory
xdp(:, 1) = xhat(:, 1);
% forward induction
for t = 1:tau-1
    xdp(:, t+1) = (A+B*K(:, :, t))*xdp(:, t) + B*d(:, t);
    udp(:, t) = K(:, :, t)*xdp(:, t) + d(:, t);
end

% compare the results of dynamic programming and the optimal trajectory
figure('Position',[0,0,800,800])
plot3(xopt(1, :), xopt(2, :), xopt(3, :), 'b', 'LineWidth', 2)
hold on
plot3(xdp(1, :), xdp(2, :), xdp(3, :), 'r--', 'LineWidth', 2)
hold off