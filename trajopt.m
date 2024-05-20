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

% parameters of the cost matrices
[nx, nu] = size(B); 
Q = diag([ones(nd, 1); zeros(nd, 1)]);
R = 0*eye(nu);
tau = 20; % length of the trajectory

% parameters of constraints
max_vel = 10; % max speed
max_acc = 10; % max acceleration magnitude

% gam = 5e-1; % scaling factor for the reference trajectory
% xhat = kron(ones(1, tau), 3*ones(nx, 1));
p0 = linspace(0, 10, tau);
xhat = [p0; 4*sin(p0); zeros(4, tau)];

figure('Position',[0,0,800,800])
plot(xhat(1, :), xhat(2, :), 'r', 'LineWidth', 2)

yalmip('clear')

% variables for trajectory optimization
x = sdpvar(nx, tau, 'full'); % variables for the state trajectory
u = sdpvar(nu, tau, 'full'); % variables for the input trajectory

% dynamics constraints for trajectory optimization

constr = [ x(:, 1) == zeros(nx, 1) ]; % initial state constraint/ boundary condition 

for t = 1:tau-1
    constr = [constr, x(:, t+1) == A*x(:, t) + B*u(:, t)]; % dynamics constraints
end

% x2vel = [zeros(3, 3), eye(nd)]; % matrix that map state to velocity
% 
% for t = 1:tau
%     constr = [constr, norm(x2vel*x(:, t), 2) <= max_vel]; % max speed
%     constr = [constr, norm(u(:, t), 2) <= max_acc]; % max acceleration
% end

obj = 0;

for t = 1:tau
    obj = obj + 0.5*(x(:, t)-xhat(:, t))'*Q*(x(:, t)-xhat(:, t))...
          +0.5*u(:, t)'*R*u(:, t);
end

options = sdpsettings('verbose',1,'solver','mosek');

solution = optimize(constr, obj, options);

xopt = value(x); % optimal state trajectory
uopt = value(u); % optimal input trajectory

% figure('Position',[0,0,800,800])
% plot3(xopt(1, :), xopt(2, :), xopt(3, :), 'k', 'LineWidth', 2)
% hold on
% plot3(xhat(1, :), xhat(2, :), xhat(3, :), 'r', 'LineWidth', 2)
% hold off

figure('Position',[0,0,800,800])
plot(xopt(1, :), xopt(2, :), 'k', 'LineWidth', 2)
hold on
plot(xhat(1, :), xhat(2, :), 'r--', 'LineWidth', 2)
hold off
