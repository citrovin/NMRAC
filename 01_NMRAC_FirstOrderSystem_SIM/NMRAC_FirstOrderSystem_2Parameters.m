%% TODO: Description of script

%% Tabula Rasa
clear all; clc;

%% Set default for plots
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultAxesXGrid', 'on', 'defaultAxesYGrid', 'on');

%% Simulation parameters

Ts = 0.01;
Tsim = 20;

% Define the time vector
t_vec = 0:Ts:Tsim;
num_steps = length(t_vec);

TOLLERANCE = eps;

SAT_VAL = 5;
a = 2/SAT_VAL; % Sigmoid saturates at 2/a.
fprintf("Saturation value: %d\n", SAT_VAL);

%% Init reference Signal
input_signal = 2;

r = zeros(1, num_steps);
r_dot = zeros(1, num_steps);

if input_signal == 1
    % Constant input signal
    r(1,:) = 1;
    
    r_dot(1,:) = 0;

elseif input_signal == 2
    % Sine input signal
    r(1,:) = 0.4*sin(t_vec) + 1;

    r_dot(1,:) = 0.4*cos(t_vec);

end

%% Reference Model
% Dynamics of the closed loop system
% xm_dot = -Am*xm + Bm*theta_m*xm
% xm_dot = (-Am+Bm*theta_m)*xm
% Hence, we know that xm = 0 constitutes an equilibrium point in continuous
% time.

Am = 10;
Bm = 10;

theta_m = 4;

IC_P_m = -1;
xm = zeros(1, num_steps);
xm(1,1) = IC_P_m;

em = zeros(1, num_steps);

am = a;


%% System
% Dymanics: x_dot = -Am*x + B*u
A = Am;
B = Bm;

% Theta now consists of two parameters theta = [theta_1, theta_2]
% u = theta_2*(theta_1*ex)
theta = ones(2, num_steps);
theta(1,1) = 4;

% Learning rate
c = 0.05;

IC_P = IC_P_m;
x = zeros(1, num_steps);
x(1,1) = IC_P;

ex = zeros(1, num_steps);

%% Parameters for the update mechanism
% Difference in internal states
e = zeros(1, num_steps);

% Difference in dynamics
alpha = zeros(1,num_steps);

%% Save values
V = zeros(1,num_steps);
V_term1 = zeros(1,num_steps);
V_term2 = zeros(1,num_steps);

dV = zeros(1,num_steps-1);
dV_term1 = zeros(1,num_steps-1);
dV_term2 = zeros(1,num_steps-1);
dV_term3 = zeros(1,num_steps-1);


 %% Simulation

fprintf("------ SIMUATION STARTED ------\n");
for t = 1:(num_steps-1)
    % Step System
    ex(1,t) = r(1,t) - x(1,t);
    ex_dot = r_dot(1,t) - x(1,t);
    
    % TODO: Don't split the values, but use matrix operations
    theta_1 = theta(1,t);
    theta_2 = theta(2,t);

    u = theta_2*sigmoid(theta_1*ex(1,t), a);
    x_dot = -A*x(1,t) + B*u;
    
    x(1,t+1) = x(1,t) + Ts*x_dot;

    % Step Model Ref
    em(1,t) = r(1,t) - xm(1,t);
    
    um = sigmoid(theta_m*em(1,t), a);
    xm_dot = -Am*xm(1,t) + Bm*um;
    
    xm(1,t+1) = xm(1,t) + Ts*xm_dot;
    
    % Compute error for update
    e(1,t) = xm(1,t) - x(1,t);
    e_dot = xm_dot - x_dot;
    
    alpha(:,t) = compute_alpha( ...
                    A, Am, x(1,t), ...
                    Bm, theta_m, ex(1,t), am, ...
                    B, theta_2, theta_1, a);

    % Update params
    theta_dot = update_params( ...
                    B, theta_1, ex(:,t), a, theta_2, ...
                    e(:,t), c, ...
                    A, Am, x_dot, ...
                    Bm, theta_m, am, ex_dot)
    theta(:,t+1) = theta(:,t) + Ts*theta_dot;
    
    % Compute Lyapunov Function
    [V(:,t), V_term1(:,t), V_term2(:,t)] = compute_lyapunov_value(e(:,t), alpha(:,t));
    
    theta_dot_1 = theta_dot(1);
    theta_dot_2 = theta_dot(2);
    

    [dV(:,t), dV_term1(:,t), dV_term2(:,t), dV_term3(:,t)] = compute_lyapunov_derivative( ...
    c, A, Am, x_dot, ...
    Bm, theta_m, ex(1,t), am, ex_dot, ...
    B, theta_1, a, theta_dot_2, ...
    theta_2, theta_dot_1, ...
    e(1,t), em(1,t));
    
end

% Compute last values for all needed values
ex(1,end) = r(1,end) - x(1,end);
em(1,end) = r(1,end) - xm(1,end);

%% Plots
% Plot the system response
figure(1);

subplot(2,1,1);
plot(t_vec, r, ...
    t_vec, xm, ...
    t_vec, x);
legend("r", "x_m", "x");
xlabel("Time [s]");
ylabel("f(x)");
title("System Response");


subplot(2,1,2);
plot(t_vec, theta_m*ones(1,num_steps), ...
    t_vec, theta(1,:), ...
    t_vec, theta(2,:));
legend("\theta_m", "\theta_1", "\theta_2");
xlabel("Time [s]");
ylabel("\theta");
title("Gains over time");

% Plot the Lyapunov function
figure(2);
subplot(2,1,1);
plot(t_vec, V, ...
    t_vec, V_term1, ...
    t_vec, V_term2);
legend("V","V_e", "V_{\alpha}");
xlabel("Time [s]");
ylabel("V(e,\alpha)");
title("Lyapunov function");

subplot(2,1,2);
plot(t_vec(2:num_steps), dV, ...
    t_vec(2:num_steps), dV_term1, ...
    t_vec(2:num_steps), dV_term2, ...
    t_vec(2:num_steps), dV_term3);
legend("dV","dV_e", "dV_{\gamma}","dV_{\alpha}");
xlabel("Time [s]");
ylabel("V(e,\alpha)");
title("Lyapunov function");


%% Function
function alpha = compute_alpha( ...
    A,Am,x, ...
    Bm, theta_m, ex, am, ...
    B, theta_2, theta_1, a)
    alpha = (A-Am)*x ...
    + Bm*sigmoid(theta_m*ex, am) ...
    - B*theta_2*sigmoid(theta_1*ex, a);
end

function [V, V_term1, V_term2] = compute_lyapunov_value(e, alpha)
    V_term1 = e^2;
    V_term2 = alpha^2;

    V = V_term1 + V_term2;
end

function [dV, dV_term1, dV_term2, dV_term3] = compute_lyapunov_derivative( ...
    c, A, Am, x_dot, ...
    Bm, theta_m, ex, am, ex_dot, ...
    B, theta_1, a, theta_dot_2, ...
    theta_2, theta_dot_1, ...
    e, em)
    
    alpha_dot = (1/c)*((A-Am)*x_dot ...
                       + Bm*dsigmoid(theta_m*ex, am)*theta_m*ex_dot ...
                       - B*sigmoid(theta_1*ex, a)*theta_dot_2 ...
                       - B*sigmoid(theta_1*ex, a)*theta_2*ex*theta_dot_1 ...
                       - B*sigmoid(theta_1*ex, a)*theta_2*ex*ex_dot);

    dV_term1 = -2*Am*e^2;
    dV_term2 = 2*e*Bm*(sigmoid(theta_m*em, am) - sigmoid(theta_m*ex, am));
    dV_term3 = (e/c + alpha_dot);
    
    dV = dV_term1 + dV_term2 + dV_term3;
end


function theta_dot = update_params( ...
    B, theta_1, ex, a, theta_2, ...
    e, c, ...
    A, Am, x_dot, ...
    Bm, theta_m, am, ex_dot)

    A_bar = [-B*dsigmoid(theta_1*ex, a)*theta_2*ex; 
        -B*sigmoid(theta_1*ex, a)]';
    b_bar = e/c ...
        + (A-Am)*x_dot ...
        + Bm*dsigmoid(theta_m*ex, am)*theta_m*ex_dot ...
        - B*dsigmoid(theta_1*ex, a)*theta_2*ex*ex_dot;

    theta_dot = A_bar\b_bar;
end

function u = sigmoid(x, a)
    % This function defines a smooth saturation, modeled by a sigmoid
    % function that saturates at the value 2/a.
        u = 2.*(1-exp(-a.*x))./(a.*(1+exp(-a.*x)));
end

function du = dsigmoid(x, a)
    % This function returns the derivative of the sigmoid function with
    % respect to x, evaluated at x.
    du = diag(4.*exp(-a.*x)./(1+exp(-a.*x)).^2);
end