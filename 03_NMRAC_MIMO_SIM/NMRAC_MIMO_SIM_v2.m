%% Tabula Rasa
clear all; clc;

%% Set default for plots
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultAxesXGrid', 'on', 'defaultAxesYGrid', 'on');

ct = clock;

SAVE_PLOTS = 0;
SHOW_FIG = 'On';

BASE_DIR = './results/';
SAVE_DIR = strjoin([BASE_DIR, strjoin(string(ct(1:5)),'-')],'');

if SAVE_PLOTS==1
    if ~exist(SAVE_DIR, 'dir')
        mkdir(SAVE_DIR)
    end
end

%% Simulation parameters

Ts = 0.001;
Tsim = 5000;

% Define the time vector
t_vec = 0:Ts:Tsim;
num_steps = length(t_vec);

TOLLERANCE = eps;

n = 2;
m = 2;

%% Init reference Signal
input_signal = 2;

r = zeros(2, num_steps);
r_dot = zeros(2, num_steps);

if input_signal == 1
    % Constant input signal
    r(1,:) = 0;
    r(2,:) = 0;

    r_dot(1,:) = 0;
    r_dot(2,:) = 0;

elseif input_signal == 2
    % Sine input signal
    r(1,:) = 0.5*sin(t_vec);
    r(2,:) = 0.5*cos(t_vec) ;

    r_dot(1,:) = 0.5*cos(t_vec);
    r_dot(2,:) = -0.5*sin(t_vec);

end


%% Init Model Reference
% Internal dynamics (pure double integrator)
Am = zeros(n,n);
Am(1,2) = 1;
Am(2,2) = 0;

% Am(1,1) = -2;
% Am(1,2) = -1;
% Am(2,1) = 1;
% Am(2,2) = 0;

% Input dynamics
Bm = zeros(n,m);
Bm(2,1) = 1;
Bm(2,2) = -1;

% Initial conditions Model Ref
IC_M = [-pi ; 0];

% Save the values
xm  = zeros(n, num_steps);
xm(:,1) = IC_M;

em  = zeros(n, num_steps);
em(:,1) = r(:,1) - xm(:,1);

um  = zeros(m, num_steps);

Q = 10*eye(n);
R = eye(m);

[L, P, ~]  = lqr(Am, Bm, Q, R);
[E, D] = eig(P);

% Feedforward gain matrix
theta_m = L;

%% Init System
% Internal dynamics
A = zeros(n,n);
A(1,2) = 1;
A(2,2) = 0;
% max_uncertainty = 0.05;
% rng(42);
% Delta_A = max_uncertainty * (rand(n) - 0.5);
% A = A + Delta_A;

% Input dynamics
B = Bm;

% Feedforward gain matrix
theta_init = 0.*theta_m;
% theta_init = theta_m;
theta = zeros(m, n, num_steps);
theta(:,:,1) = theta_init;

% Learning Rate
% Lambda = 25.*[1 0; 0 1]; % constant reference Signal
Lambda = 0.2.*[1 0; 0 1]; % sinusodial reference signal
% Lambda = [1 0; 0 1];

% Initial conditions System
% IC_P = [pi,0];
IC_P = IC_M;

% Save the values
x  = zeros(n, num_steps);
x(:,1) = IC_P;

ex  = zeros(n, num_steps);
ex(:,1) = r(:,1) - x(:,1);

u  = zeros(m, num_steps);

%% Neural network settings for Model ref. and NNC
SAT_VAL = 5;
a = 2/SAT_VAL; % Sigmoid saturates at 2/a.
fprintf("Saturation value: %d\n", SAT_VAL);

% Choose the activation function of the model reference
% 1     Sigmoid (Note: stability proof is non-trivial)
% -1    Linear activation (Note: Use Lyapunov theory to find P)
ACTIVATION_MODEL_REF = -1;
PLOT_ACTIVCATION = 0;

if ACTIVATION_MODEL_REF==-1
    am = -1;
    fprintf("Model Reference: Linear activation function\n");
    Am = (Am-Bm*theta_m);
else
    am = a;
    fprintf("Model Reference sigmoid saturation value: %d\n", SAT_VAL);
end

if PLOT_ACTIVCATION==1
    figure(1);
    vector = linspace(-100, 100, 201);
    plot(vector, sigmoid(vector, am))
    xlabel("$\tilde x$", Interpreter="latex")
    ylabel("$\sigma(\tilde{x})$", Interpreter="latex")
    title("Activation Function Model Reference")

    figure(2);
    vector = linspace(-100, 100, 201);
    plot(vector, sigmoid(vector, a))
    xlabel("$\tilde x$", Interpreter="latex")
    ylabel("$\sigma(\tilde{x})$", Interpreter="latex")
    title("Activation Function NNC")
end

%% Init Lyapunov Function
% Init Error and Alpha
% Error in internal states
e = zeros(n, num_steps);
e(:,1) = xm(:,1) - x(:,1);

% Error in Dynamics
alpha = zeros(n, num_steps);
alpha_dot = zeros(n, num_steps-1);

% Lyapunov function
V = zeros(1 , num_steps);
V_term1 = zeros(1 , num_steps);
V_term2 = zeros(1 , num_steps);

% Timer derivative Lyapunov function
dV = zeros(1 , num_steps-1);
dV_term1 = zeros(1 , num_steps-1);
dV_term2 = zeros(1 , num_steps-1);
dV_term3 = zeros(1 , num_steps-1);

dV2 = zeros(1 , num_steps-1);


 %% Simulation

fprintf("------ SIMUATION STARTED ------\n");
for t = 1:num_steps-1

    % System step
    ex(:,t) = r(:,t) - x(:,t);
    u(:,t) = sigmoid(theta(:,:,t)*ex(:,t), a);
    
    x_dot = A*x(:,t) + B*u(:,t);
    x(:,t+1) = x(:,t) + Ts*x_dot;
    
    
    % Model Reference step
    if am==-1
        um(:,t) = theta_m*r(:,t);
    else
        em(:,t) = r(:,t) - xm(:,t);
        um(:,t) = sigmoid(theta_m*em(:,t), am);
    end

    xm_dot =  Am*xm(:,t) + Bm*um(:,t);
    xm(:,t+1) = xm(:,t) + Ts*xm_dot;


    % Error in internal states
    e(:,t) = xm(:,t) - x(:,t);
    e_dot = xm_dot - x_dot;
    
    % Parameter Update
    theta_dot = compute_parameter_update( ...
        B, theta(:,:,t), ex(:,t), a, ...
        Am, A, x_dot, ...
        Bm, theta_m, am, r_dot(:,t), ...
        Lambda, P, e(:,t) ...
        );
    theta(:,:,t+1) = theta(:,:,t) + Ts.*theta_dot;
    
    % Error in dynamics
    alpha(:,t) = compute_alpha( ...
        Am, A, x(:,t), Bm, theta_m, ...
        ex(:,t), am, B, theta(:,:,t), a);
    alpha_dot(:,t) = compute_alpha_dot( ...
        Am, A, x_dot, Bm, theta_m, ex(:,t), ...
        r_dot(:,t), am, B, theta(:,:,t), a, theta_dot);
    
    % Compute Lyapunov function value and its derivative
    [V(:,t), V_term1(:,t), V_term2(:,t)] = compute_lyapunov_value( ...
        e(:,t), P, alpha(:,t), Lambda ...
        );
    [dV(:,t), dV_term1(:,t), dV_term2(:,t), dV_term3(:,t), dV2(:,t)] = compute_lyapunov_derivative_value( ...
        e(:,t), Am, P, ...
        Bm, theta_m, em(:,t), am, ex(:,t), ...
        alpha(:,t), Lambda, alpha_dot(:,t), ...
        e_dot ...
        );
end
fprintf("---- SIMUATION TERMINATED -----\n");

[max_dV, max_dV_index] = max(dV);
[min_V, min_V_index] = min(V);

fprintf("Min of Lyapunov: %d \nMaximum of lyapunov derivative: %d \n", min_V, max_dV);
fprintf("Parameters of the Model refernce: \n");
disp(theta_m);
fprintf("Final Parameters of the NNC: \n");
disp(theta(:,:,end));


%% Plot response
% x_1 component
fig = figure(2);
set(fig, 'Visible', SHOW_FIG);
subplot(2,1,1);
plot(t_vec, r(1,:), ...
     t_vec, x(1,:), ...
     t_vec, xm(1,:));
legend("r", "x", "x_m");
xlabel("Time [s]");
ylabel("x_1");
title("Respnose x_1 Component");

% x_2 component
subplot(2,1,2);
plot(t_vec, r(2,:),...
     t_vec, x(2,:),...
     t_vec, xm(2,:));
legend("r", "x", "x_m");
ylabel("x_2");
xlabel("Time [s]");
title("Respnose x_2 Component");
sgtitle("Response");

if SAVE_PLOTS == 1
    full_path = char(strcat(SAVE_DIR, '/', 'NMRAC_MIMO_Response.png'));
    saveas(gcf, full_path);
end

%% Plot theta
fig = figure(3);
set(fig, 'Visible', SHOW_FIG);
subplot(2,2,1);
plot(t_vec, squeeze(theta(1,1,:)), ...
    t_vec, theta_m(1,1)*ones(1, num_steps));
xlabel("Time [s]");
ylabel("Magnitude");
legend("\theta_{1,1}", "\theta_{m; 1,1}");
title("\theta_{1,1}")

subplot(2,2,2);
plot(t_vec, squeeze(theta(1,2,:)), ...
    t_vec, theta_m(1,2)*ones(1, num_steps));
xlabel("Time [s]");
ylabel("Magnitude");
legend("\theta_{1,2}", "\theta_{m; 1,2}");
title("\theta_{1,2}")

subplot(2,2,3);
plot(t_vec, squeeze(theta(2,1,:)), ...
    t_vec, theta_m(2,1)*ones(1, num_steps) ...
    );
xlabel("Time [s]");
ylabel("Magnitude");
legend("\theta_{2,1}", "\theta_{m; 2,1}");
title("\theta_{2,1}")

subplot(2,2,4);
plot(t_vec, squeeze(theta(2,2,:)), ...
    t_vec, theta_m(2,2)*ones(1, num_steps) ...
    );
xlabel("Time [s]");
ylabel("Magnitude");
legend("\theta_{2,2}", "\theta_{m; 2,2}");
title("\theta_{2,2}")

sgtitle("Parameters over time");

if SAVE_PLOTS == 1
    full_path = char(strcat(SAVE_DIR, '/', 'NMRAC_MIMO_Parameters.png'));
    saveas(gcf, full_path);
end


%% Plot Lyapunov Function
fig = figure(4);
set(fig, 'Visible', SHOW_FIG);
subplot(2,1,1);
plot(t_vec, V, ...
    t_vec, V_term1, ...
    t_vec, V_term2);
legend("V", "V_e", "V_{\alpha}");
xlabel("Time [s]");
ylabel("V(e,\alpha)");
title("Lyapunov Function");

subplot(2,1,2);
plot(t_vec(:,2:num_steps), dV,  ...
    t_vec(:,2:num_steps), dV_term1, ...
    t_vec(:,2:num_steps), dV_term2, ...
    t_vec(:,2:num_steps), dV_term3);

legend("dV", "dV_{Q}", "dV_{\gamma}",  "dV_{\alpha}");
xlabel("Time [s]");
ylabel("Magnitude");
title("Lyapunov Function Time Derivative");

sgtitle("Lyapunov Function");

if SAVE_PLOTS == 1
    full_path = char(strcat(SAVE_DIR, '/', 'NMRAC_MIMO_Lyapunov.png'));
    saveas(gcf, full_path);
end


%% Plot errors
fig = figure(5);
set(fig, 'Visible', SHOW_FIG);
subplot(2,1,1);
plot(t_vec, e(1,:), ...
    t_vec, ex(1,:), ...
    t_vec, em(1,:));
xlabel("Time [s]");
ylabel("Magnitude");
legend("e", "e_x", "e_m");
title("Errors x_1 component");

subplot(2,1,2);
plot(t_vec, e(2,:), ...
    t_vec, ex(2,:), ...
    t_vec, em(2,:));
legend("e", "e_x", "e_m");
xlabel("Time [s]");
ylabel("Magnitude");
title("Errors x_2 component");

sgtitle("Errors in internal states");

if SAVE_PLOTS == 1
    full_path = char(strcat(SAVE_DIR, '/', 'NMRAC_MIMO_Error.png'));
    saveas(gcf, full_path);
end


%% Plot the difference in dynamics
fig = figure(6);
set(fig, 'Visible', SHOW_FIG);
subplot(2,1,1);
plot(t_vec, alpha(1,:), ...
    t_vec, alpha(2,:));
xlabel("Time [s]");
ylabel("Magnitude");
legend("\alpha_1", "\alpha_2");
title("\alpha");

subplot(2,1,2);
plot(t_vec(:,2:num_steps), alpha_dot(1,:), ...
    t_vec(:,2:num_steps), alpha_dot(2,:));
legend("d\alpha_1", "d\alpha_2");
xlabel("Time [s]");
ylabel("Magnitude");
title("d\alpha");

sgtitle("Error in dynamics over time (\alpha)");

if SAVE_PLOTS == 1
    full_path = char(strcat(SAVE_DIR, '/', 'NMRAC_MIMO_Alpha.png'));
    saveas(gcf, full_path);
end



%% Helper functions

function theta_dot = compute_parameter_update( ...
    B, theta, ex, a, ...
    Am, A, x_dot, ...
    Bm, theta_m, am, r_dot, ...
    Lambda, P, e)
    % This function computes the weight update for the proposed MIMO NMRAC
    % of the Master thesis of Dalim Wahby
    
    if norm(ex)>eps
        ex_dot = r_dot - x_dot;
    
        Sigma = pinv(B*dsigmoid(theta*ex, a));
    
        xsi = Lambda*P*e ...
            + (Am-A)*x_dot ...
            + Bm*dsigmoid(theta_m*ex, am)*theta_m*ex_dot...
            - B*dsigmoid(theta*ex, a)*theta*ex_dot;
        
        %theta_dot = Sigma*xsi*pinv(ex);
        theta_dot = (ex'\(Sigma*xsi)')';
    else
        theta_dot = zeros(2,2);
    end
end

function [dV, dV_term1, dV_term2, dV_term3, dV2] = compute_lyapunov_derivative_value( ...
    e, Am, P, ...
    Bm, theta_m, em, am, ex, ...
    alpha, Lambda, alpha_dot, ...
    e_dot)
    
    dV_term1 = 2.*e'*P*Am*e;
    gamma = Bm*(sigmoid(theta_m*em, am) - sigmoid(theta_m*ex, am));
    dV_term2 = 2.*e'*P*gamma;
    dV_term3 = 2.*alpha'*(P*e+inv(Lambda)*alpha_dot);

    dV = dV_term1 + dV_term2 +  dV_term3;
    
    dV2_term1 = e'*P*e_dot + e_dot'*P*e;
    dV2_term2 = alpha'*inv(Lambda)*alpha_dot + alpha_dot'*inv(Lambda)*alpha;
    
    dV2 = dV2_term1 + dV2_term2;
    
end

function [V, V_term1, V_term2] = compute_lyapunov_value(e, P, alpha, Lambda)
    % Error in internal states
    V_term1 = e'*P*e;
    % Error in internal Dynamics
    V_term2 = alpha'*inv(Lambda)*alpha;
    
    V = V_term1 + V_term2;
end

function alpha = compute_alpha(Am, A, x, Bm, theta_m, ex, am, B, theta, a)
    % This term defines the difference in Dynamics between the model
    % reference and the closed-loop system. If this term is equal to 0,
    % then we know that the matching conditions on the dynamics are
    % satisfied.
    alpha = (Am-A)*x ...
            + Bm*sigmoid(theta_m*ex, am)...
            - B*sigmoid(theta*ex, a);
end

function alpha_dot = compute_alpha_dot(Am, A, x_dot, Bm, theta_m, ex, r_dot, am, B, theta, a, theta_dot)
    % This term defines the difference in Dynamics between the model
    % reference and the closed-loop system
    ex_dot = r_dot - x_dot;
    alpha_dot = (Am-A)*x_dot ...
            + Bm*dsigmoid(theta_m*ex, am)*theta_m*ex_dot ...
            - B*dsigmoid(theta*ex, a)*(theta_dot*ex + theta*ex_dot);
end


function u = sigmoid(x, a)
    % This function defines a smooth saturation, modeled by a sigmoid
    % function that saturates at the value 2/a.
    % If a = -1, then the activation function is linear and the matrix x is
    % returned
    if a==-1
        u = x;
    else
        u = 2.*(1-exp(-a.*x))./(a.*(1+exp(-a.*x)));
    end

end

function du = dsigmoid(x, a)
    % This function returns the derivative of the sigmoid function with
    % respect to x, evaluated at x.
    if a==-1
        du = eye(size(x,1));
        % TODO figure out the dimensions of du
    else
        du = diag(4.*exp(-a.*x)./(1+exp(-a.*x)).^2);
    end
end