%% MIMO Nonlinear Model Reference Adaptive Control
% This file contains the proof of concept simulations for MIMO NMRAC. It 
% simulates different trajecotries for initial conditions This
% method was developed during an internship at i3S/CNRS under the
% supervision of Guillaume Ducard and in close collaboration with Alvaro
% Detailleur.
%
% 
%
% Author: Dalim Wahby
% Version: 1.0
% Contact: wahby@i3s.unice.fr
% Date: 31.10.2025

%% Tabula Rasa
clear all; clc; close all;

%% Include toolboxes for plotting in TIKZ
addpath('C:\Users\wahby\Documents\MATLAB\Tools\matlab2tikz\src');

%% Set default for plots
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultAxesXGrid', 'on', 'defaultAxesYGrid', 'on');
% Use LaTeX for all text in this figure by default
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');


%% ======= MULTIPLE TRAJECTORY SIMULATION =======
clear; clc; close all;

%% Simulation parameters
ct = clock;

SAVE_PLOTS = 1;
SHOW_FIG = 'On';

BASE_DIR = './results/';
SAVE_DIR = strjoin([BASE_DIR, strjoin(string(ct(1:5)),'-')],'Academic_Example_Trajectories_');

if SAVE_PLOTS==1
    if ~exist(SAVE_DIR, 'dir')
        mkdir(SAVE_DIR)
    end
end

%% Simulation parameters
Ts = 0.01;
Tsim = 100;
t_vec = 0:Ts:Tsim;
num_steps = length(t_vec);
n = 2; m = 2;

IC_range = [-3, 3];

% Number of runs
num_trials = 50;

% Preallocate structure to store results
results(num_trials) = struct();

%% ========== Reference Signal ==========
input_signal = 1; % 1=constant, 2=sine
r = zeros(2, num_steps);
r_dot = zeros(2, num_steps);

if input_signal == 1
    r(1,:) = 0; r(2,:) = 0;
    r_dot(1,:) = 0; r_dot(2,:) = 0;
elseif input_signal == 2
    amp = 0.5 + 0.1*(k-1); 
    r(1,:) = amp*sin(t_vec);
    r(2,:) = amp*cos(t_vec);
    r_dot(1,:) = amp*cos(t_vec);
    r_dot(2,:) = -amp*sin(t_vec);
end

%% ========== System Initialization ==========
A = zeros(n,n); A(1,2) = 1;

% Randomize uncertainty
max_uncertainty = 0.1;
rng(42);
Delta_A = max_uncertainty * (rand(n) - 0.5);
rng(1000);
Delta_B = max_uncertainty * (rand(n) - 0.5);
A = A + Delta_A;

B = zeros(n,m); B(1,1) = 1; B(2,2) = 1;

Lambda = 0.2.*eye(n);
rng(1000);
%% ========== Model Reference ==========
Am = zeros(n,n); Am(1,2) = 1;
Bm = B;

Q = 10*eye(n); R = eye(m);
[L, P, ~] = lqr(Am, Bm, Q, R);
theta_m = L;

fprintf("====== MULTI-TRAJECTORY SIMULATION STARTED ======\n");

for k = 1:num_trials
    fprintf("\n---- Simulation %d STARTED ----\n", k);

    % Feedforward gain and learning
    theta = zeros(m, n, num_steps);
    theta(:,:,1) = zeros(m,n);
    

    % Randomize initial condition
    IC_P = IC_range(1) + diff(IC_range) * rand(n,1);
    IC_M = IC_P;

    xm = zeros(n, num_steps); xm(:,1) = IC_M;
    em = zeros(n, num_steps); em(:,1) = r(:,1) - xm(:,1);
    um = zeros(m, num_steps);

    x  = zeros(n, num_steps);  x(:,1) = IC_P;
    ex = zeros(n, num_steps);  ex(:,1) = r(:,1) - x(:,1);
    u  = zeros(m, num_steps);

    

    %% ========== Neural Network Settings ==========
    SAT_VAL = 5;
    a = 2/SAT_VAL;
    ACTIVATION_MODEL_REF = 1;
    if ACTIVATION_MODEL_REF == -1
        am = -1;
        Am = Am - Bm*theta_m;
    else
        am = a;
    end

    %% ========== Lyapunov Initialization ==========
    e = zeros(n, num_steps); e(:,1) = xm(:,1) - x(:,1);
    alpha = zeros(n, num_steps);
    alpha_dot = zeros(n, num_steps-1);
    V = zeros(1, num_steps);
    dV = zeros(1, num_steps-1);

    %% ========== Simulation ==========
    for t = 1:num_steps-1
        % System step
        ex(:,t) = r(:,t) - x(:,t);
        u(:,t) = sigmoid(theta(:,:,t)*ex(:,t), a);
        x_dot = A*x(:,t) + B*u(:,t);
        x(:,t+1) = x(:,t) + Ts*x_dot;

        % Model reference
        if am == -1
            um(:,t) = theta_m*r(:,t);
        else
            em(:,t) = r(:,t) - xm(:,t);
            um(:,t) = sigmoid(theta_m*em(:,t), am);
        end
        xm_dot = Am*xm(:,t) + Bm*um(:,t);
        xm(:,t+1) = xm(:,t) + Ts*xm_dot;

        % Errors and parameter update
        e(:,t) = xm(:,t) - x(:,t);
        e_dot = xm_dot - x_dot;

        theta_dot = compute_parameter_update( ...
            B, theta(:,:,t), ex(:,t), a, ...
            Am, A, x_dot, ...
            Bm, theta_m, am, r_dot(:,t), ...
            Lambda, P, e(:,t));
        theta(:,:,t+1) = theta(:,:,t) + Ts.*theta_dot;

        alpha(:,t) = compute_alpha( ...
            Am, A, x(:,t), Bm, theta_m, ...
            ex(:,t), am, B, theta(:,:,t), a);
        alpha_dot(:,t) = compute_alpha_dot( ...
            Am, A, x_dot, Bm, theta_m, ex(:,t), ...
            r_dot(:,t), am, B, theta(:,:,t), a, theta_dot);

        [V(:,t), ~, ~] = compute_lyapunov_value(e(:,t), P, alpha(:,t), Lambda);
        [dV(:,t), ~, ~, ~, ~] = compute_lyapunov_derivative_value( ...
            e(:,t), Am, P, ...
            Bm, theta_m, em(:,t), am, ex(:,t), ...
            alpha(:,t), Lambda, alpha_dot(:,t), e_dot);
    end

    fprintf("---- Simulation %d TERMINATED ----\n", k);

    %% ========== Store Results ==========
    results(k).x = x;
    results(k).xm = xm;
    results(k).theta = theta;
    results(k).V = V;
    results(k).dV = dV;
    results(k).IC = IC_P;
    results(k).DeltaA = Delta_A;
    results(k).DeltaB = Delta_B;
    results(k).r = r;
end

fprintf("\n====== ALL SIMULATIONS COMPLETED ======\n");

figure; hold on; grid on;

%% ---- Overlay control saturation boundaries ----
% Define a grid that covers your plotted state range
x1 = linspace(-4, 4, 400);
x2 = linspace(-4, 4, 400);
[X1, X2] = meshgrid(x1, x2);

% Compute control signals for the grid
U1 = theta_m(1,1)*X1 + theta_m(1,2)*X2;
U2 = theta_m(2,1)*X1 + theta_m(2,2)*X2;

% Draw saturation boundaries (|u_i| = 5)
contour(X1, X2, abs(U1), [SAT_VAL SAT_VAL], 'r--', 'LineWidth', 1.2, 'DisplayName', sprintf('$|u_1|=%g$', SAT_VAL));
contour(X1, X2, abs(U2), [SAT_VAL SAT_VAL], 'k--', 'LineWidth', 1.2, 'DisplayName', sprintf('$|u_2|=%g$', SAT_VAL));

title('Randomly sampled trajectories with control saturation boundaries', 'Interpreter', 'latex');
axis equal;
legend();

arrow_step = 15;
scale = 1;
traj_color = [0 0.4470 0.7410];
line_width = 0.8;

for k = 1:num_trials
    X = results(k).x(1,:);
    Y = results(k).x(2,:);
    
    % Plot main trajectory
    if k == 1
        plot(X, Y, 'Color', traj_color, 'LineWidth', line_width, 'DisplayName', 'Trajectories');
    else
        plot(X, Y, 'Color', traj_color, 'LineWidth', line_width, 'HandleVisibility', 'off');
    end
    % Add arrows along trajectory
    for i = 1:arrow_step:(num_steps-arrow_step)
        dx = X(i+arrow_step) - X(i);
        dy = Y(i+arrow_step) - Y(i);
        quiver(X(i), Y(i), dx, dy, scale, 'Color', traj_color, ...
               'MaxHeadSize', 2, 'AutoScale', 'off', 'LineWidth', 1,...
               'HandleVisibility', 'off');   
    end
end

xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
title('Randomly sampled trajectories', 'Interpreter', 'latex');
axis equal;

if SAVE_PLOTS == 1
    full_path = char(strcat(SAVE_DIR, '/', 'NRMAC_Trajectories'));
    saveas(gcf, [full_path '.png']);
    cleanfigure;

    set(gcf, 'Color', 'w');
    set(gca, 'Color', 'w');
    matlab2tikz([full_path '.tex'], ...
    'showInfo', false, ...
    'extraCode', {'\tikzset{every picture/.style={>=Stealth}}'});

end

%% Functions

function theta_dot = compute_parameter_update( ...
    B, theta, ex, a, ...
    Am, A, x_dot, ...
    Bm, theta_m, am, r_dot, ...
    Lambda, P, e)
    % This function computes the weight update for the proposed MIMO NMRAC
    
    if norm(ex)>eps
        ex_dot = r_dot - x_dot;
    
        Sigma = pinv(B*dsigmoid(theta*ex, a));
    
        xsi = Lambda*P*e ...
            + (Am-A)*x_dot ...
            + Bm*dsigmoid(theta_m*ex, am)*theta_m*ex_dot...
            - B*dsigmoid(theta*ex, a)*theta*ex_dot;
        
        % theta_dot = Sigma*xsi*pinv(ex);
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