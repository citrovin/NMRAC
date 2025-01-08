%% TODO: Description of script

%% Tabula Rasa
clear all; clc; close all;

%% Set default for plots
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultAxesXGrid', 'on', 'defaultAxesYGrid', 'on');

ct = clock;

SAVE_PLOTS = 1;
SHOW_FIG = 'On';

BASE_DIR = './results/';
SAVE_DIR = strjoin([BASE_DIR, strjoin(string(ct(1:5)),'-')],'');

if SAVE_PLOTS==1
    if ~exist(SAVE_DIR, 'dir')
        mkdir(SAVE_DIR)
    end
end

%% Simulation parameters

Ts = 0.01;
Tsim = 10;

% Define the time vector
t_vec = 0:Ts:Tsim;
num_steps = length(t_vec);

TOLLERANCE = eps;

SAT_VAL = 1;
a = 2/SAT_VAL; % Sigmoid saturates at 2/a.
fprintf("Saturation value: %d\n", SAT_VAL);

am = a;

%% Init reference Signal
input_signal = 1;

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

elseif input_signal == 3
    % Smooth pulse signal
    bias = 0.1;
    sat = 2/0.1;
    amplitude = 0.25;
    
    
    amplitude = 1;
    r(1,:) = sigmoid(amplitude.*sin(t_vec), sat) + bias;
    
    r_dot(1,:) = dsigmoid_ref(amplitude.*sin(t_vec), sat).*amplitude.*cos(t_vec);
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



%% System
% Dymanics: x_dot = -Am*x + B*u
A = Am;
B = Bm;

theta = zeros(1, num_steps);
theta(1,1) = 0;

% Learning rate
if input_signal==1
    c = 0.05;
elseif input_signal==2
    c = 0.15;
elseif input_signal==3
    c = 0.005;
end
IC_P = IC_P_m;
x = zeros(1, num_steps);
x(1,1) = IC_P;

ex = zeros(1, num_steps);

%% Parameters for the update mechanism and Lyapunov function
e = zeros(1, num_steps);

V = zeros(1, num_steps);
alpha = zeros(1, num_steps);

alpha_dot = zeros(1, num_steps-1);
dV = zeros(1, num_steps-1);
dV_term1 = zeros(1, num_steps-1);
dV_term2 = zeros(1, num_steps-1);
dV_term3 = zeros(1, num_steps-1);
dV2 = zeros(1, num_steps-1);

 %% Simulation

x_dot_prev = zeros(size(x(:,1))); % Initial x_dot
xm_dot_prev = zeros(size(xm(:,1))); % Initial xm_dot
theta_dot_prev = zeros(size(theta(:,1))); % Initial theta_dot



fprintf("------ SIMUATION STARTED ------\n");
for t = 1:(num_steps-1)
    % Step System
    ex(:,t) = r(:,t) - x(:,t);
    

    u = sigmoid(theta(:,t)*ex(:,t), a);
    x_dot = -A*x(:,t) + B*u;

    % x(:,t+1) = (Ts/2) * (x_dot + x_dot_prev) + x(:,t); % Tustin update
    x(:,t+1) = x(:,t) + Ts*x_dot;
    
    ex_dot = r_dot(:,t) - x_dot;

    % Step Model Ref
    em(:,t) = r(:,t) - xm(:,t);
    
    um = sigmoid(theta_m*em(:,t), a);
    xm_dot = -Am*xm(:,t) + Bm*um;
    
    % xm(:,t+1) = (Ts/2) * (xm_dot + xm_dot_prev) + xm(:,t); % Tustin update
    xm(:,t+1) = xm(:,t) + Ts*xm_dot;

    % Store previous derivatives for next iteration
    x_dot_prev = x_dot;
    xm_dot_prev = xm_dot;

    % Compute error for update
    e(:,t) = xm(:,t) - x(:,t);
    e_dot = xm_dot - x_dot;
    
    % Update params
    theta_dot = update_params(e(:,t), ex(:,t), ex_dot, x_dot, theta(:,t), theta_m, ...
                              A, Am, B, Bm, c, a, am, TOLLERANCE);
    theta(t+1) = theta(t) + Ts*theta_dot;
    % theta(:,t+1) = theta(:,t) + (Ts/2) * (theta_dot + theta_dot_prev); % Tustin update
    theta_dot_prev = theta_dot;
    
    % Compute Lyapunov Function
    alpha(:,t) = (A-Am)*x(1,t) ...
                + Bm*sigmoid(theta_m*ex(1,t), am) ...
                - B*sigmoid(theta(:,t)*ex(1,t), a);
    V(:,t) = e(:,t)^2 + alpha(:,t)^2;
    
    alpha_dot(:,t) = compute_alpha_dot(Am, A, x_dot, ...
        Bm, theta_m, ex(:,t), r_dot(:,t), am, B, theta(:,t), a, theta_dot);
    

    [dV(:,t), dV_term1(:,t), dV_term2(:,t), dV_term3(:,t), dV2(:,t)] = compute_lyapunov_derivative_value( ...
        e(:,t), Am, 1, ...
        Bm, theta_m, em(:,t), am, ex(:,t), ...
        alpha(:,t), c, alpha_dot(:,t), ...
        e_dot);
end

% Compute last values for all needed values
ex(1,end) = r(1,end) - x(1,end);
em(1,end) = r(1,end) - xm(1,end);

% Compute last Lyapunov values
e(:,end) = xm(1,end) - x(1, end);
alpha(:,end) = compute_alpha( ...
    A, Am, x(:,t), ...
    Bm, theta_m, ex(:,t), am, ...
    B, theta(:,t), a);
V(:,end) =  e(:,end)^2 + alpha(:,end)^2;

% Compute dV
[dV(:,end), dV_term1(:,end), dV_term2(:,end), dV_term3(:,end), dV2(:,end)] = compute_lyapunov_derivative_value( ...
        e(:,end), Am, 1, ...
        Bm, theta_m, em(:,end), am, ex(:,end), ...
        alpha(:,end), c, alpha_dot(:,end), ...
        e_dot);


%% Plots

% Plot the system response
fig = figure();
set(fig, 'Visible', SHOW_FIG);
plot(t_vec, r, ...
    t_vec, xm, ...
    t_vec, x);
legend("r", "x_m", "x");
xlabel("Time [s]");
ylabel("Response");
title("System Response");
if SAVE_PLOTS == 1
    full_path = char(strcat(SAVE_DIR, '/', 'NMRAC_First-Order_Response.png'));
    saveas(gcf, full_path);
end

fig = figure();
set(fig, 'Visible', SHOW_FIG);
plot(t_vec, em, ...
    t_vec, e, ...
    t_vec, ex);
legend("e_m", "e", "ex");
xlabel("Time [s]");
ylabel("Error");
title("Error over time");

if SAVE_PLOTS == 1
    full_path = char(strcat(SAVE_DIR, '/', 'NMRAC_First-Order_Error.png'));
    saveas(gcf, full_path);
end


% Plot the Lyapunov function
fig = figure();
set(fig, 'Visible', SHOW_FIG);
subplot(2,1,1);
% plot(t_vec, V, ...
%     t_vec, e.^2, ...
%     t_vec, alpha.^2 ...
%     );
% legend("V", "e^2", "\alpha^2");
plot(t_vec, V);
legend("V");
xlabel("Time [s]");
ylabel("V(e,\alpha)");
title("Lyapunov function");


subplot(2,1,2);
% plot(t_vec(:,2:num_steps), dV,  ...
%     t_vec(:,2:num_steps), dV_term1, ...
%     t_vec(:,2:num_steps), dV_term2, ...
%     t_vec(:,2:num_steps), dV_term3);
% legend("dV", "dV_{Q}", "dV_{\gamma}",  "dV_{\alpha}");

plot(t_vec(:,2:num_steps), dV);
xlabel("Time [s]");
ylabel("dV/dt");
title("Time Derivative of Lyapunov Function");

sgtitle("Lyapunov Function");

if SAVE_PLOTS == 1
    full_path = char(strcat(SAVE_DIR, '/', 'NMRAC_First-Order_Lyapunov.png'));
    saveas(gcf, full_path);
end


fig = figure();
set(fig, 'Visible', SHOW_FIG);
plot(t_vec, theta_m*ones(1,num_steps), ...
    t_vec, theta);
legend("\theta_m", "\theta");
xlabel("Time [s]");
ylabel("\theta");
title("Gains over time");

if SAVE_PLOTS == 1
    full_path = char(strcat(SAVE_DIR, '/', 'NMRAC_First-Order_Parameters.png'));
    saveas(gcf, full_path);
end

% Plot the difference in dynamics
fig = figure();
set(fig, 'Visible', SHOW_FIG);
plot(t_vec, alpha(1,:),...
    t_vec, e(1,:))
xlabel("Time [s]");
ylabel("Magnitude");
legend("\alpha", "e");
sgtitle("Error in states and dynamics over time");

if SAVE_PLOTS == 1
    full_path = char(strcat(SAVE_DIR, '/', 'NMRAC_First-Order_Alpha.png'));
    saveas(gcf, full_path);
end




%% Function

function theta_dot = update_params(e, ex, ex_dot, x_dot, theta, theta_m, ...
    A, Am, B, Bm, c, a, am, TOLLERANCE)
    if abs(ex)<TOLLERANCE
        theta_dot = 0;
    else
        theta_dot = 1/(ex*B*dsigmoid(theta*ex, a)) * ( ...
            (1/c) * e + (A-Am)*x_dot ...
            + Bm*dsigmoid(theta_m*ex, am)*theta_m*ex_dot) ...
            - (theta/ex) * ex_dot;
    end
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


function du = dsigmoid_ref(x, a)
    % This function returns the derivative of the sigmoid function with
    % respect to x, evaluated at x.
    du = 4.*exp(-a.*x)./(1+exp(-a.*x)).^2;
end

function alpha = compute_alpha( ...
    A, Am, x, ...
    Bm, theta_m, ex, am, ...
    B, theta, a)
    alpha = (A-Am)*x ...
                + Bm*sigmoid(theta_m*ex, am) ...
                - B*sigmoid(theta*ex, a);
end

function [dV, dV_term1, dV_term2, dV_term3, dV2] = compute_lyapunov_derivative_value( ...
    e, Am, P, ...
    Bm, theta_m, em, am, ex, ...
    alpha, Lambda, alpha_dot, ...
    e_dot)
    
    dV_term1 = -2.*e'*P*Am*e;
    gamma = Bm*(sigmoid(theta_m*em, am) - sigmoid(theta_m*ex, am));
    dV_term2 = 2.*e'*P*gamma;
    dV_term3 = 2.*alpha'*(P*e+inv(Lambda)*alpha_dot);

    dV = dV_term1 + dV_term2 +  dV_term3;
    
    dV2_term1 = e'*P*e_dot + e_dot'*P*e;
    dV2_term2 = alpha'*inv(Lambda)*alpha_dot + alpha_dot'*inv(Lambda)*alpha;
    
    dV2 = dV2_term1 + dV2_term2;
    
end

function alpha_dot = compute_alpha_dot(Am, A, x_dot, Bm, theta_m, ex, r_dot, am, B, theta, a, theta_dot)
    % This term defines the difference in Dynamics between the model
    % reference and the closed-loop system
    ex_dot = r_dot - x_dot;
    alpha_dot = (Am-A)*x_dot ...
            + Bm*dsigmoid(theta_m*ex, am)*theta_m*ex_dot ...
            - B*dsigmoid(theta*ex, a)*(theta_dot*ex + theta*ex_dot);
end