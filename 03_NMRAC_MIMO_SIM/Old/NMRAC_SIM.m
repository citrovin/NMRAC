%% Tabula Rasa
clear all; clc;

%% Simulation parameters
Ts = 0.01; % Sampling time
Tsim = 10; 

t = 0:Ts:Tsim; % Time vector
num_steps = length(t);   % Number of simulation steps

UPDATE_THRESHOLD = eps; % 1e-3;


%% Reference signal
r_1 = 0.9*sin(t) + 1;
r_2 = 0.9*cos(t) + 1;

r = 3*ones(2, num_steps);
% r(1,:) = r_1;
% r(2,:) = r_2;


%% Reference model
% Internal dynamics
Am = [0 1; 0 -0.1];

% Input dynamics
Bm = [1 0; 0 1];

% Gain matrix
theta_m = [1 1; 1 1]; 

% Initial conditions for model reference
xm = [0; 0];

% Solution to Lyapunov equation
P = [1 0; 0 1];

% Preallocate arrays to store results
xm_history = zeros(2, num_steps);
xm_history(:,1) = xm;

um_history = zeros(2, num_steps);
um_sat_history = zeros(2, num_steps);

em_history = zeros(2, num_steps);
em_history(:,1) = r(:,1) - xm;

%% System definition
% Internal dynamics
% A = [0 1; 0 0]; 
A = Am;

% Input dynamics
% B = [1 0; 0 1];
B = Bm;

% Gain matrix
% theta = [0 0; 0 0];
theta = 0.5.*ones(size(theta_m));
% Learning rate
lambda = 0.01.*[1 0; 0 1];

% Initial state vector
x = [0; 0];

x_history = zeros(2, num_steps);  % State history (2 states)
x_history(:,1) = x;

u_history = zeros(2, num_steps);
u_sat_history = zeros(2, num_steps);

ex_history = zeros(2, num_steps);
ex_history(:,1) = r(:,1)-x;

theta_history = zeros(2, 2, num_steps);
theta_history(:,:,1) = theta;

dtheta_history = zeros(2, 2, num_steps);

%% Simulation
% Sigmoid function
a = 1; 
sigma = @(x) 2.*(1-exp(-a.*x))./(a.*(1+exp(-a.*x)));
dsigma = @(x) diag(4.*exp(-a.*x)./(1+exp(-a.*x)).^2);

% Lyapunov function
alpha = @(x, ex, theta) (Am-A)*x + Bm*sigma(theta_m*ex) - B*sigma(theta*ex);
V_alpha = @(alpha)  alpha'*lambda*alpha;
V_e = @(e) e'*P*e;
V = @(V_e, V_alpha) V_e + V_alpha;

alpha_history = zeros(2, num_steps);
e_history = zeros(2, num_steps);

alpha_history(:,1) = alpha(e_history(:,1),ex_history(:,1), theta_history(:,:,1));
e_history(:,1) = xm-x;


V_e_history = zeros(1, num_steps);
V_alpha_history = zeros(1, num_steps);
V_history = zeros(1, num_steps);

V_e_history(1,1) = V_e(e_history(:,1));
V_alpha_history(1,1) = V_alpha(alpha_history(:,1));
V_history(1,1) = V_e_history(1,1) + V_alpha_history(1,1);

% Lyapunov time derivative
lyapunov_gamma_term = @(e, em, ex, theta) 2*e'*P*Bm*(sigma(theta_m*em) - sigma(theta*ex));
dalpha = @(dx, ex, dex, theta, dtheta) (Am-A)*dx + Bm*dsigma(theta_m*ex)*theta_m*dex - B*dsigma(theta*ex)*(dtheta*ex + theta*dex);
dV = @(e, lyapunov_gamma_term, alpha, dalpha) e'*(Am'*P+P*Am)*e + lyapunov_gamma_term + 2*alpha'*(P*e+inv(lambda)*dalpha);
%dV = @(e, lyapunov_gamma_term, alpha, dalpha) e'*Am'*P*e + lyapunov_gamma_term + 2*alpha'*(P*e+inv(lambda)*dalpha);

gamma_history = zeros(1, num_steps);

for i = 2:length(t)
    %% Reference model
    % Compute control inputs (you will replace this with your adaptive control law)
    em = r(:,i)-xm;
    um = theta_m * em;
    um_sat = sigma(um);
    
    % Update state using Euler's method
    xm_dot = Am * xm + Bm * um;
    xm = xm + xm_dot * Ts;
    
    % Save values
    xm_history(:, i) = xm;
    um_history(:,i) = um;
    um_sat_history(:,i) = um_sat;
    
    %% System
    ex = r(:,i)-x;
    u = theta * ex;
    u_sat = sigma(u);
    
    % Update state using Euler's method
    x_dot = A * x + B * u_sat;
    x = x + x_dot * Ts;

    ex_history(:,i) = ex;

    dex = (ex_history(:,i-1) - ex_history(:,i))./Ts;
    
    % Save values
    x_history(:, i) = x;
    u_history(:,i) = u;
    u_sat_history(:,i) = u_sat;

    %% Error
    e = xm - x;
    e_history(:,i) = e;
    de = (e_history(:,i-1) - e_history(:,i))./Ts;
    

    %% Weight Update
    if abs(norm(e))<UPDATE_THRESHOLD
        dtheta = 0;
    else
        SIGMA_plus = pinv(B*dsigma(theta*ex));
        xsi = lambda*P*e + (Am-A)*x_dot + Bm*dsigma(theta_m*ex)*theta_m*dex - B*dsigma(theta*ex)*theta*dex;
        ex_plus = pinv(ex);

        dtheta = SIGMA_plus*xsi*ex_plus;
    end
    
    theta = theta + dtheta * Ts;
    
    theta_history(:,:,i) = theta;
    dtheta_history(:,:,i) = dtheta;
    
    %% Lyapunov Function
    V_e_history(:,i) = V_e(e);

    alpha_history(:,i) = alpha(x, ex, theta);
    V_alpha_history(:,i) = V_alpha(alpha_history(:,i));
    V_history(:,i) = V(V_e_history(:,i), V_alpha_history(:,i));
    
    gamma_history(:,i) = lyapunov_gamma_term(e, em, ex, theta);
    dalpha_history(:,i) = dalpha(x_dot, ex, dex, theta, dtheta);

    j = 0;
    dV_history(:,i) = dV(e, gamma_history(:,i), alpha_history(:,i), dalpha_history(:,i));

end

%% Plot states
fig1 = figure(1);
subplot(2,1,1);
plot(t, r(1,:), ...
    t, x_history(1, :),...
    t, xm_history(1, :), ...
    LineWidth=1.5);
xlabel('Time [s]');
ylabel('x_1');
legend('r', 'x', 'x_m');
grid on;

subplot(2,1,2);
plot(t, r(2,:), ...
    t, x_history(2, :), ...
    t, xm_history(2,:), ...
    LineWidth=1.5);
xlabel('Time [s]');
ylabel('x_2');
grid on;
legend('r', 'x', 'x_m');
grid on;

sgtitle('System Response');


%% Plot theta
fig2 = figure(2);
subplot(2,2,1)
plot(t, theta_m(1,1) * ones(size(t)), ...
    t, squeeze(theta_history(1, 1, :)), ...
    LineWidth=1.5);
grid on;
title('$\theta_{1,1}$', Interpreter='latex');
legend('\theta_m', '\theta');

subplot(2,2,2)
plot(t, theta_m(1,2) * ones(size(t)), ...
    t, squeeze(theta_history(1, 2, :)), ...
    LineWidth=1.5);
grid on;
title('$\theta_{1,2}$', Interpreter='latex');
legend('\theta_m', '\theta');

subplot(2,2,3)
plot(t, theta_m(2,1) * ones(size(t)), ...
    t, squeeze(theta_history(2, 1, :)), ...
    LineWidth=1.5);
grid on;
title('$\theta_{2,1}$', Interpreter='latex');
legend('\theta_m', '\theta');

subplot(2,2,4)
plot(t, theta_m(2,2) * ones(size(t)), ...
    t, squeeze(theta_history(2, 2, :)), ...
    LineWidth=1.5);
grid on;
title('$\theta_{2,2}$', Interpreter='latex');
legend('\theta_m', '\theta');

sgtitle('Convergence of \theta');


%% Plot Lyapunov function
fig3 = figure(3);
subplot(2,1,1)
plot(t(2:end), V_history(2:end), ...
    t(2:end), V_alpha_history(2:end), ...
    t(2:end), V_e_history(2:end), ...
    LineWidth=1.5);
grid on;
legend('V(e,\alpha)', 'e', '\alpha');
title('Lyapunov function');

subplot(2,1,2)
dV_term1 = sum((e_history' * P) .* e_history', 2)';

plot(t(2:end), dV_history(2:end), ...
    t(2:end), gamma_history(2:end), ...
    t(2:end), dV_term1(2:end), ...
    LineWidth=1.5);
grid on;
legend('dV(e,\alpha)', '2e^TP\gamma', 'dV term1');
title('Derivative of the Lyapunov Function');

%% Plot santiy check
fig4 = figure(4);
subplot(2,1,1)
sanity_11 = P*e_history;
sanity_21 = inv(lambda)*dalpha_history;
sanity_3 = sanity_11+sanity_21;

plot(t, sanity_11(1,:), ...
    t, sanity_21(1,:), ...
    t, sanity_3(1,:), ...
    LineWidth=1.5);
grid on;
legend('Pe', '\Lambda d\alpha ', 'Difference');
title('Sanity check Dimension 1');

subplot(2,1,2)
plot(t, sanity_11(2,:), ...
    t, sanity_21(2,:), ...
    t, sanity_3(2,:), ...
    LineWidth=1.5);
grid on;
legend('Pe', '\Lambda d\alpha ', 'Difference');
title('Sanity check Dimension 2');