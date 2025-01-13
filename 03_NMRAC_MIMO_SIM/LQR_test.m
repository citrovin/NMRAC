clear all; clc;
n = 2;
m = 2;
Ts = 0.01;

% Internal dynamics (pure double integrator)
A = zeros(n,n);
A(1,2) = 1;
A(2,2) = 0;

% Input dynamics
B = zeros(n,m);
B(2,1) = 1;
if m==2 
    B(2,2) = -1;
end

sys = ss(A, B, eye(n), zeros(n,m));
sys_dis = c2d(sys, Ts);

Q = eye(n);
R = eye(m);


K = dlqr(sys_dis.A, sys_dis.B, Q, R)