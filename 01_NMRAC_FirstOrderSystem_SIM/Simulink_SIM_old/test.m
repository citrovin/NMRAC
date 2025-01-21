w_dot_term1 = @(ex, e, dsigma_m, dsigma_x, x_dot, xm_dot, w, r_dot) (1/(ex*b*dsigma_x))*(e + (a-am)*x_dot + bm*dsigma_m*wm*(r_dot-xm_dot));

w_dot_term2=@(w,ex,r_dot, x_dot) -(w/ex)*(r_dot-x_dot);

BETA =2;

dsigma_x = @(wex)  (4*exp(-BETA.*wex))./((1+exp(-BETA.*wex)).^2);
dsigma_m = @(wem)  (4*exp(-BETA_m*wem))/((1+exp(-BETA_m*wem))^2);

text_x = -10:0.01:10;
SHOW_FIG = 'On';
fig = figure;
set(fig, 'Visible', SHOW_FIG);
plot(text_x,dsigma_x(text_x));
grid on;