function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units> 100 x 256
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>. 256 x 1
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>. 100 x 1
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
    % E(v,h) = -sum(v*h*w)  
    c = size(visible_state,2);
    G = sum(sum(((visible_state * hidden_state') .* rbm_w') ./ c));
end
