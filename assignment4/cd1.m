function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units> 100 x 256
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases> 256 x 10
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>. 

%The variation that we're using here is the one where every time after calculating a conditional probability for a unit, 
% we sample a state for the unit from that conditional probability (using the function sample_bernoulli), and then we forget about the conditional probability. 
%There are other variations where we do less sampling, but for now, we're going to do sampling everywhere: 
%we'll sample a binary state for the hidden units conditional on the data; 
%we'll sample a binary state for the visible units conditional on that binary hidden state (this is sometimes called the "reconstruction" for the visible units); 
% and we'll sample a binary state for the hidden units conditional on that binary visible "reconstruction" state. 
%Then we base our gradient estimate on all those sampled binary states
    visible_data = sample_bernoulli(visible_data);
    c = size(visible_data,2);
    hidden_state0 = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w, visible_data));
    reconstruction = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w, hidden_state0));
    hidden_state1 = visible_state_to_hidden_probabilities(rbm_w, reconstruction); %sample_bernoulli(
    %d_G_by_rbm_w_pos = configuration_goodness_gradient(visible_data, hidden_state0)
    %d_G_by_rbm_neg = configuration_goodness_gradient(reconstruction, hidden_state1)
    ret = ((((visible_data * hidden_state0')) - ((reconstruction * hidden_state1'))) ./ c)'; %.* (1 + d_G_by_rbm_w_pos') ./ (1 + d_G_by_rbm_neg') 
end
