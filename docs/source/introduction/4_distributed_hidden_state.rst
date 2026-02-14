4. The Distributed Hidden State
===============================

As with any recurrent architecture, the biggest challenge is for the model to figure out what to keep and what to discard.
Confined by working memory (hidden state(s)), the model cannot simply store absolutely everything. Transformers work because
we keep everything in context, and dynamically create an analogue for the hidden state based on the current sequence length.

Perhaps the most well known of RNN architectures is the Long Short-Term Memory (LSTM). It solves the problem of figuring
out what to remember and what to forget by utilizing gates to control the flow of information. It works with a literal
``hidden`` and ``cell`` hidden states (short and long term), and uses gates that work on sigmoid (values between 0 and 1)
to control how much of the information to keep, how much to forget; how to update the state and what to return. The more
modern and simplified alternative to LSTMs is the Gated Recurrent Unit (GRU), which only use 2 gates instead of 3,
and 1 hidden state instead of 2, but the principle remains effectively identical: gates decide what's important and what
isn't.

SNNs on the other hand are simultaneously significantly simpler in structure and more complex in interpretation, working
on a much higher, abstract level than literal gates controlling the portion of information to keep.

In the simplest configuration, the ``LIB`` neuron works like any MLP, except with the option of dipping into time. ``mem``
stores the amount of information the neuron has accumulated, which gradually decays because of ``beta``. Firing means that
the neuron has accumulated enough information to flip the gate and signal to the downstream layer. A high ``beta`` (close to 1)
means that information is retained over a long period of time. The formula for approximating the number of timesteps
that information is retained for is ``1 / (1 - beta)``. If ``beta`` is 0.9, then it remembers the last 10 timesteps. If it's
0.99, then 100 timesteps, and so on. ``beta`` dictates the temporal window that the neuron remembers information over.
But even then, the moment that ``mem`` reaches the threshold, it resets, and now the neuron has lost all the information
it has accumulated. As you can see, this is quite problematic, as no matter how high a number we pick, we're always
constrained by the highest ``beta`` value in the entire network. The model fundamentally cannot remember information for
longer than that, and that's assuming the perfect configuration.

So, we add synaptic dynamics with ``syn`` and ``alpha`` decay. ``syn`` is an Exponential Moving Average (EMA) of the input
that the neuron receives at each timestep, and then passes that value in to ``mem``. If ``alpha`` is close to 0, then
it's as if it didn't exist as the signal is instantly integrated into ``syn``, the old information is lost, and thus the
only thing that gets to ``mem`` is the newly received input. However, the closer that ``alpha`` is to 1, the longer its
temporal window is. If ``alpha`` is 0.9, then that means that the value stored in ``syn`` is effectively an EMA of the
past 10 timesteps, and now, ``mem`` receives not the instantaneous information, but the information that the neuron has
received over time. Now, even if the neuron fires and ``mem`` resets, it's not as critical because ``syn`` can re-fill
it in the next timestep given the proper circumstances. Even now, this isn't enough. We still can't reach a theoretically
infinite attention window.

We can add bipolar spiking, which means having a negative threshold in conjunction with the positive threshold. This now
means that negative and positive signals are separate in nature, it's not as simple as "positive = excitatory, negative =
inhibitory". Negative and positive spikes are scaled by ``pos_scale`` and ``neg_scale``, which means that the neuron can
learn to output 3 independent signals. It can learn to never output positive or negative spikes by tuning the according
``*_scale`` down to 0. It can learn to have a static output by having both ``*_scale`` down at 0; it can even learn to
ignore polarity make positive and negative spikes be of one polarity by making one of the ``*_scale`` the negative of the
other ``*_scale``. With bipolar spiking, a true inhibitory signal is now a 0, not just negative values, and for this to
make sense and work, the magnitude of the signal becomes an even more important factor than before, and thus accordingly
the ``pos_threshold`` and ``neg_threshold`` learn independently: the closer they are to zero, the more sensitive the neuron
effectively is to the respective signal. But this is merely an expansion of the capability, we still have not yet reached
infinite context.

Having already separated positive and negative signals by separate thresholds and scales, it would also make sense that we
have separate traces for them too. Thus we introduce Duality: for each hidden state and for each parameter, we have a positive and negative variant,
and attain the "true" hidden state by summing the two. Different decays for different polarities means that we can unlock
unique dynamics: different speeds at which the neuron adapts based on the polarity of the signal it receives, such as
fast adaptation to positive signals but slow adaptation to negative signals. We can introduce this duality to *any* hidden
state, which means we quickly unlock another vast amount of dynamics which are in some way meaningful.

And finally, we add recurrence. ``rec`` stores the running average of the neuron's output, decayed via ``gamma``, and re-integrated
back into ``mem`` at each timestep, multiplicatively scaled by ``rec_weight``. It is with recurrence that we are finally
able to unlock complex firing patterns and self-supporting states. If ``rec`` is positive (the model output positive spikes),
and ``rec_weight`` is positive, then the neuron can enter a permanent firing state that requires heavy external suppression
in order to stop firing. On the contrary, if ``rec_weight`` is negative, and suppose that the ``gamma`` decay is very low
(fast reset, small temporal window), then the instant that the neuron outputs a positive spike, it will make ``rec``
positive too, but the negative ``rec_weight`` will re-integrate ``rec`` back into ``mem`` to make it negative, and thus
the model will constantly alternate between positive and negative spike outputs. Adjusting the ``beta`` and ``gamma`` decay
and we can get arbitrary spacing between the alternating patterns: [-1,1,-1,1] or [-1,0,1,0,-1], et cetera. We also add
``bias``, a value that just directly gets integrated into ``mem`` at each timestep. Considering bipolar spiking, and the
learnable ``pos_scale`` and ``neg_scale``, we can create an autotelic neuron that without any input has a certain firing
pattern, and reach virtually any firing sequence we need. They can function as conditional tempo clocks, working at different
speeds in tandem, and downstream layers base their outputs on how they align, akin to how fourier series works.

Parallelize the neurons, initialize them with different (can even do random) ``alpha``, ``beta``, ``gamma``, ``pos_threshold``,
``neg_threshold``, ``pos_scale``, ``neg_scale``, ``rec_weight``, ``bias``, consider the doubling due to duality; and the number of dynamics we begin with and can already
capture is unfathomably vast. Chain the layers, and it would be surprising if the model *couldn't* learn some function.
Use residual connections, pairing each SNN layer with an ``nn.Linear`` or equivalent, make them take as input a floating
point vector (electrical current for the neurons), and as output return the sum of the input and the output spikes passed
through the ``nn.Linear``, and you've solved the issue of vanishing gradients, and each layer is effectively functioning
as a high level, recurrent tensor editor.

Very quickly, interpretability becomes something very difficult to do. ``mem``, ``syn``, ``rec`` are simple enough to understand
in principle, but combine the three together, and consider the presence of downstream and upstream layers taking play, and
the dynamics that are now possible become far more than blind signal accumulation. We thus finally achieve the Distributed
Hidden State (DHS). Each hidden state in itself means very little, and it is when they work together that we get something
abstract but meaningful.
