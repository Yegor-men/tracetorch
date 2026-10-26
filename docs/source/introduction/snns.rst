Understanding the neuron updates
================================

The SNN layers differ in what they remember and how they turn that memory into
an output. We'll start with one membrane value, then add firing, input filtering
and output history. The equations use one feature to keep the notation readable;
the implementation applies the same operations across the final feature axis.

Membrane memory
---------------

A leaky integrator, ``LI``, adds the current input :math:`x_t` to a decayed copy
of its previous membrane :math:`m_{t-1}`:

.. math::

   m_t = \beta m_{t-1} + x_t, \qquad y_t = m_t.

The decay :math:`0 < \beta < 1` controls how quickly old input fades. With no
further input, an initial membrane value becomes :math:`\beta^n m_0` after
:math:`n` calls. We usually work directly with the decay: 0.9 keeps 90% of a
passive trace at each step, while 0.99 keeps 99%.

When we want a rough duration, the **time horizon** gives a convenient scale:

.. math::

   H = \frac{1}{1-\beta}, \qquad \beta = 1-\frac{1}{H}.

Thus 0.9 gives a horizon of 10 timesteps, and 0.99 gives 100. This is also the
total weight :math:`\sum_{k=0}^{\infty}\beta^k` of the exponential history.
It isn't a point where memory disappears: after roughly one horizon, a slow
passive trace retains about :math:`e^{-1}`, or 37%, of its initial value.
New input and firing resets can change that trace, so the horizon describes
decay rather than guaranteeing how long a model remembers a fact.

For recordings with a known frame duration we can also express the horizon
in seconds. For text it counts bytes, with no assigned physical duration.
``tt.utils.convert_decay`` converts between these descriptions, including
half-life when we specifically want the time for a trace to fall by half.
The :doc:`../tutorials/shd` tutorial uses the recording length to choose decays,
while :doc:`../tutorials/language_model` explores uniform random decays without
choosing a single sequence duration in advance.

LI accumulates input, so a constant input :math:`c` approaches
:math:`c/(1-\beta)`. ``LIEMA`` instead uses an exponential moving average:

.. math::

   m_t = \beta m_{t-1} + (1-\beta)x_t.

The extra input factor changes the steady value to :math:`c`, which makes this
form useful for smoothing a signal without increasing its constant magnitude.
Both forms retain time-dependent information, but both are linear in the input
sequence for a fixed decay and initial state. Memory alone doesn't make a
transformation nonlinear.

From a membrane to firing
-------------------------

``LIB`` adds a positive threshold :math:`\theta` and a previous firing state
:math:`s_{t-1}`. It first subtracts the previous event's reset, then decays the
membrane and adds the new input:

.. math::

   m_t &= \beta(m_{t-1} - \theta s_{t-1}) + x_t, \\
   s_t &= H(m_t-\theta), \qquad y_t = s_t,

where :math:`H(z)` is 1 when :math:`z \geq 0` and 0 otherwise. The threshold
therefore determines whether the current membrane produces a firing event.
That event affects the membrane on the *next* call: reset is delayed and occurs
before decay, rather than setting the current membrane to zero immediately.

We can see the distinction with a decay of 0.9, threshold 1 and constant input
0.6. Starting from zero, the first two membrane values are 0.6 and 1.14, so the
second call fires. The next update uses
:math:`0.9(1.14-1)+0.6=0.726`, which falls below threshold again. The membrane
stores continuous values even though the returned output is binary.

This firing rule is nonlinear. Its hard threshold has zero derivative away
from the boundary, so the default function uses a surrogate derivative during
training. For distance :math:`z=m-\theta`, we define

.. math::

   g(z) = \sigma(4z), \qquad g'(z)=4g(z)(1-g(z)),
   \qquad \sigma(z)=\frac{1}{1+e^{-z}}.

``deterministic`` returns the hard event in the forward pass but uses
:math:`g'(z)` during backpropagation. ``stochastic`` samples a hard event with
probability :math:`g(z)` and uses that same surrogate derivative. With
``smooth``, the forward output itself is :math:`g(z)`, so ordinary differentiation
gives its gradient. The smooth intensity also enters the next reset; changing
the firing function changes the forward dynamics as well as the output values.

The :doc:`../tutorials/mnist` tutorial plots the hard output, smooth function
and surrogate derivative, then follows them into a training update.

Representing positive and negative events
-----------------------------------------

``LIT`` uses separate positive and negative thresholds, both stored as positive
magnitudes. If :math:`s_t^+` is the positive event and :math:`s_t^-` is the signed
negative event, its update is

.. math::

   r_{t-1} &= \theta^+s_{t-1}^+ + \theta^-s_{t-1}^-, \\
   m_t &= \beta(m_{t-1}-r_{t-1})+x_t, \\
   s_t^+ &= f(m_t-\theta^+), \\
   s_t^- &= -f(-\theta^- - m_t), \\
   y_t &= s_t^+ + s_t^-.

Here :math:`f` is the selected firing function. With deterministic firing,
the output is negative, zero or positive; with smooth firing, both branches
have continuous intensities. We retain both event states separately because
their sum doesn't preserve the information each reset term needs.

``LITS`` adds positive output scales :math:`a^+` and :math:`a^-`:

.. math::

   y_t = a^+s_t^+ + a^-s_t^-.

The scales change what reaches the next layer, while reset and recurrent history
still use the unscaled events. That separates the threshold that causes firing
from the magnitude of its downstream contribution.

Filtering input before integration
----------------------------------

An ``S`` prefix adds a synaptic history :math:`a_t` before the membrane. For
example, ``SLIB`` replaces the direct membrane input with a filtered current:

.. math::

   a_t &= \alpha a_{t-1} + (1-\alpha)x_t, \\
   m_t &= \beta(m_{t-1}-\theta s_{t-1}) + a_t.

The synaptic decay :math:`\alpha` and membrane decay :math:`\beta` now control
two different histories. An input can keep contributing through the synapse
after its original timestep, while the membrane combines that current with
its own remaining charge. A larger synaptic decay gives a slower response and
a longer tail.

Keeping a trace of earlier output
---------------------------------

An ``R`` prefix adds a filtered history :math:`q_t` of the previous firing
output. For ``RLIB``, with recurrent decay :math:`\gamma` and weight :math:`w_r`,
we get

.. math::

   q_t &= \gamma q_{t-1} + (1-\gamma)s_{t-1}, \\
   m_t &= \beta(m_{t-1}-\theta s_{t-1}) + x_t + w_rq_t.

This gives the neuron's own output a further influence on future updates,
in addition to its effect through reset. The trace is per feature; it doesn't
mix all neurons through an internal dense recurrent matrix. We use an ordinary
linear projection when the model needs learned mixing between neurons.
In ternary layers, the non-dual recurrent trace filters the sum of the two
unscaled signed events.

Separating positive and negative histories
------------------------------------------

The ``D`` prefix gives positive and negative input separate traces, which can
learn different decays. For the continuous ``DLI`` layer:

.. math::

   m_t^+ &= \beta^+m_{t-1}^+ + \max(x_t,0), \\
   m_t^- &= \beta^-m_{t-1}^- + \min(x_t,0), \\
   y_t &= m_t^+ + m_t^-.

Firing variants calculate their output from the combined membrane and divide
the reset equally between the two membrane traces *before* each branch's decay.
When synaptic traces are present, each signed synaptic history feeds its
corresponding membrane directly. Dual recurrent ternary layers likewise filter
the positive and negative events separately, rather than summing and splitting
them again.

These options describe different state updates, rather than a ranking of neuron
quality. A model might need only a membrane, or it might have a reason to retain
input and output histories separately. :doc:`layers` maps these choices to the
available names, while the API reference gives the full update for each variant.
