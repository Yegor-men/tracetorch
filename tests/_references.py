"""Slow scalar SNN oracle: Python arithmetic, no production layer helpers.

The update order is reset from the previous event, filter the new current,
update recurrent history, integrate membranes, then emit the current event.
Keeping neuron instances separate makes these tests independent of tensor
broadcasting and vectorization in the library.
"""

import math


class ScalarNeuron:
    def __init__(self, family, parameters, *, dual=False, synaptic=False,
                 recurrent=False, ema=False, smooth=False):
        self.family = family
        self.parameters = parameters
        self.dual = dual
        self.synaptic = synaptic
        self.recurrent = recurrent
        self.ema = ema
        self.smooth = smooth
        self.branches = ("pos", "neg") if dual else ("",)
        self.states = {}
        for branch in self.branches:
            for trace, enabled in (("mem", True), ("syn", synaptic), ("rec", recurrent)):
                if enabled:
                    self.states[self._name(branch, trace)] = 0.0
        if family == "LIB":
            self.states["prev_output"] = 0.0
        elif family in ("LIT", "LITS"):
            self.states["pos_prev_output"] = 0.0
            self.states["neg_prev_output"] = 0.0

    @staticmethod
    def _name(branch, name):
        return f"{branch}_{name}" if branch else name

    @staticmethod
    def _signed(value, branch):
        if branch == "pos":
            return max(value, 0.0)
        if branch == "neg":
            return min(value, 0.0)
        return value

    def step(self, value):
        p = self.parameters
        previous = self.states.get("prev_output", 0.0)
        events = None
        reset = 0.0
        if self.family == "LIB":
            reset = previous * p["threshold"]
        elif self.family in ("LIT", "LITS"):
            events = {branch: self.states[f"{branch}_prev_output"] for branch in ("pos", "neg")}
            previous = sum(events.values())
            reset = events["pos"] * p["pos_threshold"] + events["neg"] * p["neg_threshold"]

        currents = {}
        if self.synaptic:
            for branch in self.branches:
                name = self._name(branch, "syn")
                alpha = p[self._name(branch, "alpha")]
                self.states[name] = alpha * self.states[name] + (1 - alpha) * self._signed(value, branch)
                currents[branch] = self.states[name]

        recurrences = {}
        if self.recurrent:
            for branch in self.branches:
                name = self._name(branch, "rec")
                gamma = p[self._name(branch, "gamma")]
                drive = events[branch] if events is not None and self.dual else self._signed(previous, branch)
                self.states[name] = gamma * self.states[name] + (1 - gamma) * drive
                recurrences[branch] = self.states[name]

        membrane = 0.0
        for branch in self.branches:
            name = self._name(branch, "mem")
            beta = p[self._name(branch, "beta")]
            drive = currents[branch] if self.synaptic else self._signed(value, branch)
            if self.ema:
                drive *= 1 - beta
            if self.recurrent:
                drive += recurrences[branch] * p[self._name(branch, "rec_weight")]
            reset_share = reset / len(self.branches)
            self.states[name] = beta * (self.states[name] - reset_share) + drive
            membrane += self.states[name]

        if self.family == "LI":
            return membrane
        if self.family == "LIB":
            event = self._fire(membrane - p["threshold"])
            self.states["prev_output"] = event
            return event
        else:
            positive = self._fire(membrane - p["pos_threshold"])
            negative = -self._fire(-p["neg_threshold"] - membrane)
        # Scaled outputs do not scale the reset or recurrent event history.
        self.states["pos_prev_output"] = positive
        self.states["neg_prev_output"] = negative
        if self.family == "LITS":
            return positive * p["pos_scale"] + negative * p["neg_scale"]
        return positive + negative

    def _fire(self, distance):
        return 1 / (1 + math.exp(-4 * distance)) if self.smooth else float(distance >= 0)
