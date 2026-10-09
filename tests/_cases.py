"""Explicit public layer inventory shared by parameterized contract tests."""

from tracetorch import rnn, snn


SNN_NAMES = (
    "LI", "DLI", "SLI", "DSLI", "LIEMA", "DLIEMA", "SLIEMA", "DSLIEMA",
    "LIB", "DLIB", "SLIB", "RLIB", "DSLIB", "DRLIB", "SRLIB", "DSRLIB",
    "LIT", "DLIT", "SLIT", "RLIT", "DSLIT", "DRLIT", "SRLIT", "DSRLIT",
    "LITS", "DLITS", "SLITS", "RLITS", "DSLITS", "DRLITS", "SRLITS", "DSRLITS",
)
SNN_CLASSES = tuple(getattr(snn, name) for name in SNN_NAMES)
RNN_CLASSES = (rnn.SimpleRNN, rnn.GRU, rnn.LSTM)
