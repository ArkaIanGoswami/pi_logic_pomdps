"""Automata as dictionaries."""

def make_dfa(states, initial, accepting, rejecting, step, formula):
    accepting, rejecting = frozenset(accepting), frozenset(rejecting)

    def nxt(q, labels):
        if q in accepting or q in rejecting:
            return q
        return step(q, frozenset(labels))

    return dict(states=tuple(states), initial=initial, accepting=accepting,
                rejecting=rejecting, next=nxt, formula=formula)


def trivial_dfa():
    return make_dfa(('q0',), 0, (), (), lambda q, labels: 0, 'none')


def is_accepting(dfa, q):
    return q in dfa['accepting']


def is_rejecting(dfa, q):
    return q in dfa['rejecting']
