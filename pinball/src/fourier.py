import numpy, itertools

class TrivialBasis:
    """Uses the features themselves as a basis. However, does a little bit of basic manipulation
    to make things more reasonable. Specifically, this allows (defaults to) rescaling to be in the
    range [-1, +1].
    """

    def __init__(self, nvars, ranges):
        self.numTerms = nvars
        self.high_ranges = ranges.high
        self.low_ranges = ranges.low

    def scale(self, value, pos):
        if self.low_ranges[pos] == self.high_ranges[pos]:
            return 0.0
        else:
            return (value - self.low_ranges[pos]) / (self.high_ranges[pos] - self.low_ranges[pos])

    def getNumBasisFunctions(self):
        return int(self.numTerms)

    def __call__(self, features):
        if len(features) == 0:
            return numpy.ones((1,))
        return (numpy.array([self.scale(features[i],i) for i in range(len(features))]) - 0.5)*2.


class FourierBasis(TrivialBasis):
    """Fourier Basis linear function approximation. Requires the ranges for each dimension, and is thus able to
    use only sine or cosine (and uses cosine). So, this has half the coefficients that a full Fourier approximation
    would use.
    From the paper:
    G.D. Konidaris, S. Osentoski and P.S. Thomas.
    Value Function Approximation in Reinforcement Learning using the Fourier Basis.
    In Proceedings of the Twenty-Fifth Conference on Artificial Intelligence, pages 380-385, August 2011.
    """

    def __init__(self, nvars, ranges, order=3):
        nterms = pow(order + 1.0, nvars)
        self.numTerms = int(nterms)
        self.order = order
        self.low_ranges = ranges.low
        self.high_ranges = ranges.high
        iter = itertools.product(range(order+1), repeat=nvars)
        self.multipliers = numpy.array([list(map(int,x)) for x in iter]) # int[256][4]

    def __call__(self, features):
        """
        Arguments:
            features {[type]} -- [description]
        Returns:
            int[nvars**nvars][nvars]
        """
        if len(features) == 0:
            return numpy.ones((1,))
        basisFeatures = numpy.array([self.scale(features[i],i) for i in range(len(features))])
        return numpy.cos(numpy.pi * numpy.dot(self.multipliers, basisFeatures)) 

    def __len__(self):
        return self.numTerms

if __name__ == "__main__":
    features = FourierBasis(1, ((0,4),))
    print(features([3,]))
