require_relative "combine"

module Jackknife

  # Estimate a parameter by averaging multiple estimators, each on a subsample
  # having one observation removed.
  #
  # See: http://en.wikipedia.org/wiki/Jackknife_resampling
  #
  def self.estimate(observations)
    Combine.mean(observations.length.times.map do |n|
      yield observations[0, n].concat(observations[n+1 .. observations.length])
    end)
  end

  # Estimate the variance of an estimator
  def self.variance(observations)
    n   = observations.length.to_f
    all = yield(observations)

    # Multiple estimate by `n` to get back to SUM((xs[i] - all)^2), then
    # multiply by (n - 1)/n, which results in (n - 1) * estimate(...)
    (n - 1) * estimate(observations){|xs| (yield(xs) - all) ** 2 }
  end

  # Estimate the bias of an estimator over the entire sample
  def self.bias(observations)
    all    = yield(observations)
    jacked = estimate(observations){|xs| yield xs }

    observations.length * all - (observations.length - 1) * jacked
  end
end
