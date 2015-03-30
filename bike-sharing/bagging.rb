require_relative "pool"
require_relative "patch"
require_relative "sample"
require_relative "combine"

  # Crawford (1989) examines using bootstrapping to estimate the
  # misclassification rates needed for the selection of the right
  # sized tree

class Bagging

  # Build a collection of classifiers using bootstrap aggregation
  # - count:        number of classifiers to build on randomly-selected samples
  # - width:        number of randomly-selected features to include in each classifier
  # - observations: training set of labeled observations (Array[Hash[K, V]])
  #
  # Example:
  #
  #   Bagging.bootstrap(250, 10, observations) do |sample|
  #     # Predict numeric 'count' by minimizing variance
  #     CART.train(:variance, :count, sample)
  #   end
  #
  def self.bootstrap(count, width, label, observations)
    # Build models in parallel
    pool = ForkPool.new(8)

    # Possible features to select
    candidates  = observations.inject(Set.new){|s,o| s.merge(o.keys) }.entries
    candidates -= [label]

    count.times do
      pool.enqueue do |n|
        $stderr.puts "Starting #{n}"

        # Select a random subset of features from a random bootstrap sample of observations
        features = candidates.sample_without_replacement(width) | [label]
        selected = observations.sample_with_replacement(observations.size).map{|o| o.slice(features) }

        yield selected
      end
    end

    new(pool.collect)
  end

  def initialize(models)
    @models = models
  end

  # Combine predictions of each model using `method` (:mean, :median, :mode, or :identity)
  def predict(observation, method)
    Combine.send(method, @models.map{|t| t.predict(observation, method) })
  end

  def inspect
    "#<Bagging:... (ensemble of #{@models.size} models)>"
  end
end
