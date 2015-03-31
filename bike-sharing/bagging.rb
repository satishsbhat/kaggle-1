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
  def self.bootstrap(count, label, observations)
    # Build models in parallel
    pool = ForkPool.new(8)

    count.times do
      pool.enqueue do |n|
        $stderr.puts "Starting #{n}"

        # Select a random subset of observations (bootstrap sample)
        yield observations.sample_with_replacement(observations.size)
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
