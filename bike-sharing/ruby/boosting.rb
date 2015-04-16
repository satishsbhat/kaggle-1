class Boosting

  #
  # Example:
  #
  #   Boosting.stochastic(100, 0.5, :count, observations) do |sample|
  #     CART.regression(:count, sample, 3)
  #   end
  #
  def self.stochastic(iterations, fraction, label, observations)
    initial = yield(observations)
    models  = [initial]
    subsize = (fraction * observations.length).to_i

    iterations.times.map do |n|
      $stderr.puts "Starting #{n}"
      previous = models.last

      # Target labels for the next model are the error in the previous one
      observations = observations.map{|o| o.merge(label => o[label] - previous.predict(o)) }

      # Select a random sample of observations
      sample = observations.sample_without_replacement(subsize)
      #stderr.puts sample.map{|o| "%02.2f" % o[label] }.join(", ")

      # Yield the training set and expect a traind model in return
      yield(sample).tap{|h| models << h }
    end

    new(models)
  end

  def initialize(models)
    @models = models
  end

  def predict(observation)
    @models.inject(0){|sum, h| sum + h.predict(observation) }
  end
end

class Dumb
  def self.train(observations, label)
    new(Combine.mean(observations.map{|o| o[label] }))
  end

  def initialize(mean)
    @mean = mean
  end

  def predict(observation)
    @mean
  end
end

