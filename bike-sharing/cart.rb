#!/usr/bin/env ruby
require "pp"
require "set"
require_relative "combine"

# Variables
# - nominal (no ordering)
# - ordinal (has ordering, no distances)
# - interval (has ordering and distances equally spaced)
# - ratio (has ordering and distances and a zero)
#
# merging leaves
# - voting: Bag   <> Bag
# - mean:   Gauss <> Gauss
# - median: TODO
#
# predict
# - mode:   Bag
# - mean:   Gauss
# - median: TODO

module CART
  def self.regression(label, observations, depth = -1)
    train(:variance, label, observations, depth)
  end

  def self.classify(label, observations, depth = -1)
    train(:entropy, label, observations, depth)
  end

  def self.train(measure, label, observations, depth = -1)
    score = send(measure, observations.map{|o| o[label] })
    count = observations.size.to_f

    return Leaf.new(score, observations.map{|o| o[label] }) if depth.zero?

    # Build set of unique values (and comparison operator) per column
    attrs = observations.inject({}) do |sum, o|
      o.each do |k, v|
        next if k == label
        sum[k] ||= [Set.new, Numeric === v ? :< : :==]
        sum[k][0].add(v)
      end; sum
    end

    # Remember the best gain, feature, test op, threshold, and t/f partitions
    best = [0, nil, nil, nil, [], []]

    # Compute the gain from partitioning observations by each feature/threshold pair
    attrs.each do |feature, (values, op)|
      values.each do |threshold|
        ts, fs = observations.partition{|o| o[feature].send(op, threshold) }
        ratio  = ts.length / count

        # Measure improvement of dividing observations by feature/threshold
        gain = score - ratio * send(measure, ts.map{|o| o[label] }) -
                 (1 - ratio) * send(measure, fs.map{|o| o[label] })

        # Remember the best gain, feature, test op, threshold, and t/f partitions
        best = [gain, feature, op, threshold, ts, fs] if gain > best[0]
      end
    end

    gain, feature, op, threshold, ts, fs = best

    if gain <= 0
      Leaf.new(score, observations.map{|o| o[label] })
    else
      Node.new(feature, op, threshold,
        CART.train(measure, label, ts, depth - 1),
        CART.train(measure, label, fs, depth - 1))
    end
  end

private

  # Count how many times each label occurs
  def self.count(labels)
    labels.inject(Hash.new{|h,k| h[k] = 0 }) do |h,x|
      h[x] += 1; h
    end
  end

  # Compute the gini impurity for a given set of labels
  def self.gini(labels)
    counts = count(labels)
    total  = counts.values.inject(0, &:+).to_f
    freqs  = counts.values.map{|v| v/total }

    freqs.permutation(2).inject(0){|sum, (a, b)| sum + a*b }
  end

  # Compute the entropy for a given set of labels
  def self.entropy(labels)
    counts = count(labels)
    total  = counts.values.inject(0, &:+).to_f
    freqs  = counts.values.map{|x| x / total }

    freqs.inject(0){|sum, p| sum - p*Math.log2(p) }
  end

  # Compute the variance (sum of squared differences from the mean) for a given set of labels
  def self.variance(labels)
    count = labels.length.to_f
    mean  = labels.inject(0, &:+) / count
    labels.inject(0){|sum, x| sum + (x - mean) ** 2 } / count
  end
end

class Node
  attr_reader :t, :f

  def initialize(feature, op, threshold, t, f)
    @feature, @op, @threshold, @t, @f =
      feature, op, threshold, t, f
  end

  # Predict the label for a given observation. Combine multiple labels in
  # a leaf using `method` (one of :mean, :median, :mode, or :identity)
  def predict(observation, method = :identity)
    observation[@feature].send(@op, @threshold) ?
      @t.predict(observation, method) :
      @f.predict(observation, method)
  end

  # Reduce the tree from the bottom up, combining leaves in the tree if the resulting
  # increase in score (according to measure, which is one of :entropy, :gini, or
  # :variance) remains below the threshold
  def prune(measure, threshold)
    pruned = Node.new(@feature, @op, @threshold,
      t = @t.prune(measure, threshold),
      f = @f.prune(measure, threshold))

    return pruned unless Leaf === t and Leaf === f

    before  = (t.score + f.score) / 2.0

    # What if we merged the two leaves into one big leaf?
    labels  = t.labels + f.labels
    after   = CART.send(measure, labels)

    if after - before < threshold
      Leaf.new(after, labels)
    else
      pruned
    end
  end

  # Number of leaves and nodes in the tree
  def size
    1 + @t.size + @f.size
  end

  def inspect(indent = "")
    "IF #{@feature} #{@op} #{@threshold}\n" <<
    "#{indent}THEN #{@t.inspect(indent + "     ")}\n" <<
    "#{indent}ELSE #{@f.inspect(indent + "     ")}"
  end
end

class Leaf
  attr_reader :score, :labels

  def initialize(score, labels)
    @score, @labels = score, labels
  end

  # Predict the label for a given observation. Combine multiple labels in
  # a leaf using `method` (one of :mean, :median, :mode, or :identity)
  def predict(observation, method = :identity)
    Combine.send(method, @labels)
  end

  # Reduce the tree from the bottom up, combining leaves in the tree if
  # the resulting increase in score remains below the threshold
  def prune(measure, threshold)
    self
  end

  # Number of leaves and nodes in the tree
  def size
    1
  end

  def inspect(indent = "")
    @labels.inspect
  end
end
