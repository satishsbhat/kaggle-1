#!/usr/bin/env ruby
require "pp"
require "set"

# Variables
# - nominal
# - ordinal
# - interval
# - ratio
#
# prune
# - voting: Bag   <> Bag
# - mean:   Gauss <> Gauss
# - median: TODO
#
# predict
# - voting: Bag (mode)
# - mean:   Gauss
# - median: TODO

class Tree
  def self.cart(measure, label, observations, depth = -1)
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

    # Compute the gain of dividing observations by all feature/threshold
    gains = attrs.flat_map do |feature, (values, op)|
      values.map do |threshold|
        ts, fs = observations.partition{|o| o[feature].send(op, threshold) }
        ratio  = ts.length / count

        # Measure improvement of dividing observations by feature/threshold
        gain = score - ratio * send(measure, ts.map{|o| o[label] }) -
                 (1 - ratio) * send(measure, fs.map{|o| o[label] })

        [gain, feature, op, threshold, ts, fs]
      end
    end

    # Find the best division
    gain, feature, op, threshold, ts, fs = gains.max_by{|g,_,_,_,_| g }

    if gain and gain <= 0
      Leaf.new(score, observations.map{|o| o[label] })
    else
      Node.new(score, feature, op, threshold,
        Tree.cart(measure, label, ts, depth - 1),
        Tree.cart(measure, label, fs, depth - 1))
    end
  end

  def self.count(labels)
    labels.inject(Hash.new{|h,k| h[k] = 0 }) do |h,x|
      h[x] += 1; h
    end
  end

  def self.gini(labels)
    counts = count(labels)
    total  = counts.values.inject(0, &:+).to_f
    freqs  = counts.values.map{|v| v/total }

    freqs.permutation(2).inject(0) do |sum, (a, b)|
      sum + a*b
    end
  end

  def self.entropy(labels)
    counts = count(labels)
    total  = counts.values.inject(0, &:+).to_f
    freqs  = counts.values.map{|x| x / total }

    freqs.inject(0) do |sum, p|
      sum - p*Math.log2(p)
    end
  end

  def self.variance(labels)
    count = labels.length.to_f
    return 0 if count == 0

    mean  = labels.inject(0, &:+) / count
    labels.inject(0){|sum, x| sum + (x - mean) ** 2 } / count
  end
end

class Node
  attr_reader :t, :f, :score

  def initialize(score, feature, op, threshold, t, f)
    @score, @feature, @op, @threshold, @t, @f =
      score, feature, op, threshold, t, f
  end

  def decide(observation)
    observation[@feature].send(@op, @threshold) ?
      @t.decide(observation) :
      @f.decide(observation)
  end

  def prune(measure, threshold)
    pruned = Node.new(@score, @feature, @op, @threshold,
      t = @t.prune(measure, threshold),
      f = @f.prune(measure, threshold))

    return pruned unless Leaf === t and Leaf === f

    before  = (t.score + f.score) / 2.0

    # What if we merged the two leaves into a big leaf?
    labels  = t.labels + f.labels
    after   = Tree.send(measure, labels)

    if after - before < threshold
      Leaf.new(after, labels)
    else
      pruned
    end
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

  def decide(observation)
    self
  end

  def prune(measure, threshold)
    self
  end

  def inspect(indent = "")
    @labels.inspect
  end

  def median
    if @labels.length.odd?
      @labels.sort[@labels.length / 2]
    else
      mean(@labels.sort[@labels.length / 2 - 1, 2])
    end
  end

  def mean
    @labels.inject(0, &:+) / @labels.length.to_f
  end

  def mode
    @labels.inject(Hash{|h,k| h[k] = 0}){|h,x| h[x] += 1; h }.max_by{|k,v| v }[0]
  end
end

