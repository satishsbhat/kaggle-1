#!/usr/bin/env ruby

require "set"
require_relative "pool"
require_relative "patch"

class Forest

  def self.bootstrap(repeat, observations, fraction = 1.00)
    size = fraction * observations.length

    repeat.times.each do
      yield size.to_i.times.map { observations.relement }
    end
  end

  def self.cart(measure, label, observations)
    pool = ForkPool.new(8)
    all  = observations.inject(Set.new){|s,o| s.merge(o.keys) }
    all -= [label]

    bootstrap(250, observations) do |samples|
      pool.enqueue do |n|
        $stderr.puts "Starting #{n}"

        # Select a random subset of features
        attrs  = all.rsubset(1 + rand(all.size)) | [label]
        subset = samples.map{|o| o.slice(attrs) }

        Tree.cart(measure, label, subset, -1)
      end
    end

    new(pool.collect)
  end

  def initialize(trees)
    @trees = trees
  end

  # use method = :mode for classification, and :mean or :median for regression
  def decide(observation, method = :mean)
    send(method, @trees.map{|t| t.decide(observation).send(method) })
  end

  def median(xs)
    if xs.length.odd?
      xs.sort[xs.length / 2]
    else
      mean(xs.sort[xs.length / 2 - 1, 2])
    end
  end

  def mean(xs)
    xs.inject(0, &:+) / xs.length.to_f
  end

  def mode(xs)
    xs.inject(Hash{|h,k| h[k] = 0}){|h,x| h[x] += 1; h }.max_by{|k,v| v }[0]
  end

end

