#!/usr/bin/env ruby

class Array
  def random
    self[rand(size - 1)]
  end
end

class Pool
  def initialize(count)
    @count, @queue = count, Queue.new
  end

  def enqueue(&block)
    @queue.push(block)
  end

  def collect
    count   = 0
    results = []

    @count.times.map do
      Thread.new do
        begin
          while work = @queue.pop(true)
            results.push(work.call(count += 1))
          end
        rescue ThreadError
        end
      end
    end.map(&:join)

    results
  end
end

class Fork
  def initialize(count)
    @pool = Pool.new(count)
  end

  def enqueue(&block)
    @pool.enqueue do |count|
      pr, pw = IO.pipe
      pid    = fork do
        # Child
        pr.close
        pw.write(Marshal.dump(block.call(count)))
        pw.close
      end

      # Parent
      pw.close
      result = Marshal.load(pr.read)
      pr.close

      Process.wait(pid)
      result
    end
  end

  def collect
    @pool.collect
  end
end

class Forest

  def self.sample(repeat, fraction, observations)
    size = fraction * observations.length

    repeat.times.each do
      yield size.to_i.times.map { observations.random }
    end
  end

  def self.cart(measure, label, observations)
    pool = Fork.new(4)

    sample(250, 0.50, observations) do |subset|
      pool.enqueue do |n|
        $stderr.puts "Starting #{n}"
        tree = Tree.cart(measure, label, subset, 5)
        tree.prune(measure, tree.score * 0.01)
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

