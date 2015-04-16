class ThreadPool
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

class ForkPool
  def initialize(count)
    @pool = ThreadPool.new(count)
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

