require "set"

class Set
  def rsubset(n)
    entries.rsubset(n).to_set
  end
end

class Array
  def relement
    self[rand(size - 1)]
  end

  def rsubset(n)
    (size - n).times.inject(dup) do |xs,_|
      xs.delete_at(rand(xs.length)); xs
    end
  end
end

class Hash
  def except(keys)
    keys.inject(dup){|h,k| h.delete(k); h }
  end

  def slice(keys)
    keys.inject({}){|h,k| h[k] = self[k]; h }
  end
end

