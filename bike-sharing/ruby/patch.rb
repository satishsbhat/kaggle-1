require "set"

class Hash
  def except(keys)
    keys.inject(dup){|h,k| h.delete(k); h }
  end

  def slice(keys)
    keys.inject({}){|h,k| h[k] = self[k]; h }
  end
end

