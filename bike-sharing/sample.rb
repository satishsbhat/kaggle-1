
class Array
  def sample_with_replacement(size)
    size.times.map { at(rand(size)) }
  end

  def sample_without_replacement(size)
    if size > length or size < 0
      raise "Nope"
    elsif size == length
      self
    elsif size < length/2
      size.times.inject([[], dup]){|(ss,xs),_| ss.push(xs.delete_at(rand(xs.size))); [ss, xs] }[0]
    else
      (length - size).times.inject(dup){|xs,_| xs.delete_at(rand(xs.size)); xs }
    end
  end
end
