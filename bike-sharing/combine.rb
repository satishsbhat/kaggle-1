module Combine

  # Select the median label value (works only for numeric labels)
  def self.median(labels)
    if labels.length.odd?
      labels.sort[labels.length / 2]
    else
      mean(labels.sort[labels.length / 2 - 1, 2])
    end
  end

  # Select the mean label value (works only for numeric labels)
  def self.mean(labels)
    labels.inject(0, &:+) / labels.length.to_f
  end

  # Select the most common label (works for numeric and non-numeric labels)
  def self.mode(labels)
    labels.inject(Hash{|h,k| h[k] = 0}){|h,x| h[x] += 1; h }.max_by{|k,v| v }[0]
  end

  # Select all labels (works for numeric and non-numeric labels)
  def self.identity(labels)
    labels
  end
end
