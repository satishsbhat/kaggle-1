#!/usr/bin/env ruby

require "pp"
require "time"
require_relative "cart"
require_relative "patch"
require_relative "bagging"
require_relative "boosting"

def csv(io)
  names = nil

  io.each_line do |line|
    if names.nil?
      names = line.strip.split(",")
    else
      yield Hash[names.zip(line.strip.split(","))]
    end
  end
end

def time_features(time)
  # Note day of month is NOT included as a feature, because we know the
  # training set doesn't include instances of 20-31 and a decision tree
  # won't correctly generalize without those instances.

  return {
    mon_a:          %w(_ jan feb mar apr may jun jul aug sep oct nov dec)[time.month],
    mon_n:          time.month,                                   # 12
    mon_hod_a:      [time.month, time.hour].join("-"),            # 288 = 12 * 24
  # mon_hod_n:      time.month * 100 + time.hour,                 # 288 = 12 * 24
    mon_dow_a:      [time.month, time.wday].join("-"),            # 84 = 12 * 7
  # mon_dow_n:      time.month * 100 + time.wday,                 # 84 = 12 * 7
    dow_a:          %w(sun mon tue wed thu fri sat)[time.wday],   # 7
  # dow_n:          time.wday,                                    # 7
    dow_hod_a:      [time.wday, time.hour].join("-"),             # 168 = 24 * 7
  # dow_hod_n:      time.wday * 100 + time.hour,                  # 168 = 24 * 7
    hod_a:          time.hour.to_s,                               # 24
    hod_n:          time.hour,                                    # 24
    year_a:         time.year.to_s,                               # 2
    year_n:         time.year,                                    # 2
    year_mon:       [time.year, time.month].join("-"),            # 24 = 2 * 12
    year_dow:       [time.year, time.wday].join("-"),             # 14 = 2 * 14
    year_hod:       [time.year, time.hour].join("-"),             # 48 = 2 * 24
  # year_mon_hod_a: [time.year, time.month, time.hour].join("-"), # 576 = 2 * 12 * 24
    year_mon_dow_a: [time.year, time.month, time.wday].join("-")  # 168 = 2 * 12 * 7,
  }
end

def features(x)
  time = Time.parse(x["datetime"])

  return time_features(time).update \
    workday:      x["workingday"] != "0",                              # 2
    season:       %w(_ spring summer autumn winter)[x["season"].to_i], # 4 -- this is equivalent to (mon / 4 + 1)
    holiday:      x["holiday"] != "0",                                 # 2
    weather:      %w(_ clear mist light heavy)[x["weather"].to_i],     # 4
    temperature:  x["temp"].to_i,                                      # continuous
    feels_like:   x["atemp"].to_i,                                     # continuous
    humidity:     x["humidity"].to_i,                                  # continuous
    wind:         x["windspeed"].to_i,                                 # continuous
    sin_month:    3*Math.sin((time.month-1)/2.0 * Math::PI/12 - Math::PI/8.0),
    work_sin_hr:  x["workingday"] == "0" ? -Math.sin(time.hour * (Math::PI/12) + Math::PI/4) : 0,
    unwrk_sin_hr: x["workingday"] == "1" ? 2*Math.sin(time.hour * (Math::PI/12) + Math::PI/4)
                                         + 3*Math.sin(2 * time.hour * (Math::PI/12) + Math::PI*3/4) : 0,

    unregistered: x["casual"].to_i,                                    # label
    registered:   x["registered"].to_i,                                # label
    total:        x["count"].to_i                                      # label
end

def features__(x)
  time = Time.parse("#{x["dteday"]} #{x["hr"]}:00:00")

  return time_features(time).update \
    dom_n:        time.day,
    workday:      x["workingday"] != "0",
    season:       %w(_ spring summer autumn winter)[x["season"].to_i],
    holiday:      x["holiday"] != "0",
    weather:      %w(_ clear mist light heavy)[x["weathersit"].to_i],
    temperature:  x["temp"].to_i,
    feels_like:   x["atemp"].to_i,
    humidity:     x["hum"].to_i,
    wind:         x["windspeed"].to_i,

    unregistered: x["casual"].to_i,
    registered:   x["registered"].to_i,
    total:        x["cnt"].to_i
end

# Train two random forest models (or read them from disk)
def train(observations)
  registered =
    begin
      Marshal.load(File.read("registered.t"))
    rescue
      $stderr.puts "Training model for 'registered'"

      Bagging.bootstrap(150, :registered, observations.map{|x| x.except([:unregistered, :total]) }) do |sample|
        CART.regression(:registered, sample, -1, 5)
      end.tap do |ensemble|
        File.open("registered.t", "w"){|io| io.write(Marshal.dump(ensemble)) }
      end

      # Boosting.stochastic(50, 0.65, :registered, observations.map{|x| x.except([:unregistered, :total]) }) do |sample|
      #   CART.regression(:registered, sample, 3)
      # end.tap do |ensemble|
      #   File.open("registered.t", "w"){|io| io.write(Marshal.dump(ensemble)) }
      # end
    end

  unregistered =
    begin
      Marshal.load(File.read("unregistered.t"))
    rescue
      $stderr.puts "Training model for 'unregistered'"

      Bagging.bootstrap(150, :unregistered, observations.map{|x| x.except([:registered, :total]) }) do |sample|
        CART.regression(:unregistered, sample, -1, 5)
      end.tap do |ensemble|
        File.open("unregistered.t", "w"){|io| io.write(Marshal.dump(ensemble)) }
      end

      # Boosting.stochastic(50, 0.65, :unregistered, observations.map{|x| x.except([:registered, :total]) }) do |sample|
      #   CART.regression(:unregistered, sample, 3)
      # end.tap do |ensemble|
      #   File.open("unregistered.t", "w"){|io| io.write(Marshal.dump(ensemble)) }
      # end
    end

  [registered, unregistered]
end

# Make predictions (on test data set)
def predict(registered, unregistered, io)
  $stdout.puts "datetime,count"

  csv(io) do |x|
    y = features(x)
    puts "%s,%d" % [x["datetime"], registered.predict(y, :mean) + unregistered.predict(y, :mean)]
  end
end

# Compute the error of a single prediction
def error(actual, predicted)
  (Math.log(1 + predicted.abs.to_i) - Math.log(1 + actual.to_i)) ** 2
end

# Evaluate error (using whole data set)
def evaluate(registered, unregistered, io)
  count = 0.0
  total = 0
  
  csv(io) do |x|
    y = features__(x)

    # Ignore data from training set
    next if y[:dom_n] <= 19
  
    count += 1
    total += error(y[:total], registered.predict(y, :mean) + unregistered.predict(y, :mean))
  end
  
  $stderr.puts "error:  #{Math.sqrt(total / count)}"
end

if __FILE__ == $0
  observations = []
  csv(STDIN.tty? ? File.open(ARGV[0]) : STDIN){|x| observations << features(x) }

  registered, unregistered = train(observations)

  File.open("data/test.csv"){|io| predict(registered, unregistered, io)  }
  File.open("data/hour.csv"){|io| evaluate(registered, unregistered, io) }
end
