#!/usr/bin/env ruby

require "pp"
require "time"
require_relative "tree"
require_relative "forest"

class Hash
  def except(*keys)
    keys.inject(self.dup){|h,k| h.delete(k); h }
  end
end

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

def prepare(x)
  time = Time.parse(x["datetime"])

  return {
    mon_a:        %w(_ jan feb mar apr may jun jul aug sep oct nov dec)[time.month],
    mon_n:        time.month,
    mon_hod_a:    "#{time.month}-#{time.hour}",
    mon_hod_n:    time.month * 100 + time.hour,
    mon_dow_a:    "#{time.month}-#{time.wday}",
    mon_dow_n:    time.month * 100 + time.wday,
    dow_a:        %w(sun mon tue wed thu fri sat)[time.wday],
    dow_n:        time.wday,
    dow_hod_a:    "#{time.wday}-#{time.hour}",
    dow_hod_n:    time.wday * 100 + time.hour,
    hod_a:        time.hour.to_s,
    hod_n:        time.hour,
    year_a:       time.year.to_s,
    year_n:       time.year,
    year_mon:     "#{time.year}-#{time.month}",
    year_dow:     "#{time.year}-#{time.wday}",
    year_hod:     "#{time.year}-#{time.hour}",
    workday:      x["workingday"] != "0",
    season:       %w(_ spring summer autumn winter)[x["season"].to_i],
    holiday:      x["holiday"] != "0",
    weather:      %w(_ clear mist light heavy)[x["weather"].to_i],
    temperature:  x["temp"].to_f,
    feels_like:   x["atemp"].to_f,
    humidity:     x["humidity"].to_f,
    wind:         x["windspeed"].to_f,
    unregistered: x["casual"].to_i,
    registered:   x["registered"].to_i,
    total:        x["count"].to_i
  }
end

def prepare_(x)
  time = Time.parse("#{x["dteday"]} #{x["hr"]}:00:00")

  return {
    mon_a:        %w(_ jan feb mar apr may jun jul aug sep oct nov dec)[time.month],
    mon_n:        time.month,
    mon_hod_a:    "#{time.month}-#{time.hour}",
    mon_hod_n:    time.month * 100 + time.hour,
    mon_dow_a:    "#{time.month}-#{time.wday}",
    mon_dow_n:    time.month * 100 + time.wday,
    dow_a:        %w(sun mon tue wed thu fri sat)[time.wday],
    dow_n:        time.wday,
    dow_hod_a:    "#{time.wday}-#{time.hour}",
    dow_hod_n:    time.wday * 100 + time.hour,
    hod_a:        time.hour.to_s,
    hod_n:        time.hour,
    year_a:       time.year.to_s,
    year_n:       time.year,
    workday:      x["workingday"] != "0",
    season:       %w(_ spring summer autumn winter)[x["season"].to_i],
    holiday:      x["holiday"] != "0",
    weather:      %w(_ clear mist light heavy)[x["weathersit"].to_i],
    temperature:  x["temp"].to_f,
    feels_like:   x["atemp"].to_f,
    humidity:     x["hum"].to_f,
    wind:         x["windspeed"].to_f,
    unregistered: x["casual"].to_i,
    registered:   x["registered"].to_i,
    total:        x["cnt"].to_i
  }
end


def error(p, a)
  (Math.log(1 + p.to_i) - Math.log(1 + a.to_i)) ** 2
end

observations = []

csv(STDIN.tty? ? File.open(ARGV[0]) : STDIN){|x| observations << prepare(x) }

begin
  reg_tree = Marshal.load(File.read("reg.t"))
rescue
  reg_tree = Forest.cart(:variance, :registered, observations.map{|x| x.except(:unregistered, :total) })
  File.open("reg.t", "w"){|io| io.write(Marshal.dump(reg_tree)) }
end

begin
  unr_tree = Marshal.load(File.read("unr.t"))
rescue
  unr_tree = Forest.cart(:variance, :unregistered, observations.map{|x| x.except(:registered, :total) })
  File.open("unr.t", "w"){|io| io.write(Marshal.dump(unr_tree)) }
end

begin
  all_tree = Marshal.load(File.read("all.t"))
rescue
  all_tree = Forest.cart(:variance, :total, observations.map{|x| x.except(:registered, :unregistered) })
  File.open("all.t", "w"){|io| io.write(Marshal.dump(all_tree)) }
end

File.open("/Users/kputnam/Downloads/test.csv") do |io|
  csv(io) do |x|
    puts "%s,%d" %
      [x["datetime"], mean(prun.decide(prepare(x)).labels)]
  end
end

# File.open("hour.csv") do |io|
#   count        = 0.0
# 
#   sum_sep_tree = 0
#   sum_all_tree = 0
# 
#   csv(io) do |x|
#     y = prepare_(x)
#     next if y[:dom_n] <= 19
# 
#     # q = tree.decide(y).labels
#     # r = prun.decide(y).labels
# 
#     # p [y[:unregistered],
#     #     [mean(q),
#     #      mean(r),
#     #      median(q),
#     #      median(r)]]
# 
#     count += 1
# 
#     sum_sep_tree += error(y[:total], unr_tree.decide(y) + reg_tree.decide(y))
#     sum_all_tree += error(y[:total], all_tree.decide(y))
#   end
# 
#   puts "count: #{count}"
#   puts
#   puts "sep_tree:  #{Math.sqrt(sum_sep_tree / count)}"
#   puts "all_tree:  #{Math.sqrt(sum_all_tree / count)}"
# end
