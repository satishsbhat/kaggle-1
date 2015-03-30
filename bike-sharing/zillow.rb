#!/usr/bin/env ruby
require "nokogiri"
require "open-uri"

def download(city_or_zip, addresses)
  addresses.flat_map.with_index do |address, k|
    url =  "http://www.zillow.com/webservice/GetDeepSearchResults.htm?"
    url << "zws-id=%s&citystatezip=%s&address=%s" % [
      "X1-ZWz1chwxis15aj_9skq6",
      city_or_zip.strip.gsub(" ", "%20"),
      address.strip.gsub(" ", "%20")]

    $stderr.puts "Fetching #{k+1} of #{addresses.length}"
    xml = Nokogiri::XML(open(url)){|cfg| cfg.noblanks.noent.strict.nonet }

    # Zero means success, otherwise an error
    unless xml.at_css("code").text == "0"
      $stderr.puts "... FAILED"
      next []
    end

    xml.css("results result").map do |house|
      { zip:   house.css("zipcode").text,
        use:   house.css("useCode").text, # single family, etc
        year:  house.css("yearBuilt").text.to_i,
        baths: house.css("bathrooms").text.to_f,
        beds:  house.css("bedrooms").text.to_f,
        rooms: house.css("totalRooms").text.to_f,
      # sqft:  house.css("finishedSqFt").text.to_f,
        price: house.css("zestimate amount").text.to_f }
    end
  end
end

if __FILE__ == $0
  require_relative "cart"

  # city_or_zip = ARGV[0]
  # addresses   = STDIN.each_line.map(&:strip)
  # training = download(city_or_zip, addresses)
  # training.reject!{|o| o[:price] == 0 }

  # unless training.empty?
  #   File.open("training.bin", "w+"){|io| io.write(Marshal.dump(training)) }
  # end

  training = File.open("training.bin"){|io| Marshal.load(io) }
  tree     = CART.regression(:price, training)

  pp tree
end
