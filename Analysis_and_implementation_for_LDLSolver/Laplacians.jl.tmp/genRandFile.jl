using Random

Random.seed!(12138)
randNums = rand(5000000) # 5 million random doubles

open("randomNums.bin", "w") do file
  write(file, randNums)
end