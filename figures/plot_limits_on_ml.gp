#!/usr/bin/gnuplot

set yrange [0:1]
set xrange [0:1]

set ylabel "error"

set terminal pngcairo enhanced size 360, 220

# We will be plotting synthetic lines, use enough samples to make them look okay.
set samples 100
unset xtics
unset ytics
set xtics (0)
set ytics (0)

# First plot: error as a function of capacity
set xlabel "capacity (h)"

set output "error_vs_capacity.png"
plot 0.25 + (x-0.5)**2 - (x-0.5)**3 w lines title "testing error" dashtype 1 lc "black",\
     0.625 - 0.625*(1 - 0.01**x) w lines title "training error" dashtype 2 lc "black"

# Second plot: error as a function of training set size
set xlabel "training set size (t)"
set arrow from 0,0.3 to 1,0.3 nohead dashtype 4 lc "gray"
set label at 0.01,0.4 "E_{t→∞}"

set output "error_vs_datasize.png"
plot 0.6 - 0.3*(1 - 0.01**x) w lines title "testing error" dashtype 1 lc "black",\
     x < 0.05 ? 0 : 0.3*(1 - 0.02**x) w lines title "training error" dashtype 2 lc "black"

unset arrow
unset label

# Third plot: error as a function of noise
set xlabel "capacity (h)"
set arrow from 0,0.3 to 1,0.3 nohead dashtype 4 lc "black"
set label at 0.01,0.4 "E_{t→∞}"
set label at 0.15,0.15 "Intrinsic Noise Level"

set output "error_vs_noise.png"
plot 0.6 - 0.3*(1 - 0.01**x) w lines title "testing error" dashtype 1 lc "black"
unset arrow
unset label
