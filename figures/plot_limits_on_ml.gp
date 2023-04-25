#!/usr/bin/gnuplot

set yrange [0:1]
set xrange [0:1]

set ylabel "error"

set terminal pngcairo enhanced size 480, 280 fontscale 1.3 linewidth 1.5

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

# Fourth plat: car
unset xlabel
unset ylabel
unset ytics
unset xtics
unset border

set output "car_fov.png"

# Draw a "car"
carwidth=0.25
carheight=0.15
x1 = 0.0
x2 = carwidth
y1 = 0.5-(carheight/2)
y2 = 0.5+(carheight/2)
set arrow from x1,y1 to x2,y1 nohead dashtype 1 lc "black"
set arrow from x1,y1 to x1,y2 nohead dashtype 1 lc "black"
set arrow from x2,y2 to x1,y2 nohead dashtype 1 lc "black"
set arrow from x2,y2 to x2,y1 nohead dashtype 1 lc "black"
set label at carwidth/3,0.5 "Car"

# Triangular field of view
set arrow from carwidth,0.5 to 0.9,1 nohead dashtype 1 lc "black"
set arrow from carwidth,0.5 to 0.9,0 nohead dashtype 1 lc "black"
set arrow from 0.9,1 to 0.9,0 nohead dashtype 1 lc "black"
set label at carwidth+0.05,0.5 "FOV"

# Labels for 'd' and 'width'
set arrow from carwidth+0.01,0.5 to 0.89,0.5 heads size 0.04,90 dashtype 2 lc "gray"
set label at (1+carwidth)/2,0.55 "d"

set arrow from 0.95,1 to 0.95,0 heads size 0.04,90 dashtype 2 lc "gray"
set label at 0.97,0.5 "w"

plot -10 notitle
