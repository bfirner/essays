#!/usr/bin/gnuplot

set yrange [0:3]
set xrange [0:3]

unset xtics
unset ytics

set terminal pngcairo enhanced size 480, 280 fontscale 1.3 linewidth 1.5

###############
# Live examples
###############

set output "gol_live_examples.png"

set multiplot layout 1,2

set size square

set xlabel "Center Cell\nStays Alive"

# Draw three "live" cells
set object 1 rect from 0,1 to 1,2 fc rgb "black"
set object 2 rect from 1,2 to 2,3 fc rgb "black"
set object 3 rect from 1,1 to 2,2 fc rgb "black"

set label "0" at 0.5,0.5 center font ",20" textcolor rgb "black"
set label "0" at 1.5,0.5 center font ",20" textcolor rgb "black"
set label "0" at 2.5,0.5 center font ",20" textcolor rgb "black"

set label "1" at 0.5,1.5 center font ",20" textcolor rgb "white"
set label "1" at 1.5,1.5 center font ",20" textcolor rgb "white"
set label "0" at 2.5,1.5 center font ",20" textcolor rgb "black"

set label "0" at 0.5,2.5 center font ",20" textcolor rgb "black"
set label "1" at 1.5,2.5 center font ",20" textcolor rgb "white"
set label "0" at 2.5,2.5 center font ",20" textcolor rgb "black"

plot 0 notitle

set xlabel "Center Cell\nBecomes Alive"
unset label
unset object

# Draw three "live" cells
set object 1 rect from 1,2 to 2,3 fc rgb "black"
set object 2 rect from 1,0 to 2,1 fc rgb "black"
set object 3 rect from 2,1 to 3,2 fc rgb "black"

set label "0" at 0.5,0.5 center font ",20" textcolor rgb "black"
set label "1" at 1.5,0.5 center font ",20" textcolor rgb "white"
set label "0" at 2.5,0.5 center font ",20" textcolor rgb "black"

set label "0" at 0.5,1.5 center font ",20" textcolor rgb "black"
set label "0" at 1.5,1.5 center font ",20" textcolor rgb "black"
set label "1" at 2.5,1.5 center font ",20" textcolor rgb "white"

set label "0" at 0.5,2.5 center font ",20" textcolor rgb "black"
set label "1" at 1.5,2.5 center font ",20" textcolor rgb "white"
set label "0" at 2.5,2.5 center font ",20" textcolor rgb "black"

plot 0 notitle

unset multiplot

###############
# Dead examples
###############

set output "gol_dead_examples.png"

set multiplot layout 1,2

set size square

set xlabel "Center Cell\nStays Dead"
unset label
unset object

# Draw two "live" cells
set object 1 rect from 0,1 to 1,2 fc rgb "black"
set object 3 rect from 1,2 to 2,3 fc rgb "black"

set label "0" at 0.5,0.5 center font ",20" textcolor rgb "black"
set label "0" at 1.5,0.5 center font ",20" textcolor rgb "black"
set label "0" at 2.5,0.5 center font ",20" textcolor rgb "black"

set label "1" at 0.5,1.5 center font ",20" textcolor rgb "white"
set label "0" at 1.5,1.5 center font ",20" textcolor rgb "black"
set label "0" at 2.5,1.5 center font ",20" textcolor rgb "black"

set label "0" at 0.5,2.5 center font ",20" textcolor rgb "black"
set label "1" at 1.5,2.5 center font ",20" textcolor rgb "white"
set label "0" at 2.5,2.5 center font ",20" textcolor rgb "black"

plot 0 notitle

set xlabel "Center Cell\nBecomes Dead"
unset label
unset object

# Draw five "live" cells
set object 1 rect from 1,0 to 2,1 fc rgb "black"
set object 2 rect from 0,1 to 1,2 fc rgb "black"
set object 3 rect from 1,1 to 2,2 fc rgb "black"
set object 4 rect from 2,1 to 3,2 fc rgb "black"
set object 5 rect from 1,2 to 2,3 fc rgb "black"

set label "0" at 0.5,0.5 center font ",20" textcolor rgb "black"
set label "1" at 1.5,0.5 center font ",20" textcolor rgb "white"
set label "0" at 2.5,0.5 center font ",20" textcolor rgb "black"

set label "1" at 0.5,1.5 center font ",20" textcolor rgb "white"
set label "1" at 1.5,1.5 center font ",20" textcolor rgb "white"
set label "1" at 2.5,1.5 center font ",20" textcolor rgb "white"

set label "0" at 0.5,2.5 center font ",20" textcolor rgb "black"
set label "1" at 1.5,2.5 center font ",20" textcolor rgb "white"
set label "0" at 2.5,2.5 center font ",20" textcolor rgb "black"

plot 0 notitle

unset multiplot
